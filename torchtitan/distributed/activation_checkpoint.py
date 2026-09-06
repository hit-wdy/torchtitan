# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import os
from dataclasses import dataclass, field
from typing import Annotated, cast

import torch
import torch._functorch.config
import torch.nn as nn
import tyro
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.config import Configurable
from torchtitan.tools.logging import logger


def _get_default_save_ops() -> set:
    """Returns the default set of ops whose activations should be saved
    (compute + comm).

    Each op spec is either an op object (always included) or a tuple
    (root, dotted_path) for conditionally available ops — resolved via
    getattr and silently skipped if not registered.
    不值得重算的算子，SelectiveAC 通常会保存它们的输出，避免反向阶段再次执行。
    """
    # Ops whose outputs are expensive to recompute (matmuls, attention, etc.)
    compute_ops = [
        # SDPA variants
        # 原因：Attention 计算量大
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        # For low precision training, always save the absolute maximum used
        # to compute the scaling factor for quantization.
        # 低精度训练中常用最大绝对值计算量化 scale。重新计算可能影响量化路径或增加开销，所以默认保存。
        torch.ops.aten.max.default,
        # FlexAttention (torch.ops.higher_order.flex_attention is the same object)
        torch._higher_order_ops.flex_attention,
        torch.ops.aten.linear.default,
        # topk can be non-deterministic; save to keep MoE expert assignments
        # stable between forward and recompute.
        # topk 决定 token 被发送到哪些专家。它可能存在非确定性，因此保存它的结果保证确定性
        torch.ops.aten.topk.default,

        # 编译和可选后端算子没有直接访问属性，而是保存为：(root, "a.b.c")，因为这些算子不一定存在
        # Inductor compiled code (available when torch.compile is used)
        (torch._higher_order_ops, "inductor_compiled_code"),
        # torch_attn custom backend
        (torch.ops, "torch_attn._varlen_attn.default"),
    ]

    # Communication ops whose outputs should be saved to avoid re-communication.
    comm_ops = [
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
        # DeepEP (available when deepep is installed)
        (torch.ops, "deepep.dispatch.default"),
        (torch.ops, "deepep.combine.default"),
        # HybridEP (available when hybridep is installed)
        (torch.ops, "hybridep.dispatch.default"),
        (torch.ops, "hybridep.combine.default"),
    ]

    def _resolve_ops(op_specs: list) -> dict:
        ops = {}
        for spec in op_specs:
            if isinstance(spec, tuple):
                obj, path = spec
                try:
                    for part in path.split("."):
                        obj = getattr(obj, part)
                    ops[obj] = CheckpointPolicy.MUST_SAVE
                except AttributeError:
                    pass
            else:
                ops[spec] = CheckpointPolicy.MUST_SAVE
        return ops

    aten_op_types = get_default_op_list()
    # 不只依赖自己手写的列表，还合并了 PyTorch 编译器认为计算密集的算子。
    save_ops = {
        op.default  # pyrefly: ignore [missing-attribute]
        for op in aten_op_types.compute_intensive_ops
    }
    save_ops.update(_resolve_ops(compute_ops))
    save_ops.update(_resolve_ops(comm_ops))
    return save_ops


def _disable_dynamo_lru_cache() -> None:
    # Disable dynamo LRU cache to workaround an interaction between SAC, PP, and Flex:
    #
    # When forward runs with a second PP microbatch, it triggers recompilation with dynamic
    # shapes enabled. Now there are two valid compiled graphs. By default, dynamo selects
    # the latest one (the dynamic shapes version), so the runtime wrapper expects an extra
    # symint output. When SAC caches the inductor HOP output from the static graph for
    # batch_idx=0, it would miss that symint and cause an assertion failure. The workaround
    # here is to disable the LRU cache, and select graphs in insertion order instead.
    #
    # Also see: https://github.com/pytorch/pytorch/issues/166926
    # pyrefly: ignore [missing-attribute]
    # 为了规避以下组合的问题：
    # SelectiveAC + Pipeline Parallel + FlexAttention + torch.compile
    # PP 的第二个 microbatch 可能触发动态 shape 重编译，此时 Dynamo 里同时存在静态和动态图
    # 默认 LRU 策略可能选中较新的动态图，但 SelectiveAC 缓存的却是第一次静态图产生的结果。动态图可能额外需要一个 SymInt，最终造成输出结构不匹配。
    # 禁用 LRU 后，Dynamo 按插入顺序选择图，避免静态/动态图混用。
    torch._C._dynamo.eval_frame._set_lru_cache(False)


class ActivationCheckpointing(Configurable):
    """Base class for activation checkpointing policies.

    A policy is selected via the Trainer config (see ``ActivationCheckpointingConfig``)
    and applied to a model with ``policy.apply(model)``. To customize the per-op SAC
    save set, subclass ``SelectiveAC`` and override ``get_save_ops``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        preserve_rng_state: bool = True
        """
        If deterministic output compared to non-checkpointed passes is required, set
        to true. Results in stashing and restoring the RNG state during each checkpoint,
        may be slower. See https://docs.pytorch.org/docs/stable/checkpoint.html
        for details.
        """

        determinism_check: str = "default"
        """
        A string specifying the determinism function. See
        https://docs.pytorch.org/docs/stable/checkpoint.html for details.
        """

        debug: bool = False
        """
        Capture ac debug information. Will be slower. See
        https://docs.pytorch.org/docs/stable/checkpoint.html for details.
        """

    def __init__(self, config: "ActivationCheckpointing.Config", dump_folder: str = ""):
        self.config = config
        self.dump_folder = dump_folder

    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        """Wrap a single transformer block with this policy's checkpointing."""
        raise NotImplementedError

    def apply(self, model: nn.Module) -> None:
        """Apply activation checkpointing to every transformer block of the model."""
        _disable_dynamo_lru_cache()
        layers = model.get_submodule("layers")
        # 逐层把原来的 Transformer Block 替换成 checkpoint wrapper
        for layer_id, transformer_block in layers.named_children():
            transformer_block = self._wrap_block(
                transformer_block, base_fqn=f"layers.{layer_id}"
            )
            layers.register_module(layer_id, transformer_block)
        logger.info(
            f"Applied {type(self).__name__} activation checkpointing to the model"
        )


class FullAC(ActivationCheckpointing):
    """Recompute the entire transformer block during the backward pass."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
        pass

    # 大致行为：
    # 原 forward：
    #     输入 x
    #     执行完整 Block
    #     不保留 Block 内部的大量中间激活
    # backward：
    #     根据 x 重新执行完整 Block
    #     恢复内部激活
    #     计算梯度
    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        # checkpoint_wrapper 会在 forward() 中调用 torch.utils.checkpoint.checkpoint()
        # checkpoint 机制不会像普通 Autograd 那样长期保留 Block 内部所有反向所需的中间激活。
        # 【原理】通过在 autograd 保存张量的管道上挂 pack/unpack 钩子实现：前向时把中间激活"调包"成
        # 占位符省显存；反向时哪个 op 需要激活，unpack 就重算整个模块并"交货"。
        return ptd_checkpoint_wrapper(
            module,
            preserve_rng_state=self.config.preserve_rng_state,
            determinism_check=self.config.determinism_check,
            early_stop=False,
            debug=self.config.debug,
        )


# 可以把 SelectiveAC 理解成三层规则：
# 第一层：必须保存
#     Attention
#     topk
#     通信
#     CUDA → CPU
#     其他昂贵/不适合重放的操作

# 第二层：明确的特殊例外
#     moe.router.gate 对应形状的 mm
#     始终重算

# 第三层：剩余普通 mm
#     没有更精确信息
#     简单保存一个、重算一个

# 其他便宜算子
#     默认重算
class SelectiveAC(ActivationCheckpointing):
    """Per-op selective activation checkpointing.

    Saves the outputs of compute/communication ops that are expensive to
    recompute (see ``get_save_ops``) while recomputing the rest, and recomputes
    every second matmul to balance memory and compute. Override ``get_save_ops``
    in a subclass to tune which ops are saved.
    普通 mm/linear	隔一个保存、隔一个重算  --> 这好像只是一种工程上的折衷考虑？
    指定形状的 mm/linear	强制重算    --> 真实设计意图是：要重算 moe.router.gate
                          形状匹配只是因为运行时拿不到可靠模块名而采用的间接识别方式。
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
        # 为什么一定要重算gate
        # moe.router.gate 是一个已知“不值得长期保存输出”的例外。
        # Router Gate 通常是：
        # hidden_states [seq_lens, hidden_size]
        #         @
        # weight        [hidden_size, num_experts]
        #         ↓
        # router_logits [seq_lens, num_experts]
        # num_experts 通常远小于 FFN intermediate_size
        # 相对于 Attention 大投影、Expert FFN 来说，Router Gate 是一个规模较小的矩阵乘。
        # 重新计算它的绝对开销较低，没有必要仅仅因为它碰巧位于奇数位置，就把输出一直保存到 backward。
        force_recompute_mm_shapes_by_fqns: list[str] = field(
            default_factory=lambda: ["moe.router.gate"]
        )
        """
        This list of fully qualified names is used to determine which mm shapes to
        force recompute, rather than being considered by rest of the sac policy,
        e.g save every other mm. Only nn.Linear modules are supported today.

        Note: this config applies to mms not limited to those matching the specified
        fqns, e.g. if "moe.router.gate", corresponding to Linear(in, out), is specified,
        ANY mm with shape matching (*, in) x (in, out) will be force recomputed.
        """

    def get_save_ops(self) -> set:
        """Returns the set of ops whose activations should be saved. Override
        to customize the save set."""
        return _get_default_save_ops()

    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        config = cast("SelectiveAC.Config", self.config)
        save_ops = self.get_save_ops()

        # Collect weight shapes to force-recompute, stored as mm RHS shape
        # (in_f, out_f). For aten.linear we transpose args[1].shape at lookup
        # time to match, since linear's weight is (out_f, in_f).
        mm_recompute_shapes = set()
        mm_recompute_fqns = config.force_recompute_mm_shapes_by_fqns

        if mm_recompute_fqns:
            for module_fqn, submod in module.named_modules():
                fqn = f"{base_fqn}.{module_fqn}" if base_fqn else module_fqn
                if not any(f in fqn for f in mm_recompute_fqns):
                    continue
                if not isinstance(submod, nn.Linear):
                    raise ValueError(
                        "force_recompute_mm_shapes_by_fqns expected to "
                        f"match a nn.Linear, but got: {submod}"
                    )
                out_f, in_f = submod.weight.shape
                mm_recompute_shapes.add((in_f, out_f))

        # Some backends (e.g. PrivateUse1) register aten.linear as a leaf op
        # instead of decomposing it into aten.mm, so we must handle both.
        mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)

        def _get_custom_policy():
            meta = {"forward_mm_count": 0, "recompute_mm_count": 0}

            # if 是 CUDA -> CPU 拷贝:
            #     MUST_SAVE
            # if 是 mm/linear 且形状被强制重算:
            #     PREFER_RECOMPUTE
            # if func in save_ops:
            #     if 是第 2、4、6... 个 mm/linear:
            #         PREFER_RECOMPUTE
            #     else:
            #         MUST_SAVE
            # return PREFER_RECOMPUTE
            def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
                # Always save CUDA→CPU results to avoid recomputing them
                # (e.g. MoE D2H sync for all-to-all metadata).
                if (
                    func == torch.ops.aten._to_copy.default
                    and "cuda" in str(args[0].device)
                    and "device" in kwargs
                    and str(kwargs["device"]) == "cpu"
                ):
                    return CheckpointPolicy.MUST_SAVE

                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"

                if func in mm_ops:
                    weight_shape = args[1].shape
                    # linear weight is (out, in); normalize to (in, out) to match mm
                    if func == torch.ops.aten.linear.default:
                        weight_shape = torch.Size((weight_shape[1], weight_shape[0]))
                    if weight_shape in mm_recompute_shapes:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    meta[mm_count_key] += 1

                # Save all compute/comm ops, except every second mm/linear.
                if func in save_ops:
                    # [TODO] 采用计数策略应该只是因为实现简单
                    if func in mm_ops and meta[mm_count_key] % 2 == 0:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            return wrapped_policy

        return ptd_checkpoint_wrapper(
            module,
            context_fn=lambda: create_selective_checkpoint_contexts(
                _get_custom_policy()
            ),
            preserve_rng_state=config.preserve_rng_state,
            determinism_check=config.determinism_check,
            early_stop=False,
            debug=config.debug,
        )


class MemoryBudgetAC(ActivationCheckpointing):
    """Let the compiler partitioner trade compute for memory via a memory budget.

    Requires the model to be compiled (validated in ``Trainer.Config``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
        memory_budget: float = 0.5
        """
        This value determines how much partitioner in the compiler should trade off
        compute for memory. 0.0 corresponds to the activation memory from applying
        activation checkpointing to the full compiled region, and 1.0 corresponds to
        the activation memory from the default runtime-optimized strategy. Read here:
        https://pytorch.org/blog/activation-checkpointing-techniques/
        """

        visualize_memory_budget_pareto: bool = False
        """
        This dumps out a SVG visualization of the expected runtime vs. activation
        memory tradeoffs for all memory budget values from 0 to 1 in increments of
        0.05 in {--dump_folder}/memory_budget_pareto folder. See an example here:
        https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
        """

    def apply(self, model: nn.Module) -> None:
        _disable_dynamo_lru_cache()
        config = cast("MemoryBudgetAC.Config", self.config)
        if config.visualize_memory_budget_pareto:
            pareto_dir = os.path.join(self.dump_folder, "memory_budget_pareto")
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir, exist_ok=True)
            torch._functorch.config.memory_budget_pareto_dir = pareto_dir
            torch._functorch.config.visualize_memory_budget_pareto = True

        torch._functorch.config.activation_memory_budget = config.memory_budget
        logger.info(f"Selected {config.memory_budget} budget option")


# Trainer config field type: select a policy via tyro subcommand, or ``None`` to
# disable activation checkpointing. Explicit subcommand names are required because
# every nested Config class is named "Config" and would otherwise collide.
ActivationCheckpointingConfig = (
    Annotated[SelectiveAC.Config, tyro.conf.subcommand("selective")]
    | Annotated[FullAC.Config, tyro.conf.subcommand("full")]
    | Annotated[MemoryBudgetAC.Config, tyro.conf.subcommand("memory-budget")]
    | Annotated[None, tyro.conf.subcommand("none")]
)
