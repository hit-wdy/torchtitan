# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.full_dtensor import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
    validate_config,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.deepseek_v3 import DeepSeekV3Model


# DeepSeekV3 meta 模型
# │
# ├── CP 包装（如果开启）
# ├── TP：切分 attention、embedding、lm_head
# ├── EP：切分 MoE routed experts
# ├── Activation Checkpointing
# ├── torch.compile（如果配置）
# └── FSDP：进一步切分参数并安装 all-gather/reduce-scatter hook
# 主要完成的是：
# 确定参数属于哪个 mesh；
# 确定每个参数怎样切分；
# 将参数变成 DTensor/FSDP 管理的形式；
# 安装 forward 前后的通信逻辑；
# 还没有为所有参数真正分配存储。

def parallelize_deepseekv3(
    model: DeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        # 验证开启CP的情况下attn是不是FlexAttention,否则报错
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    else:
        # CP: wrap inner attention forward BEFORE parallelize() so CP logic
        # runs inside the local_map boundary on local tensors.
        if parallel_dims.cp_enabled:
            apply_cp_to_forward(
                # pyrefly: ignore [missing-attribute]
                [block.attention.inner_attention for block in model.layers.values()],
                parallel_dims.get_mesh("cp"),
            )
        if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
            model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    # 先 compile 原始 block 再 AC ，编译器可能无法正确感知：
    # - 哪部分 forward 需要保存；
    # - 哪部分反向时需要重算；
    # - AC wrapper 和 compiled graph 的边界。

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if model_compile_enabled:
        apply_compile(model, compile_config)

    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model
