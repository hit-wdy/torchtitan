# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from collections.abc import Callable

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger


# TODO: Remove this monkeypatch once FakeTensorMode.__init__ is decorated with
# @torch.compiler.disable(recursive=True) upstream.
# See https://github.com/pytorch/pytorch/issues/178887
FakeTensorMode.__init__ = torch.compiler.disable(  # type: ignore[method-assign]
    FakeTensorMode.__init__, recursive=True
)


# Toggled on by ``_maybe_regional_inductor_backend`` when the model is compiled
# with a non-inductor backend that needs inductor-only regions (e.g.
# FlexAttention) scooped into an inductor sub-compile. Read by
# ``maybe_regional_inductor`` at trace time; left False on the default inductor
# / eager paths so no annotation metadata is emitted.
_regional_inductor_enabled: bool = False


def apply_compile(model: nn.Module, compile_config: CompileConfig) -> None:
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # Needed for torch.compile to handle data-dependent dynamic shapes in
    # token-choice MoE dispatch. Harmless for dense models.
    # 开启该选项可以让 Dynamo 尽量把标量输出保留在编译图中，而不是立刻 graph break。
    torch._dynamo.config.capture_scalar_outputs = True
    # Skip replaying forward side effects (e.g. RoPE cache updates) during
    # the AC recompute in backward. Eager AC replays the forward python
    # side-effects in backward, but torch.compile has no easy way to reapply
    # python mutations in the backward. Setting this flag accepts this eager
    # and compile divergence by skipping reapplication of side effects.
    # 用于协调：
    # - torch.compile；
    # - Activation Checkpointing；
    # - RoPE cache 等 forward 副作用。
    # AC 反向时会重算 forward，但编译模式不方便重新执行 Python 层副作用，所以这里要求跳过这些副作用的重复应用。
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
        True  # pyrefly: ignore [bad-assignment]
    )

    backend = _maybe_regional_inductor_backend(model, compile_config.backend)

    # 为什么按 TransformerBlock 编译
    # TransformerBlock 是自然的重复计算单元。
    # 相比整个模型一起编译：
    # - 图的规模更可控；
    # - 编译时间更合理；
    # - 不容易被跨层 Python 控制流打断；
    # - 各层结构类似，更容易复用缓存；
    # - 更容易和 Activation Checkpointing、FSDP 组合。
    # 
    # 并且fullgraph=True 要求每个 TransformerBlock 尽量形成一个完整 FX graph。
    # 如果中间遇到无法捕获的 Python 操作，不希望悄悄 graph break 成多个小图，而是尽早暴露问题。

    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block.compile(backend=backend, fullgraph=True)

    logger.info("Compiling each TransformerBlock with torch.compile")


def _maybe_regional_inductor_backend(model: nn.Module, backend: str) -> str | Callable:
    """Wrap the ``aot_eager`` backend so inductor-only flex regions are scooped out.

    ``regional_inductor`` lowers just the regions annotated with ``compile_with_inductor`` (see
    ``FlexAttention.forward``) to inductor while the rest stays in aot_eager.

    Only applied for ``aot_eager`` on models that actually use FlexAttention, so
    dense/non-flex aot_eager paths are left untouched. Other non-inductor backends
    can't be scooped here and raise rather than silently degrading.
    """
    from torchtitan.models.common.attention import FlexAttention

    uses_flex = any(isinstance(m, FlexAttention) for m in model.modules())
    # Non-flex models never need the scoop; the default inductor backend already
    # lowers the flex region directly. Both are left on the unmodified backend.
    if not uses_flex or backend == "inductor":
        return backend

    # FlexAttention only has an inductor lowering. Under a non-inductor backend
    # other than aot_eager it would decompose to eager aten ops (no Triton
    # kernel), which we can't transparently scoop here -- fail loudly.
    if backend != "aot_eager":
        raise ValueError(
            f"Model uses FlexAttention but compile backend {backend!r} is neither "
            f"'inductor' nor 'aot_eager'; the flex region would decompose to eager "
            f"aten ops (no Triton kernel). Use 'inductor' or 'aot_eager'."
        )

    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.passes.regional_inductor import regional_inductor

    global _regional_inductor_enabled
    _regional_inductor_enabled = True

    logger.info("regional_inductor is enabled")
    return aot_autograd(fw_compiler=regional_inductor, bw_compiler=regional_inductor)


def maybe_regional_inductor(
    inductor_configs: dict,
) -> contextlib.AbstractContextManager:
    """Context manager that marks the wrapped region for ``regional_inductor``.

    Returns a null context unless regional inductor is enabled (see
    ``_maybe_regional_inductor_backend``). When enabled, the region is tagged
    with ``compile_with_inductor`` so a non-inductor outer compile lowers just
    this region to inductor with ``inductor_configs``.
    """
    if not _regional_inductor_enabled:
        return contextlib.nullcontext()
    return fx_traceback.annotate(
        {"compile_with_inductor": {"inductor_configs": inductor_configs}}
    )
