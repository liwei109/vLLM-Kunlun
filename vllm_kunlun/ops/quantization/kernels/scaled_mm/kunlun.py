"""
Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
"""

from typing import Optional

import torch
import xspeedgate_ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform

from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (  # noqa: E501
    CutlassScaledMMLinearKernel,
)

from vllm.platforms import PlatformEnum
from vllm.model_executor.layers.quantization.kernels.scaled_mm import _POSSIBLE_KERNELS


class KunlunScaledMMLinearKernel(CutlassScaledMMLinearKernel):

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, Optional[str]]:

        if not current_platform.is_kunlun():
            return False, "KunlunScaledMM requires running on XPU."

        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None

        x_q = torch.empty_like(x, dtype=torch.int8, device=x.device)
        if i_s is not None:
            # static-per-tensor quantization
            torch.ops._C.static_scaled_int8_quant(x_q, x.contiguous(), i_s, i_zp)
            x_s = i_s
            x_zp = i_zp
        else:
            # dynamic-per-token quantization
            x_s = torch.empty(
                (x.numel() // x.shape[-1], 1), device=x.device, dtype=torch.float32
            )
            x_zp = None if symmetric else torch.empty_like(x_s, dtype=torch.int32)
            torch.ops._C.dynamic_scaled_int8_quant(x_q, x.contiguous(), x_s, x_zp)

        out = torch.empty(
            (x_q.shape[0], w_q.shape[1]), dtype=x.dtype, device=x_q.device
        )
        if x_zp is not None:
            # asymmetric
            # Currently, static is always per-tensor and dynamic is per-token

            # FIXME: azp_adj speedup function is working in progress
            # static = i_zp is not None
            # azp = None if static else x_zp

            azp = x_zp
            torch.ops._C.cutlass_scaled_mm_azp(
                out,
                x_q.contiguous(),
                w_q.contiguous(),
                x_s,
                w_s,
                azp_adj,
                azp,
                None if bias is None else bias.contiguous(),
            )
            return out
        else:
            # symmetric
            w_s = w_s.transpose(0, 1)  # NOTE: scale dim is [n, 1]
            torch.ops._C.cutlass_scaled_mm(
                out,
                x_q.contiguous(),
                w_q.contiguous(),
                x_s,
                w_s,
                None if bias is None else bias.contiguous(),
            )
            return out


_POSSIBLE_KERNELS[PlatformEnum.CUDA] = [KunlunScaledMMLinearKernel]


print(
    f"[vllm_kunlun] ScaledMM kernels: {[k.__name__ for k in _POSSIBLE_KERNELS[PlatformEnum.CUDA]]}"
)
