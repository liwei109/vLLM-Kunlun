"""
Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
"""

from typing import Optional

import torch
from vllm_kunlun.ops.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
    CompressedTensorsKVCacheMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm_kunlun.ops.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    get_linear_transform_schemes,
)


def get_quant_method(
    self,
    layer: torch.nn.Module,
    prefix: str,
) -> Optional["QuantizeMethodBase"]:
    from vllm_kunlun.ops.attention.layer import Attention  # Avoid circular import

    if isinstance(layer, LinearBase):
        # collect schemes
        quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
        input_tfms, output_tfms = get_linear_transform_schemes(
            layer, prefix, self.transform_config, self.packed_modules_mapping
        )

        # choose quantization method
        quant_method: LinearMethodBase = UnquantizedLinearMethod()
        if quant_scheme is not None:
            layer.scheme = quant_scheme
            quant_method = CompressedTensorsLinearMethod(self)

        # choose transform method
        if any((input_tfms, output_tfms)):
            return CompressedTensorsLinearTransformMethod.from_schemes(
                quant_method, quant_scheme, input_tfms, output_tfms
            )

        else:
            return quant_method

    if isinstance(layer, Attention):
        return CompressedTensorsKVCacheMethod(self)
    if isinstance(layer, FusedMoE):
        return CompressedTensorsMoEMethod.get_moe_method(self, layer)
    return None


CompressedTensorsConfig.get_quant_method = get_quant_method
