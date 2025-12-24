"""
Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
"""

from typing import Optional, Callable

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsW8A8Int8MoEMethod,
)


def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,  # 添加这个参数
    expert_load_view: Optional[torch.Tensor] = None,  # 添加这个参数
    logical_to_physical_map: Optional[torch.Tensor] = None,  # 添加这个参数
    logical_replica_count: Optional[torch.Tensor] = None,  # 添加这个参数
    linear_weights: Optional[torch.Tensor] = None,  # 添加这个参数
) -> torch.Tensor:

    output = torch.empty_like(x)
    torch.ops._C.moe_ffn_per_token_block(
        x=x,
        inter_weight=layer.w13_weight,
        inter_scale=layer.w13_weight_scale * 127.0,  # NOTE: xtorch_ops use max as scale
        outer_weight=layer.w2_weight,
        outer_scale=layer.w2_weight_scale * 127.0,
        top_k=top_k,
        global_num_experts=global_num_experts,
        linear_weights=linear_weights,
        expert_map=expert_map,
        activation=activation,
        output=output,
        use_expert_parallel=expert_map is not None,
        ep_size=expert_map.size(0) if expert_map is not None else 1,
        ep_rank=0,
    )
    return output


CompressedTensorsW8A8Int8MoEMethod.apply = apply
