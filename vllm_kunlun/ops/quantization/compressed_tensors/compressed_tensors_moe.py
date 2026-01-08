#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Li Wei, Tang Shiwen
# Email: liwei157@baidu.com, tangshiwen@baidu.com
# This file is a part of the vllm-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsW8A8Int8MoEMethod,
)


def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    # NOTE: xtorch_ops use max as scale
    with torch.no_grad():
        layer.w13_weight_scale.mul_(127.0)
        layer.w2_weight_scale.mul_(127.0)


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
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
    linear_weights: Optional[torch.Tensor] = None,  # 添加这个参数
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

    output = torch.empty_like(x)
    torch.ops._C.moe_ffn_per_token_block(
        x=x,
        inter_weight=layer.w13_weight,
        inter_scale=layer.w13_weight_scale,
        outer_weight=layer.w2_weight,
        outer_scale=layer.w2_weight_scale,
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


CompressedTensorsW8A8Int8MoEMethod.process_weights_after_loading = (
    process_weights_after_loading
)
CompressedTensorsW8A8Int8MoEMethod.apply = apply
