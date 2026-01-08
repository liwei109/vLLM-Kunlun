"""layer.py"""

from contextlib import nullcontext
from typing import Callable, Optional, Union, get_args

import torch
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.distributed import get_ep_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.platforms import current_platform


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
) -> torch.Tensor:
    """apply"""
    if enable_eplb:
        raise NotImplementedError(
            "EPLB not supported for `UnquantizedFusedMoEMethod` yet."
        )

    """forward_kunlun"""
    from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

    if self.moe.use_ep:
        return ops.fused_moe_ep(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            self.moe.ep_rank,
            top_k,
            renormalize=renormalize,
            inplace=True,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )
    else:
        return ops.fused_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            self.moe.ep_rank,
            top_k,
            renormalize=renormalize,
            inplace=True,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )


UnquantizedFusedMoEMethod.apply = apply


class VllmFusedMoE(FusedMoE):
    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = 0,
        topk_group: Optional[int] = 0,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: Optional[int] = 0,
        zero_expert_type: Optional[str] = None,
    ):
        super().__init__(
            num_experts=num_experts,  # Global number of experts
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            prefix=prefix,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=activation,
            enable_eplb=enable_eplb,
            num_redundant_experts=num_redundant_experts,
            has_bias=has_bias,
            is_sequence_parallel=is_sequence_parallel,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
        )
        self.has_bias = has_bias
        self.register_parameter("w13_bias", None)
        self.register_parameter("w2_bias", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor = None,
        linear_weights: torch.Tensor = None,
    ):
        """forward"""
        # TODO: Once the OOM issue for the TPU backend is resolved, we will
        # switch to using the moe_forward custom op.
        if current_platform.is_tpu():
            return self.forward_impl(hidden_states, router_logits)
        else:
            forward_context: ForwardContext = get_forward_context()
            self = forward_context.no_compile_layers[self.layer_name]
            assert self.quant_method is not None
            return self.forward_impl(hidden_states, router_logits, linear_weights)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        linear_weights: torch.Tensor = None,
    ):
        """forward_impl"""
        assert self.quant_method is not None
        if (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
        ):
            return self.forward_impl_chunked(hidden_states, router_logits)

        do_naive_dispatch_combine: bool = (
            self.dp_size > 1 and not self.moe_parallel_config.use_deepep_ht_kernels
        )
        if do_naive_dispatch_combine:
            hidden_states, router_logits = get_ep_group().dispatch(
                hidden_states, router_logits
            )

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_eplb=self.enable_eplb,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
            linear_weights=linear_weights,
        )

        if do_naive_dispatch_combine:
            final_hidden_states = get_ep_group().combine(final_hidden_states)

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.
            final_hidden_states = self.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states


FusedMoE = VllmFusedMoE
