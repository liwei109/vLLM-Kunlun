# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.models.vision import get_vit_attn_backend

# from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
# from vllm.v1.attention.backends.registry import AttentionBackendEnum
# from vllm.v1.attention.ops.vit_attn_wrappers import (
#     vit_flash_attn_wrapper,
#     vit_torch_sdpa_wrapper,
# )

logger = init_logger(__name__)


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = (
            3 if (device_capability.major == 9 and is_fa_version_supported(3)) else 2
        )

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major == 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def vit_flash_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
        q,
        k,
        v,
        batch_size,
        is_rocm_aiter,
        fa_version,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def vit_torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.torch_sdpa_wrapper(q, k, v, scale, cu_seqlens)


# Before codes are dependencies
################################################################################


# --8<-- [start:mm_encoder_attn]
@CustomOp.register("mm_encoder_attn")
class MMEncoderAttention(CustomOp):
    """Multi-headed attention without any cache, used for multimodal encoder."""

    # --8<-- [end:mm_encoder_attn]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MultiHeadAttention
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layer_name = prefix

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) is not "
            f"divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()

        # Get device-specific vision attention backend.
        self.attn_backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=dtype,
        )

        self.is_flash_attn_backend = self.attn_backend

        self._fa_version = (
            get_flash_attn_version() if self.is_flash_attn_backend else None
        )

        logger.info_once(f"Using {self.attn_backend} for MMEncoderAttention.")

    @classmethod
    def enabled(cls) -> bool:
        return True

    def maybe_reshape_qkv_to_4d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 4D tensors:
        (batch_size, seq_len, num_heads, head_size)
        """
        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        return query, key, value

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.maybe_reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        output = vit_torch_sdpa_wrapper(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        assert (cu_seqlens is not None and max_seqlen is not None) or (
            cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.maybe_reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        output = vit_flash_attn_wrapper(
            q=query,
            k=key,
            v=value,
            batch_size=bsz,
            is_rocm_aiter= False,
            fa_version=self._fa_version,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        if self.is_flash_attn_backend:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        else:
            return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_cpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        assert (
            self.is_flash_attn_backend
        ), "XPU only supports FLASH_ATTN for vision attention."
        return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
