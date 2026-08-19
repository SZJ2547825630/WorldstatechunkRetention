import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import os
import warnings

from torch.nn.attention import SDPBackend, sdpa_kernel

# Default: cuDNN fused Hopper SDPA on unmasked self-attention (~1.15x e2e on
# H20 / WS-CR v2 vs FA2; PSNR still above window on default_loop). Set
# WAN_ATTN_BACKEND=flash to restore FA2 for paper / ablation tables.
# See docs/experiments/ATTENTION_BACKEND.md.
_SDPA_BACKEND = os.getenv('WAN_ATTN_BACKEND', 'auto').lower()
# Boxed so the first failure disables the path for the rest of the process.
_CUDNN_SDPA_USABLE = [True]
# Which path each call took, so benchmarks can confirm the backend actually
# switched instead of silently falling back.
ATTN_PATH_COUNTS = {'cudnn': 0, 'flash': 0}

__all__ = [
    'flash_attention',
    'attention',
]


def _flash_attention_impl(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    return x.type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.

    Dispatches to whichever exact-attention backend is fastest for the given
    arguments; `version` forces the FlashAttention path.
    """
    if version is None and _sdpa_eligible(
            q, k, q_lens, k_lens, dropout_p, window_size, deterministic, causal):
        try:
            out = _sdpa_attention(q, k, v, dropout_p, softmax_scale, q_scale,
                                  causal, dtype)
            ATTN_PATH_COUNTS['cudnn'] += 1
            return out
        except Exception as e:
            _CUDNN_SDPA_USABLE[0] = False
            warnings.warn(f'cuDNN attention unavailable ({e}); using flash attention.')

    ATTN_PATH_COUNTS['flash'] += 1
    return _flash_attention_impl(
        q=q, k=k, v=v,
        q_lens=q_lens, k_lens=k_lens,
        dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale,
        causal=causal, window_size=window_size, deterministic=deterministic,
        dtype=dtype, version=version,
    )


def _sdpa_attention(q, k, v, dropout_p, softmax_scale, q_scale, causal, dtype):
    """Unpadded attention via cuDNN's fused Hopper kernel.

    Self-attention in the DiT passes no padding mask, so the varlen machinery in
    `_flash_attention_impl` degenerates to a single full-length sequence. At the
    shape this model runs (q=4680, kv=23400, 20 heads, d=128, bf16) cuDNN reaches
    ~133 TFLOPS on H20 against FlashAttention 2's ~85, i.e. 1.6x, and agrees with
    it to one bf16 ULP. Attention is ~40% of a forward, so this is ~15% overall.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype
    if q.dtype not in half_dtypes:
        q = q.to(dtype)
    k = k.to(q.dtype)
    v = v.to(q.dtype)
    if q_scale is not None:
        q = q * q_scale
    # [B, L, N, C] -> [B, N, L, C]. Left non-contiguous on purpose: cuDNN handles
    # the strided layout and the copy would cost more than it saves.
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous().type(out_dtype)


def _sdpa_eligible(q, k, q_lens, k_lens, dropout_p, window_size, deterministic,
                   causal=False):
    return (_SDPA_BACKEND != "flash"
            and _CUDNN_SDPA_USABLE[0]
            and q_lens is None and k_lens is None
            and dropout_p == 0.
            and tuple(window_size) == (-1, -1)
            and not deterministic
            and q.size(2) == k.size(2)  # no GQA
            and q.size(-1) <= 128       # cuDNN fused kernel head-dim limit
            # FlashAttention anchors a causal mask to the bottom-right of the
            # score matrix, `is_causal` to the top-left. They only agree when the
            # matrix is square.
            and (not causal or q.size(1) == k.size(1)))


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
            or _sdpa_eligible(q, k, q_lens, k_lens, dropout_p, window_size,
                              deterministic, causal)):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
