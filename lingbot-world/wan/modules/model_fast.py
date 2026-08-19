"""Some of the functions are borrowed from SelfForcing (https://github.com/guandeh17/Self-Forcing)."""
import math
from einops import rearrange

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    rope_params,
    sinusoidal_embedding_1d
)

from .attention import flash_attention


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def is_dynamic_kv_cache(kv_cache):
    return kv_cache is not None and kv_cache.get("mode") == "dynamic"


def _seg_tensor(segment, key, dtype=None):
    """Return a segment's K or V tensor."""
    return segment[key]


def build_dynamic_kv_tensors(
    kv_cache,
    current_key,
    current_value,
    max_attention_size,
    exclude_chunk_ids=None,
):
    """Concatenate historical segments + current key/value.

    ``exclude_chunk_ids`` (optional set/list) drops matching historical segments
    without mutating the cache — used by the attn-output-regret oracle.
    """
    segments = kv_cache.get("segments", [])
    if exclude_chunk_ids:
        exclude = {int(x) for x in exclude_chunk_ids}
        segments = [s for s in segments if int(s.get("chunk_id", -1)) not in exclude]
    k_parts = [_seg_tensor(segment, "k", current_key.dtype) for segment in segments]
    v_parts = [_seg_tensor(segment, "v", current_value.dtype) for segment in segments]
    k_parts.append(current_key)
    v_parts.append(current_value)
    if len(k_parts) == 1:
        k_cache = k_parts[0]
        v_cache = v_parts[0]
    else:
        k_cache = torch.cat(k_parts, dim=1)
        v_cache = torch.cat(v_parts, dim=1)
    if max_attention_size is not None and max_attention_size > 0 and k_cache.shape[1] > max_attention_size:
        # Sink-aware fallback. Eviction normally keeps the dynamic cache inside
        # ``attention_budget`` so this branch is a safety net; if it fires we
        # preserve the contiguous sink prefix instead of blindly tail-trimming
        # (which would silently drop motion-aware archive picks).
        sink_tokens = 0
        for seg in segments:
            if seg.get("is_sink", False):
                sink_tokens += seg["token_count"]
            else:
                break
        if sink_tokens <= 0 or sink_tokens >= max_attention_size:
            k_cache = k_cache[:, -max_attention_size:]
            v_cache = v_cache[:, -max_attention_size:]
        else:
            keep_after_sink = max_attention_size - sink_tokens
            k_cache = torch.cat(
                [k_cache[:, :sink_tokens], k_cache[:, -keep_after_sink:]], dim=1)
            v_cache = torch.cat(
                [v_cache[:, :sink_tokens], v_cache[:, -keep_after_sink:]], dim=1)
    return k_cache, v_cache


def build_dynamic_kv_segment(current_key, current_value):
    return {
        "token_count": int(current_key.shape[1]),
        "k": current_key.detach().clone(),
        "v": current_value.detach().clone(),
    }


# ── Chunk-selector oracle (Learned / World-State CR) ─────────────────────────
# During a *teacher* generation pass we record, for the current chunk's query,
# how much attention mass each historical chunk (segment) receives. This is the
# supervision target for the ChunkSelector. OFF by default; the extra softmax is
# only computed on `probe_every`-th DiT layer during the context forward
# (return_kv_segment=True). Lives at module scope so it survives FSDP.
_ORACLE_STATE = {
    "on": False,
    "probe_every": 8,
    "records": [],
    "mode": "attention_mass",
    "regret_max_cands": 8,
}


def oracle_reset():
    _ORACLE_STATE["records"] = []


def oracle_set(on, probe_every=8, mode="attention_mass", regret_max_cands=8):
    _ORACLE_STATE["on"] = bool(on)
    _ORACLE_STATE["probe_every"] = max(1, int(probe_every))
    _ORACLE_STATE["mode"] = mode or "attention_mass"
    _ORACLE_STATE["regret_max_cands"] = max(1, int(regret_max_cands))


def oracle_state():
    return _ORACLE_STATE


@torch.no_grad()
def _attention_output_from_kv(roped_query, k_cache, v_cache, head_dim):
    """Cheap mean-query attention output for regret proxy: [B, H, D]."""
    qbar = roped_query.float().mean(dim=1)  # [B, H, D]
    scale = head_dim ** 0.5
    logits = torch.einsum("bthd,bhd->bth", k_cache.float(), qbar) / scale
    probs = torch.softmax(logits, dim=1)  # [B, Tk, H]
    # probs: [B,Tk,H], v: [B,Tk,H,D] -> out [B,H,D]
    out = torch.einsum("bth,bthd->bhd", probs, v_cache.float())
    return out


@torch.no_grad()
def _record_oracle_mass(kv_cache, roped_query, k_cache, head_dim, v_cache=None):
    """Append one per-segment attention-mass (and optional regret) record.

    roped_query: [B, Tq, H, D] (this rank's local sequence shard, all heads)
    k_cache:     [B, Tk, H, D] = cat(historical segments' keys) + current key at tail
    v_cache:     optional, required for attn_output_regret mode

    Mass is computed with the mean query (cheap, O(Tk·H·D)) and averaged over heads;
    under Ulysses SP each rank sees a token slice, so we all-reduce the per-segment
    mass to a rank-consistent estimate. The current chunk's own key tail is excluded.
    """
    segments = kv_cache.get("segments", [])
    if not segments:
        return
    hist_tokens = sum(int(s["token_count"]) for s in segments)
    tq = roped_query.shape[1]
    # Only record when the cache was NOT sink-aware-truncated (collection runs at a
    # high budget so this holds); otherwise segment slices would misalign.
    if k_cache.shape[1] != hist_tokens + tq:
        return
    qbar = roped_query.float().mean(dim=1)                                  # [B,H,D]
    logits = torch.einsum("bthd,bhd->bth", k_cache.float(), qbar) / (head_dim ** 0.5)
    probs = torch.softmax(logits, dim=1)                                    # [B,Tk,H] over Tk
    tokmass = probs.mean(dim=2).mean(dim=0)                                 # [Tk] mean over heads,B
    if dist.is_initialized() and dist.get_world_size() > 1:
        tm = tokmass.to(torch.float64).contiguous()
        dist.all_reduce(tm, op=dist.ReduceOp.SUM)
        tokmass = (tm / dist.get_world_size())
    seg_mass = []
    seg_ids = []
    off = 0
    for s in segments:
        tc = int(s["token_count"])
        seg_slice = tokmass[off:off + tc]
        seg_mass.append(float(seg_slice.sum().item()))
        seg_ids.append(int(s["chunk_id"]))
        off += tc

    seg_regret = None
    mode = _ORACLE_STATE.get("mode", "attention_mass")
    if mode == "attn_output_regret" and v_cache is not None:
        # Full attention output vs drop-one-segment outputs on this probe layer.
        if v_cache.shape[1] != k_cache.shape[1]:
            seg_regret = [None] * len(seg_ids)
        else:
            full_out = _attention_output_from_kv(roped_query, k_cache, v_cache, head_dim)
            # Rank archive candidates by mass; only drop top-A non-sink segments.
            cand_order = sorted(
                range(len(seg_ids)),
                key=lambda i: seg_mass[i],
                reverse=True,
            )
            max_c = int(_ORACLE_STATE.get("regret_max_cands", 8))
            regrets = [None] * len(seg_ids)
            n_done = 0
            for idx in cand_order:
                if n_done >= max_c:
                    break
                sid = seg_ids[idx]
                # Skip sink-looking first segment if marked.
                if segments[idx].get("is_sink", False):
                    continue
                k_drop, v_drop = build_dynamic_kv_tensors(
                    kv_cache,
                    current_key=k_cache[:, -tq:],
                    current_value=v_cache[:, -tq:],
                    max_attention_size=None,
                    exclude_chunk_ids={sid},
                )
                drop_out = _attention_output_from_kv(roped_query, k_drop, v_drop, head_dim)
                # Normalized L2 over [B,H,D]
                diff = (full_out - drop_out).flatten()
                denom = full_out.flatten().norm().clamp_min(1e-8)
                regrets[idx] = float((diff.norm() / denom).item())
                n_done += 1
            if dist.is_initialized() and dist.get_world_size() > 1:
                # Average finite regrets across ranks.
                for i, r in enumerate(regrets):
                    t = torch.tensor(
                        [0.0 if r is None else float(r), 0.0 if r is None else 1.0],
                        dtype=torch.float64, device=roped_query.device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    if t[1].item() > 0:
                        regrets[i] = float(t[0].item() / t[1].item())
            seg_regret = regrets

    rec = {
        "gen_chunk_id": int(kv_cache.get("current_chunk_id", -1)),
        "layer_idx": int(kv_cache.get("layer_idx", -1)),
        "seg_ids": seg_ids,
        "seg_mass": [float(x) for x in seg_mass],
    }
    if seg_regret is not None:
        rec["seg_regret"] = seg_regret
    _ORACLE_STATE["records"].append(rec)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        return_kv_segment=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        current_end = current_start + roped_query.shape[1]
        if is_dynamic_kv_cache(kv_cache):
            k_cache, v_cache = build_dynamic_kv_tensors(
                kv_cache=kv_cache,
                current_key=roped_key,
                current_value=v,
                max_attention_size=max_attention_size,
            )
            x = attention(roped_query, k_cache, v_cache)
            ctx_tokens = int(k_cache.shape[1])
            if "global_end_index" in kv_cache:
                kv_cache["global_end_index"].fill_(current_end)
            if "local_end_index" in kv_cache:
                kv_cache["local_end_index"].fill_(ctx_tokens)
            x = x.flatten(2)
            x = self.o(x)
            if return_kv_segment:
                if _ORACLE_STATE["on"]:
                    layer_idx = int(kv_cache.get("layer_idx", 0))
                    if layer_idx % _ORACLE_STATE["probe_every"] == 0:
                        _record_oracle_mass(
                            kv_cache, roped_query, k_cache, self.head_dim, v_cache=v_cache)
                return x, build_dynamic_kv_segment(roped_key, v)
            return x
        sink_tokens = self.sink_size * frame_seqlen
        # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = roped_query.shape[1]
        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            # Calculate the number of new tokens added in this step
            # Shift existing cache content left to discard oldest tokens
            # Clone the source slice to avoid overlapping memory error
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            # Insert the new keys/values at the end
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            # Assign new keys/values directly up to current_end
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        k_cache = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
        v_cache = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]
        x = attention(roped_query, k_cache, v_cache)

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        if return_kv_segment:
            return x, None
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if crossattn_cache["is_init"].item() == 0:
                crossattn_cache["is_init"].fill_(1)
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"].copy_(k)
                crossattn_cache["v"].copy_(v)
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim=dim,
                                                num_heads=num_heads,
                                                local_attn_size=local_attn_size,
                                                sink_size=sink_size,
                                                qk_norm=qk_norm,
                                                eps=eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def _ica_cast_like(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return t.to(device=ref.device, dtype=ref.dtype)

    def _ica_cam_inject(self, x, dit_cond_dict, block_index: int = 0):
        from .cond_hoist import block_cam_mod
        return block_cam_mod(self, x, dit_cond_dict, block_index=block_index)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        return_kv_segment=False,
        block_index=0,
        ica_ctrl=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ica_ctrl (`dict`, optional): ICA Oracle per-block control. Modes:
                collect | B0 | B1 | B2 | B3. Diagnosis only when used from ICA path.
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        mode = None
        prev = {}
        if ica_ctrl is not None:
            mode = str(ica_ctrl.get("mode") or "collect")
            from .ica_oracle import ica_slot_for
            if mode == "collect":
                prev = ica_slot_for(int(ica_ctrl.get("block_index", block_index)), write=False)
            else:
                prev = ica_slot_for(int(ica_ctrl.get("block_index", block_index)), write=False)

        x_in = x
        kv_segment = None
        do_profile = False
        if mode == "collect" and ica_ctrl is not None:
            from .ica_oracle import ica_want_profile
            do_profile = bool(ica_want_profile()) and torch.cuda.is_available()

        def _cuda_ms(fn):
            if not do_profile:
                return fn(), 0.0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            return out, float(start.elapsed_time(end))

        # ── B0: whole-block residual reuse ──────────────────────────────
        if mode == "B0" and prev.get("delta") is not None:
            x = x_in + self._ica_cast_like(prev["delta"], x_in)
            if return_kv_segment:
                return x, None
            return x

        # ── Self-attention (B1: reuse raw attn_y, current gate e[2]) ─────
        self_ms = 0.0
        if mode == "B1" and prev.get("attn_y") is not None:
            y = self._ica_cast_like(prev["attn_y"], x_in)
        else:
            def _run_self():
                return self.self_attn(
                    self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
                    seq_lens, grid_sizes, freqs, kv_cache, current_start,
                    max_attention_size,
                    return_kv_segment=return_kv_segment)
            self_attn_out, self_ms = _cuda_ms(_run_self)
            if return_kv_segment:
                y, kv_segment = self_attn_out
            else:
                y = self_attn_out
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # Camera inject always recomputed elemwise; Linear×4 may be hoisted.
        def _run_cam():
            return self._ica_cam_inject(x, dit_cond_dict, block_index=block_index)
        x, cam_ms = _cuda_ms(_run_cam)

        # ── Cross-attn (B3: reuse cross_y) ───────────────────────────────
        cross_ms = 0.0
        if mode == "B3" and prev.get("cross_y") is not None:
            cross_y = self._ica_cast_like(prev["cross_y"], x)
            x = x + cross_y
        else:
            def _run_cross():
                return self.cross_attn(
                    self.norm3(x), context, context_lens,
                    crossattn_cache=crossattn_cache)
            cross_y, cross_ms = _cuda_ms(_run_cross)
            x = x + cross_y

        # ── FFN (B2: reuse raw ffn_y, current gate e[5]) ─────────────────
        x_norm = self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        ffn_ms = 0.0
        if mode == "B2" and prev.get("ffn_y") is not None:
            y_ffn = self._ica_cast_like(prev["ffn_y"], x)
        else:
            def _run_ffn():
                return self.ffn(x_norm)
            y_ffn, ffn_ms = _cuda_ms(_run_ffn)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y_ffn * e[5].squeeze(2)

        if do_profile:
            from .ica_oracle import ica_record_submodule_times
            ica_record_submodule_times(
                self_ms=self_ms, cross_ms=cross_ms, ffn_ms=ffn_ms, cam_ms=cam_ms)

        # ── Collect caches + local counterfactual block errors ──────────
        if mode == "collect" and ica_ctrl is not None:
            from .ica_oracle import ica_slot_for, ica_on_block_result
            slot = ica_slot_for(int(ica_ctrl.get("block_index", block_index)), write=True)
            slot["delta"] = (x - x_in).detach()
            slot["attn_y"] = y.detach()
            slot["cross_y"] = cross_y.detach()
            slot["ffn_y"] = y_ffn.detach()

            if ica_ctrl.get("record_local") and prev.get("delta") is not None:
                # Local e_block only — no extra self-attn (avoids KV pollution).
                # B1/B2/B3 quality is judged primarily by end-to-end e_x0 CF.
                approx = {}
                approx["B0"] = x_in + self._ica_cast_like(prev["delta"], x_in)
                # Gate-rebased single-residual proxies (cheap, same x_in):
                # B1 proxy: replace attn contribution with cached y * current gate
                #   true: x = (...attn path...); proxy swaps only the attn residual term
                if prev.get("attn_y") is not None:
                    # x_full = x_in + y*e2 + cam_delta + cross + ffn*e5
                    # approx: x_in + attn_y_prev*e2 + (x_full - x_in - y*e2)
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        attn_term_true = y * e[2].squeeze(2)
                        attn_term_hat = self._ica_cast_like(prev["attn_y"], x) * e[2].squeeze(2)
                        approx["B1"] = x + (attn_term_hat - attn_term_true)
                if prev.get("ffn_y") is not None:
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        ffn_term_true = y_ffn * e[5].squeeze(2)
                        ffn_term_hat = self._ica_cast_like(prev["ffn_y"], x) * e[5].squeeze(2)
                        approx["B2"] = x + (ffn_term_hat - ffn_term_true)
                if prev.get("cross_y") is not None:
                    approx["B3"] = x + (
                        self._ica_cast_like(prev["cross_y"], x) - cross_y)

                # Group e_group: B0 residual chain only (exact for whole-block reuse).
                from .ica_oracle import ica_state
                st = ica_state()
                n_layers = int(ica_ctrl.get("num_layers") or 0)
                n_groups = int(ica_ctrl.get("num_groups") or 0)
                if n_layers > 0:
                    from .xcache_groups import block_to_group
                    gi = block_to_group(
                        int(ica_ctrl.get("block_index", block_index)),
                        n_layers, n_groups)
                    branch = (st.get("branch") or {}).get(gi, {})
                    if "B0" in branch and branch["B0"] is not None:
                        approx["B0_branch"] = (
                            branch["B0"] + self._ica_cast_like(prev["delta"], branch["B0"]))

                ica_on_block_result(
                    ctrl=ica_ctrl, x_in=x_in, x_full=x, approx=approx)

        if return_kv_segment:
            return x, kv_segment
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModelFast(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 control_type='cam',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            control_type (`str`, *optional*, defaults to 'cam'):
               Type of conditioning control signal - 'cam' (6-dim camera Plucker
               embeddings) or 'act' (7-dim action embeddings including WASD movement)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        if control_type == 'cam':
            control_dim = 6
        elif control_type == 'act':
            control_dim = 7

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        return_kv_segments=False,
        xcache=None,
        ica=None,
        cond_hoist=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            dit_cond_dict (`dict`, *optional*, defaults to None):
                Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
            kv_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, kv_size, num_heads, head_dim]), ``global_end_index``,
                and ``local_end_index`` (scalar Tensors tracking cache position).
            crossattn_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
            current_start (`int`, *optional*, defaults to 0):
                Token offset of the current chunk in the full sequence. Used to index
                into the KV cache and compute positional embeddings correctly.
            max_attention_size (`int`, *optional*, defaults to 1_000_000):
                Maximum number of KV tokens each query can attend to. Limits the
                effective context window of self-attention to control memory usage.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert y is not None

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Patch embedding: optional exact Conv3d static/dynamic split for i2v.
        from .cond_hoist import (
            hoist_enabled, hoist_state, get_or_compute_global_cam,
            ensure_block_cam_slots, patch_embed_split, conv3d_static_contribution,
        )
        ch = cond_hoist if isinstance(cond_hoist, dict) else None
        use_conv_split = (
            hoist_enabled()
            and bool(hoist_state().get("conv_split"))
            and y is not None
        )
        if use_conv_split:
            st = hoist_state()
            if st.get("conv_static") is None:
                st["conv_static"] = conv3d_static_contribution(self, y)
            x = patch_embed_split(self, x, y, static_list=st["conv_static"])
        else:
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_lens)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_lens)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # cam — hoist global embedding (+ optional per-block scale/shift)
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            emb = get_or_compute_global_cam(self, dit_cond_dict)
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = emb
            dit_cond_dict["_cam_emb_ready"] = True
            # FSDP-safe: allocate slots; fill lazily inside each block forward.
            if hoist_enabled() and hoist_state().get("block_cam"):
                ensure_block_cam_slots(len(self.blocks))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dit_cond_dict=dit_cond_dict,
            max_attention_size=max_attention_size)

        pending_kv_segments = [] if return_kv_segments else None
        xc_on = (xcache is not None) and xcache.get("enabled", False) and (not return_kv_segments)
        xc_skip = xcache["skip_blocks"] if xc_on else frozenset()
        xc_step = xcache["step_idx"] if xc_on else -1
        if xc_on:
            from .xcache import xcache_begin_forward
            xcache_begin_forward(xcache)
        ica_on = (
            ica is not None
            and bool(ica.get("enabled", False))
            and (not return_kv_segments)
        )
        if ica_on:
            from .ica_oracle import ica_begin_forward, ica_ctrl_for_block
            ica_begin_forward(ica)
        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "block_index": block_index,
                    "ica_ctrl": (
                        ica_ctrl_for_block(ica, block_index) if ica_on else None
                    ),
                }
            )
            if return_kv_segments:
                x, kv_segment = block(x, **kwargs, return_kv_segment=True)
                pending_kv_segments.append(kv_segment)
            elif xc_on and block_index in xc_skip:
                from .xcache import xcache_apply_block
                x = xcache_apply_block(
                    block, x, kwargs, block_index, xc_step, xcache)
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        outputs = [u.float() for u in x]
        if return_kv_segments:
            return outputs, pending_kv_segments
        return outputs


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
