"""
epr_paged_attention.py — Industrial PagedAttention Engine & Triton Kernel for EPR Attention (v3.0)

Implements a vLLM-style Block-Space Physical Memory Allocator + Custom Fused `@triton.jit`
PagedAttention Decoding Kernel tailored to EPR's asymmetric Dual-Regime topology:

  1. Asymmetric Physical Page Pools (`page_size = 16` by default):
     - Damped Pool: Stores `(K_real, V)` in `bf16/fp16` and `tau_k` in `fp32` for damped KV heads.
       Maintains at most `ceil(window_size / page_size) + 1` physical pages per sequence, automatically
       recycling evicted pages back to the free-block stack in O(1) as decoding crosses page boundaries!
     - Wormhole Pool: Allocates full-context physical pages on demand strictly for the `n_wh_kv`
       zero-decay NoPE Wormhole KV head(s), eliminating memory fragmentation across variable-length
       concurrent requests.

  2. Zero-Read Online-Softmax Vacuum Folding in Triton SRAM:
     Because `V_vac = 0_D`, initializing each threadblock's online Softmax state at
     `(m_i = v0[kv_h], l_i = 1.0, acc = 0_D)` incorporates the exact learnable Vacuum Floor `e^{v_0}`
     in zero memory reads and zero extra loop iterations.

  3. Native GQA + Variable-Length Continuous Batching + Exact PyTorch Fallback:
     Works identically on CUDA (fused Triton kernel) and CPU/MPS/`fp64` (exact PyTorch block-table gather)
     for bit-level mathematical auditing.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# =============================================================================
# 1. Custom Fused Triton PagedAttention Kernel (Damped + Wormhole + Vacuum + GQA)
# =============================================================================

if _HAS_TRITON:
    @triton.jit
    def _epr_paged_decode_kernel(
        Q_ptr,                  # [B, H_q, D]
        Out_ptr,                # [B, H_q, D]
        V0_ptr,                 # [H_kv] learnable vacuum logit per KV head
        TauQ_ptr,               # [B, H_q] current query cumulative proper time
        SeqLens_ptr,            # [B] int32 current total sequence length (including current step)
        # --- Damped Pool & Block Tables ---
        KD_ptr,                 # [P_d, H_d_kv, PAGE_SIZE, D]
        VD_ptr,                 # [P_d, H_d_kv, PAGE_SIZE, D]
        TauD_ptr,               # [P_d, H_d_q, PAGE_SIZE] float32
        DBlockTable_ptr,        # [B, MAX_D_BLKS] int32 physical page IDs
        DBlockStart_ptr,        # [B, MAX_D_BLKS] int32 global start token index of each mapped page
        DNumBlks_ptr,           # [B] int32 number of active damped pages for seq b
        # --- Wormhole Pool & Block Tables ---
        KW_ptr,                 # [P_w, H_w_kv, PAGE_SIZE, D]
        VW_ptr,                 # [P_w, H_w_kv, PAGE_SIZE, D]
        WBlockTable_ptr,        # [B, MAX_W_BLKS] int32 physical page IDs
        WNumBlks_ptr,           # [B] int32 number of active wormhole pages for seq b
        # --- Strides & Constants ---
        stride_qb, stride_qh,
        stride_ob, stride_oh,
        stride_tqb,
        stride_kdp, stride_kdh, stride_kdt,
        stride_tdp, stride_tdh,
        stride_dbt_b,
        stride_kwp, stride_kwh, stride_kwt,
        stride_wbt_b,
        window_size: tl.constexpr,
        n_rep: tl.constexpr,
        n_damped_q: tl.constexpr,
        n_damped_kv: tl.constexpr,
        USE_VACUUM: tl.constexpr,
        USE_TIME_DECAY: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)
        kv_idx = h_idx // n_rep

        offs_d = tl.arange(0, HEAD_DIM)
        offs_t = tl.arange(0, PAGE_SIZE)

        seq_len = tl.load(SeqLens_ptr + b_idx)
        q_ptrs = Q_ptr + b_idx * stride_qb + h_idx * stride_qh + offs_d
        q_vec = tl.load(q_ptrs).to(tl.float32)

        # Online Softmax state: initialize directly at Vacuum Floor (m = v0, l = 1.0, acc = 0)!
        if USE_VACUUM:
            m_i = tl.load(V0_ptr + kv_idx).to(tl.float32)
            l_i = 1.0
        else:
            m_i = -float("inf")
            l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        if h_idx < n_damped_q:
            # --- Branch A: Damped Head (O(W) Ring-Evicted Page Table + Proper-Time Decay) ---
            n_blks = tl.load(DNumBlks_ptr + b_idx)
            tau_q = tl.load(TauQ_ptr + b_idx * stride_tqb + h_idx).to(tl.float32) if USE_TIME_DECAY else 0.0
            win_start = seq_len - window_size

            for blk_i in range(0, n_blks):
                phys_page = tl.load(DBlockTable_ptr + b_idx * stride_dbt_b + blk_i)
                page_start = tl.load(DBlockStart_ptr + b_idx * stride_dbt_b + blk_i)
                tok_pos = page_start + offs_t
                valid = (tok_pos < seq_len) & (tok_pos >= win_start)

                base_kv = phys_page * stride_kdp + kv_idx * stride_kdh + offs_t[:, None] * stride_kdt + offs_d[None, :]
                k_tile = tl.load(KD_ptr + base_kv, mask=valid[:, None], other=0.0).to(tl.float32)
                scores = tl.sum(k_tile * q_vec[None, :], axis=1)

                if USE_TIME_DECAY:
                    tau_k = tl.load(
                        TauD_ptr + phys_page * stride_tdp + h_idx * stride_tdh + offs_t,
                        mask=valid,
                        other=0.0,
                    ).to(tl.float32)
                    dt = tl.maximum(tau_q - tau_k, 0.0)
                    scores = scores - dt

                scores = tl.where(valid, scores, -float("inf"))
                blk_max = tl.max(scores, axis=0)
                m_new = tl.maximum(m_i, blk_max)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new)

                v_tile = tl.load(VD_ptr + base_kv, mask=valid[:, None], other=0.0).to(tl.float32)
                acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                m_i = m_new
        else:
            # --- Branch B: Wormhole Head (Full-Context Page Table, Zero Time Decay) ---
            wh_kv_idx = kv_idx - n_damped_kv
            n_blks = tl.load(WNumBlks_ptr + b_idx)

            for blk_i in range(0, n_blks):
                phys_page = tl.load(WBlockTable_ptr + b_idx * stride_wbt_b + blk_i)
                tok_pos = blk_i * PAGE_SIZE + offs_t
                valid = tok_pos < seq_len

                base_kv = phys_page * stride_kwp + wh_kv_idx * stride_kwh + offs_t[:, None] * stride_kwt + offs_d[None, :]
                k_tile = tl.load(KW_ptr + base_kv, mask=valid[:, None], other=0.0).to(tl.float32)
                scores = tl.sum(k_tile * q_vec[None, :], axis=1)
                scores = tl.where(valid, scores, -float("inf"))

                blk_max = tl.max(scores, axis=0)
                m_new = tl.maximum(m_i, blk_max)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new)

                v_tile = tl.load(VW_ptr + base_kv, mask=valid[:, None], other=0.0).to(tl.float32)
                acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                m_i = m_new

        out_vec = acc / tl.maximum(l_i, 1e-20)
        out_ptrs = Out_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_d
        tl.store(out_ptrs, out_vec.to(Out_ptr.dtype.element_ty))


# =============================================================================
# 2. Asymmetric Block-Space Manager & Paged KV-Cache (`EPRPagedKVCache`)
# =============================================================================

class EPRPagedKVCache:
    """
    Production-Style Block-Table Paged KV-Cache for EPR Attention:
      - Pre-allocates fixed-size physical pages (`page_size = 16`) in separate Damped and Wormhole
        page pools so damped pages can be O(1) recycled every `page_size` steps once sequence length
        exceeds `window_size`, with zero fragmentation across variable-length batched requests!
      - Supports both uniform batch decoding (`Transformer.generate`) and variable-length continuous
        batching (`append_slot` / `free_sequence`).
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        n_wormhole_kv_heads: int = 1,
        window_size: Optional[int] = 128,
        max_seq_len: int = 8192,
        max_batch_size: int = 16,
        page_size: int = 16,
        num_damped_pages: Optional[int] = None,
        num_wh_pages: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cpu",
        use_triton: bool = True,
    ):
        assert head_dim in (16, 32, 64, 128), f"Triton PagedAttention requires power-of-2 head_dim, got {head_dim}"
        assert page_size in (8, 16, 32, 64), f"page_size must be power-of-2 in {{8,16,32,64}}, got {page_size}"

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.n_wh_kv = n_wormhole_kv_heads
        self.n_damped_kv = n_kv_heads - n_wormhole_kv_heads
        self.n_wh_q = n_wormhole_kv_heads * self.n_rep
        self.n_damped_q = n_heads - self.n_wh_q
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.window_size = max_seq_len if (window_size is None or window_size >= max_seq_len) else window_size
        self.max_batch_size = max_batch_size
        self.page_size = page_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_triton = use_triton and _HAS_TRITON and self.device.type == "cuda"

        # Maximum pages a single sequence ever needs simultaneously:
        # Damped head needs at most ceil((window_size - 1) / page_size) + 1 pages!
        self.max_d_blks_per_seq = (
            math.ceil(max_seq_len / page_size)
            if self.window_size >= max_seq_len
            else (math.ceil(self.window_size / page_size) + 1)
        )
        self.max_w_blks_per_seq = math.ceil(max_seq_len / page_size)

        if num_damped_pages is None:
            num_damped_pages = max(1, max_batch_size * self.max_d_blks_per_seq)
        if num_wh_pages is None:
            num_wh_pages = max(1, max_batch_size * self.max_w_blks_per_seq)
        self.num_damped_pages = num_damped_pages
        self.num_wh_pages = num_wh_pages

        tau_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        self.tau_dtype = tau_dtype

        # Per-layer running proper-time state [B, H, 1] (compatible with EPRKVCache API!)
        self.tau_state = [
            torch.zeros(max_batch_size, n_heads, 1, dtype=tau_dtype, device=self.device)
            for _ in range(n_layers)
        ]

        # Physical Page Pools per layer
        d_kv_dim = max(1, self.n_damped_kv)
        d_q_dim = max(1, self.n_damped_q)
        w_kv_dim = max(1, self.n_wh_kv)

        self.k_damped_pool = [
            torch.zeros(num_damped_pages, d_kv_dim, page_size, head_dim, dtype=dtype, device=self.device)
            for _ in range(n_layers)
        ]
        self.v_damped_pool = [
            torch.zeros(num_damped_pages, d_kv_dim, page_size, head_dim, dtype=dtype, device=self.device)
            for _ in range(n_layers)
        ]
        self.tau_damped_pool = [
            torch.zeros(num_damped_pages, d_q_dim, page_size, dtype=tau_dtype, device=self.device)
            for _ in range(n_layers)
        ]

        self.k_wh_pool = [
            torch.zeros(num_wh_pages, w_kv_dim, page_size, head_dim, dtype=dtype, device=self.device)
            for _ in range(n_layers)
        ]
        self.v_wh_pool = [
            torch.zeros(num_wh_pages, w_kv_dim, page_size, head_dim, dtype=dtype, device=self.device)
            for _ in range(n_layers)
        ]

        # Shared Block Tables across layers (all layers advance token positions identically!)
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self._cpu_seq_lens = [0] * max_batch_size

        self.d_block_table = torch.full((max_batch_size, self.max_d_blks_per_seq), -1, dtype=torch.int32, device=self.device)
        self.d_block_start = torch.zeros((max_batch_size, self.max_d_blks_per_seq), dtype=torch.int32, device=self.device)
        self.d_num_blks = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

        self.w_block_table = torch.full((max_batch_size, self.max_w_blks_per_seq), -1, dtype=torch.int32, device=self.device)
        self.w_num_blks = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

        # CPU mirror of page tables + O(1) free-page stacks for zero-sync page management
        self.free_d_pages: List[int] = list(range(num_damped_pages - 1, -1, -1))
        self.free_w_pages: List[int] = list(range(num_wh_pages - 1, -1, -1))
        self._d_pages_seq: List[List[Tuple[int, int]]] = [[] for _ in range(max_batch_size)]  # [(phys_id, start_pos)]
        self._w_pages_seq: List[List[int]] = [[] for _ in range(max_batch_size)]

    @property
    def seq_pos(self) -> int:
        """Current uniform sequence position (compatible with `Transformer.forward` RoPE indexing)."""
        return self._cpu_seq_lens[0]

    def reset_sequence(self, b: int):
        """Free all physical pages held by sequence slot `b` back to the global free pool in O(1)."""
        for phys_id, _ in self._d_pages_seq[b]:
            self.free_d_pages.append(phys_id)
        for phys_id in self._w_pages_seq[b]:
            self.free_w_pages.append(phys_id)
        self._d_pages_seq[b].clear()
        self._w_pages_seq[b].clear()
        self._cpu_seq_lens[b] = 0
        self.seq_lens[b] = 0
        self.d_num_blks[b] = 0
        self.w_num_blks[b] = 0
        for l in range(self.n_layers):
            self.tau_state[l][b].zero_()

    def _ensure_token_position_allocated(self, b: int, pos: int) -> Tuple[int, int, int]:
        """
        Ensure physical damped & wormhole pages exist for sequence `b` at global token index `pos`,
        recycling any expired damped page strictly outside `[pos + 1 - window_size, pos]`.
        Returns `(damped_phys_page, wh_phys_page, page_offset)`.
        """
        offset = pos % self.page_size
        page_start = pos - offset

        # 1. Damped Pool allocation + O(1) ring eviction of expired pages
        if self.n_damped_kv > 0:
            win_start = pos + 1 - self.window_size
            # Recycle oldest page(s) whose ENTIRE span `[p_start, p_start + page_size - 1]` is `< win_start`
            while self._d_pages_seq[b] and (self._d_pages_seq[b][0][1] + self.page_size <= win_start):
                old_phys, _ = self._d_pages_seq[b].pop(0)
                self.free_d_pages.append(old_phys)

            if not self._d_pages_seq[b] or self._d_pages_seq[b][-1][1] != page_start:
                if not self.free_d_pages:
                    raise RuntimeError("Out of physical Damped KV pages in EPRPagedKVCache!")
                new_d = self.free_d_pages.pop()
                self._d_pages_seq[b].append((new_d, page_start))
            d_phys = self._d_pages_seq[b][-1][0]
        else:
            d_phys = 0

        # 2. Wormhole Pool allocation
        if self.n_wh_kv > 0:
            blk_idx = pos // self.page_size
            if blk_idx >= len(self._w_pages_seq[b]):
                if not self.free_w_pages:
                    raise RuntimeError("Out of physical Wormhole KV pages in EPRPagedKVCache!")
                new_w = self.free_w_pages.pop()
                self._w_pages_seq[b].append(new_w)
            w_phys = self._w_pages_seq[b][blk_idx]
        else:
            w_phys = 0

        self._cpu_seq_lens[b] = pos + 1
        return d_phys, w_phys, offset

    def _sync_tables_to_device(self, active_B: int):
        """Sync compact integer page tables `[active_B, ...]` to GPU once per token step (layer 0)."""
        seq_t = torch.tensor(self._cpu_seq_lens[:active_B], dtype=torch.int32, device=self.device)
        self.seq_lens[:active_B] = seq_t

        if self.n_damped_kv > 0:
            d_cnt = [len(self._d_pages_seq[b]) for b in range(active_B)]
            self.d_num_blks[:active_B] = torch.tensor(d_cnt, dtype=torch.int32, device=self.device)
            max_d = max(d_cnt) if d_cnt else 0
            if max_d > 0:
                bt = [[-1] * max_d for _ in range(active_B)]
                bs = [[0] * max_d for _ in range(active_B)]
                for b in range(active_B):
                    for j, (pid, pst) in enumerate(self._d_pages_seq[b]):
                        bt[b][j] = pid
                        bs[b][j] = pst
                self.d_block_table[:active_B, :max_d] = torch.tensor(bt, dtype=torch.int32, device=self.device)
                self.d_block_start[:active_B, :max_d] = torch.tensor(bs, dtype=torch.int32, device=self.device)

        if self.n_wh_kv > 0:
            w_cnt = [len(self._w_pages_seq[b]) for b in range(active_B)]
            self.w_num_blks[:active_B] = torch.tensor(w_cnt, dtype=torch.int32, device=self.device)
            max_w = max(w_cnt) if w_cnt else 0
            if max_w > 0:
                wt = [[-1] * max_w for _ in range(active_B)]
                for b in range(active_B):
                    for j, pid in enumerate(self._w_pages_seq[b]):
                        wt[b][j] = pid
                self.w_block_table[:active_B, :max_w] = torch.tensor(wt, dtype=torch.int32, device=self.device)

    def write_and_attend(
        self,
        layer_idx: int,
        q_aug: torch.Tensor,            # [B, H_q, T, D]
        k_real: torch.Tensor,           # [B, H_kv, T, D]
        v: torch.Tensor,                # [B, H_kv, T, D]
        tau: torch.Tensor,              # [B, H_q, T]
        v0: torch.Tensor,               # [H_kv]
        use_vacuum: bool = True,
        use_time_decay: bool = True,
    ) -> torch.Tensor:
        """
        Write `T` new tokens (`T >= 1` prefill or `T == 1` decode) into physical pages and compute
        attention output `[B, H_q, T, D]`.
        - For `T == 1` decoding on CUDA: launches the fused `@triton.jit` `_epr_paged_decode_kernel`!
        - For `T > 1` prefill or CPU/fp64: writes pages and runs exact block-table gather / prefill!
        """
        B, H_q, T, D = q_aug.shape
        self.tau_state[layer_idx][:B] = tau[:, :, -1:].to(self.tau_dtype)

        # Fast vectorized page scatter for all (b, t)
        d_pages, w_pages, offsets, b_indices, t_indices = [], [], [], [], []
        for t in range(T):
            for b in range(B):
                if layer_idx == 0:
                    pos = self._cpu_seq_lens[b]
                    dp, wp, off = self._ensure_token_position_allocated(b, pos)
                else:
                    pos = self._cpu_seq_lens[b] - T + t
                    off = pos % self.page_size
                    p_start = pos - off
                    dp = 0
                    if self.n_damped_kv > 0:
                        # Page might already be recycled if T > window_size and t is early in prefill
                        dp = -1
                        for pid, pst in self._d_pages_seq[b]:
                            if pst == p_start:
                                dp = pid
                                break
                    wp = self._w_pages_seq[b][pos // self.page_size] if self.n_wh_kv > 0 else 0
                d_pages.append(dp)
                w_pages.append(wp)
                offsets.append(off)
                b_indices.append(b)
                t_indices.append(t)

        if layer_idx == 0:
            self._sync_tables_to_device(B)

        b_idx_t = torch.tensor(b_indices, dtype=torch.long, device=self.device)
        t_idx_t = torch.tensor(t_indices, dtype=torch.long, device=self.device)
        off_t = torch.tensor(offsets, dtype=torch.long, device=self.device)

        if self.n_damped_kv > 0:
            dp_t = torch.tensor(d_pages, dtype=torch.long, device=self.device)
            keep = dp_t >= 0
            if keep.any():
                dp_v = dp_t[keep]
                bv = b_idx_t[keep]
                tv = t_idx_t[keep]
                ov = off_t[keep]
                self.k_damped_pool[layer_idx][dp_v, :, ov, :] = k_real[bv, : self.n_damped_kv, tv, :].to(self.dtype)
                self.v_damped_pool[layer_idx][dp_v, :, ov, :] = v[bv, : self.n_damped_kv, tv, :].to(self.dtype)
                self.tau_damped_pool[layer_idx][dp_v, :, ov] = tau[bv, : self.n_damped_q, tv].to(self.tau_dtype)

        if self.n_wh_kv > 0:
            wp_t = torch.tensor(w_pages, dtype=torch.long, device=self.device)
            self.k_wh_pool[layer_idx][wp_t, :, off_t, :] = k_real[b_idx_t, self.n_damped_kv :, t_idx_t, :].to(self.dtype)
            self.v_wh_pool[layer_idx][wp_t, :, off_t, :] = v[b_idx_t, self.n_damped_kv :, t_idx_t, :].to(self.dtype)

        # --- Decode Fast Path (`T == 1`) ---
        if T == 1 and self.use_triton and q_aug.is_cuda and q_aug.dtype in (torch.bfloat16, torch.float16, torch.float32):
            return self._decode_triton(layer_idx, q_aug[:, :, 0, :], tau[:, :, 0], v0, use_vacuum, use_time_decay).unsqueeze(2)

        # --- Exact Reference / Multi-Token Prefill Path ---
        return self._attend_paged_reference(layer_idx, q_aug, tau, v0, use_vacuum, use_time_decay)

    def _decode_triton(
        self,
        layer_idx: int,
        q_step: torch.Tensor,           # [B, H_q, D]
        tau_q: torch.Tensor,            # [B, H_q]
        v0: torch.Tensor,               # [H_kv]
        use_vacuum: bool,
        use_time_decay: bool,
    ) -> torch.Tensor:
        B, H_q, D = q_step.shape
        q_c = q_step.contiguous()
        tau_q_c = tau_q.float().contiguous()
        v0_c = v0.float().contiguous()
        out = torch.empty_like(q_c)

        kd = self.k_damped_pool[layer_idx]
        vd = self.v_damped_pool[layer_idx]
        td = self.tau_damped_pool[layer_idx]
        kw = self.k_wh_pool[layer_idx]
        vw = self.v_wh_pool[layer_idx]

        grid = (B, H_q)
        _epr_paged_decode_kernel[grid](
            q_c, out, v0_c, tau_q_c, self.seq_lens,
            kd, vd, td, self.d_block_table, self.d_block_start, self.d_num_blks,
            kw, vw, self.w_block_table, self.w_num_blks,
            q_c.stride(0), q_c.stride(1),
            out.stride(0), out.stride(1),
            tau_q_c.stride(0),
            kd.stride(0), kd.stride(1), kd.stride(2),
            td.stride(0), td.stride(1),
            self.d_block_table.stride(0),
            kw.stride(0), kw.stride(1), kw.stride(2),
            self.w_block_table.stride(0),
            window_size=self.window_size,
            n_rep=self.n_rep,
            n_damped_q=self.n_damped_q,
            n_damped_kv=self.n_damped_kv,
            USE_VACUUM=use_vacuum,
            USE_TIME_DECAY=use_time_decay,
            PAGE_SIZE=self.page_size,
            HEAD_DIM=D,
        )
        return out

    def _attend_paged_reference(
        self,
        layer_idx: int,
        q_aug: torch.Tensor,            # [B, H_q, T, D]
        tau: torch.Tensor,              # [B, H_q, T]
        v0: torch.Tensor,               # [H_kv]
        use_vacuum: bool,
        use_time_decay: bool,
    ) -> torch.Tensor:
        """Exact block-table reference supporting variable-length batches, T>=1 prefill, and fp64 auditing."""
        B, H_q, T, D = q_aug.shape
        acc_dtype = torch.float64 if q_aug.dtype == torch.float64 else torch.float32
        out = torch.zeros_like(q_aug)

        kd_pool = self.k_damped_pool[layer_idx]
        vd_pool = self.v_damped_pool[layer_idx]
        td_pool = self.tau_damped_pool[layer_idx]
        kw_pool = self.k_wh_pool[layer_idx]
        vw_pool = self.v_wh_pool[layer_idx]

        for b in range(B):
            total_len = self._cpu_seq_lens[b]
            q_positions = torch.arange(total_len - T, total_len, device=self.device)  # [T]

            # 1. Damped Heads
            if self.n_damped_q > 0:
                d_pages = [pid for pid, _ in self._d_pages_seq[b]]
                d_starts = [pst for _, pst in self._d_pages_seq[b]]
                # Gather [n_blks, H_d_kv, P, D] -> [H_d_kv, n_blks*P, D]
                kd_cat = kd_pool[d_pages].permute(1, 0, 2, 3).reshape(self.n_damped_kv, -1, D)
                vd_cat = vd_pool[d_pages].permute(1, 0, 2, 3).reshape(self.n_damped_kv, -1, D)
                td_cat = td_pool[d_pages].permute(1, 0, 2).reshape(self.n_damped_q, -1)
                pos_cat = torch.cat([
                    torch.arange(pst, pst + self.page_size, device=self.device) for pst in d_starts
                ], dim=0)  # [L_d]

                if self.n_rep > 1:
                    kd_cat = kd_cat.repeat_interleave(self.n_rep, dim=0)
                    vd_cat = vd_cat.repeat_interleave(self.n_rep, dim=0)

                qd = q_aug[b, : self.n_damped_q].to(acc_dtype)  # [H_d_q, T, D]
                scores_d = torch.matmul(qd, kd_cat.to(acc_dtype).transpose(-2, -1))  # [H_d_q, T, L_d]
                if use_time_decay:
                    dt = (tau[b, : self.n_damped_q].to(acc_dtype).unsqueeze(-1) - td_cat.to(acc_dtype).unsqueeze(1)).clamp(min=0.0)
                    scores_d = scores_d - dt

                valid_d = (pos_cat.unsqueeze(0) <= q_positions.unsqueeze(1)) & (
                    (q_positions.unsqueeze(1) - pos_cat.unsqueeze(0)) < self.window_size
                )
                scores_d = scores_d.masked_fill(~valid_d.unsqueeze(0), float("-inf"))

                if use_vacuum:
                    v0_d = v0[: self.n_damped_kv].repeat_interleave(self.n_rep).to(acc_dtype).view(-1, 1, 1).expand(-1, T, 1)
                    scores_d = torch.cat([v0_d, scores_d], dim=-1)
                    vd_pad = F.pad(vd_cat.to(acc_dtype), (0, 0, 1, 0), value=0.0)
                    w_d = F.softmax(scores_d, dim=-1)
                    out[b, : self.n_damped_q] = torch.matmul(w_d, vd_pad).to(q_aug.dtype)
                else:
                    w_d = F.softmax(scores_d, dim=-1)
                    out[b, : self.n_damped_q] = torch.matmul(w_d, vd_cat.to(acc_dtype)).to(q_aug.dtype)

            # 2. Wormhole Heads
            if self.n_wh_q > 0:
                w_pages = self._w_pages_seq[b]
                kw_cat = kw_pool[w_pages].permute(1, 0, 2, 3).reshape(self.n_wh_kv, -1, D)[:, :total_len, :]
                vw_cat = vw_pool[w_pages].permute(1, 0, 2, 3).reshape(self.n_wh_kv, -1, D)[:, :total_len, :]
                if self.n_rep > 1:
                    kw_cat = kw_cat.repeat_interleave(self.n_rep, dim=0)
                    vw_cat = vw_cat.repeat_interleave(self.n_rep, dim=0)

                qw = q_aug[b, self.n_damped_q :].to(acc_dtype)
                scores_w = torch.matmul(qw, kw_cat.to(acc_dtype).transpose(-2, -1))
                pos_w = torch.arange(total_len, device=self.device)
                valid_w = pos_w.unsqueeze(0) <= q_positions.unsqueeze(1)
                scores_w = scores_w.masked_fill(~valid_w.unsqueeze(0), float("-inf"))

                if use_vacuum:
                    v0_w = v0[self.n_damped_kv :].repeat_interleave(self.n_rep).to(acc_dtype).view(-1, 1, 1).expand(-1, T, 1)
                    scores_w = torch.cat([v0_w, scores_w], dim=-1)
                    vw_pad = F.pad(vw_cat.to(acc_dtype), (0, 0, 1, 0), value=0.0)
                    w_w = F.softmax(scores_w, dim=-1)
                    out[b, self.n_damped_q :] = torch.matmul(w_w, vw_pad).to(q_aug.dtype)
                else:
                    w_w = F.softmax(scores_w, dim=-1)
                    out[b, self.n_damped_q :] = torch.matmul(w_w, vw_cat.to(acc_dtype)).to(q_aug.dtype)

        return out

    def active_physical_bytes(self) -> int:
        """Actual physical page bytes currently allocated across all active requests (zero fragmentation!)."""
        used_d = self.num_damped_pages - len(self.free_d_pages)
        used_w = self.num_wh_pages - len(self.free_w_pages)
        elem_kv = self.k_damped_pool[0].element_size()
        elem_tau = self.tau_damped_pool[0].element_size()
        b_d = used_d * self.n_layers * self.page_size * (
            2 * self.n_damped_kv * self.head_dim * elem_kv + self.n_damped_q * elem_tau
        )
        b_w = used_w * self.n_layers * self.page_size * (2 * self.n_wh_kv * self.head_dim * elem_kv)
        return b_d + b_w

    def standard_contiguous_bytes(self, max_alloc_len: Optional[int] = None) -> int:
        """Contiguous pre-allocated KV bytes required by non-paged standard MHA/GQA cache."""
        elem_kv = self.k_damped_pool[0].element_size()
        active_seqs = sum(1 for s in self._cpu_seq_lens if s > 0)
        alloc_T = max_alloc_len if max_alloc_len is not None else self.max_seq_len
        return active_seqs * 2 * self.n_layers * self.n_kv_heads * alloc_T * self.head_dim * elem_kv
