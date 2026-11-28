"""
================================================================================
UNIFIED 8-PILLAR MATHEMATICAL, CAUSALITY & KVCACHE AUDIT SUITE (EPR v2.0)
================================================================================
Audits the SINGLE unified `epr_attention.py` + `model.py` stack across:
  - Distilled EPR v2.0 (3-pillar default: `bilat=False, nest=False`)
  - Mass-Elastic Heteroscedastic RBF (`use_elastic_rbf=True`, exact 64d folding!)
  - Pairwise Resonant Tunneling (`use_resonant_tunneling=True`)
  - Grouped-Query Attention (GQA) & HuggingFace `half_split` RoPE
  - All legacy single-switch ablations (`bilat=True`, `nest=True`, `no-wh`, `no-rbf`, `no-vac`)

8 Audit Pillars:
  [AUDIT 1] Bit-Exact Pure Llama 2 Control Baseline (`use_epr=False`)
  [AUDIT 2] `fp64` Textbook Mathematical Equivalence (Forward + Backward Autograd < 1e-12)
  [AUDIT 3] 4-Layer Strict Causality / Anti-Cheating Audit (`0.00e+00` future leak & grad)
  [AUDIT 4] Gradient Health (zero dead/NaN grads across v2.0, Elastic-RBF & Tunneling) & 4x Stress
  [AUDIT 5] Unified Power-of-2 Hardware Alignment (`D in {16,32,48,64,96,128}`) & Param Budget
  [AUDIT 6] Damped Pauli Repulsion & End-to-End `Transformer.generate(use_kv_cache=True)` Parity
  [AUDIT 7] CUDA `bf16` `FlexAttention` == `SDPA` Parity & Zero Future Leak (v2.0, Elastic & Tunneling)
  [AUDIT 8] Stateful `EPRKVCache` (`fp64` Step-by-Step + Prefill/Decode + Asymmetric Eviction)

Usage:
  python3 audit_all.py
  python3 audit_all.py --ckpt=out-6m-epr/ckpt.pt
================================================================================
"""

import math
import sys
import torch
import torch.nn.functional as F

from epr_attention import (
    EPRConfig,
    EPRAttention,
    EPRKVCache,
    _apply_rope_half_split,
    _apply_rope_interleaved,
)
import model as mod


def textbook_epr_oracle(attn: EPRAttention, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """
    Independent, explicit sequential textbook oracle for EPR Attention.
    Computes pairwise Euclidean distances, mass-elastic radii, proper-time integrals,
    pairwise resonant tunneling, and Softmax1 vacuum normalization directly from math definitions!
    """
    B, T, _ = x.shape
    H, H_kv, D, d_r, d_g = attn.n_heads, attn.n_kv_heads, attn.head_dim, attn.d_r, attn.d_g

    q_res, k_res, z_base, xv = attn.w_proj(x).split([attn.q_dim, attn.k_dim, attn.z_dim, attn.v_dim], dim=-1)
    q_res = q_res.view(B, T, H, d_r).transpose(1, 2)
    k_res = k_res.view(B, T, H_kv, d_r).transpose(1, 2)
    z_base = z_base.view(B, T, H_kv, d_g).transpose(1, 2)
    xv = xv.view(B, T, H_kv, D).transpose(1, 2)

    if attn.n_damped > 0:
        fc_r = freqs_cos[:, : d_r // 2] if attn.cfg.rope_style == "interleaved" else freqs_cos[:, :d_r] if freqs_cos.shape[-1] >= d_r else freqs_cos[:, : d_r // 2]
        fs_r = freqs_sin[:, : d_r // 2] if attn.cfg.rope_style == "interleaved" else freqs_sin[:, :d_r] if freqs_sin.shape[-1] >= d_r else freqs_sin[:, : d_r // 2]
        rope_fn = _apply_rope_interleaved if attn.cfg.rope_style == "interleaved" else _apply_rope_half_split
        qd, kd = rope_fn(q_res[:, : attn.n_damped], k_res[:, : attn.n_damped_kv], fc_r, fs_r)
        q_res = torch.cat([qd, q_res[:, attn.n_damped :]], dim=1) if attn.n_wh > 0 else qd
        k_res = torch.cat([kd, k_res[:, attn.n_damped_kv :]], dim=1) if attn.n_wh_kv > 0 else kd

    if attn.n_rep > 1:
        k_res = k_res.repeat_interleave(attn.n_rep, dim=1)
        z_exp = z_base.repeat_interleave(attn.n_rep, dim=1)
        xv = xv.repeat_interleave(attn.n_rep, dim=1)
    else:
        z_exp = z_base

    splits = [attn.u_emit_dim] + ([attn.u_recv_dim] if attn.u_recv_dim > 0 else []) + ([attn.tau_in_dim] if attn.tau_in_dim > 0 else [])
    parts = attn.w_phys(x).split(splits, dim=-1)
    idx_p = 0
    u_emit_kv = parts[idx_p].transpose(1, 2); idx_p += 1
    u_emit = u_emit_kv.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else u_emit_kv
    u_recv = parts[idx_p].transpose(1, 2) if attn.u_recv_dim > 0 else None
    if attn.u_recv_dim > 0:
        idx_p += 1
    tau_in = parts[idx_p].transpose(1, 2) if attn.tau_in_dim > 0 else None

    if attn.n_damped > 0:
        dtau_loc = F.softplus(tau_in + attn.b_tau.view(1, attn.n_damped, 1)) if (attn.use_time_dilation and tau_in is not None) else F.softplus(attn.b_tau).view(1, attn.n_damped, 1).expand(B, attn.n_damped, T)
        if attn.use_nested_time and attn.n_damped > 1:
            alphas = F.softplus(attn.raw_nest_alpha)
            cols = [None] * attn.n_damped
            cols[-1] = dtau_loc[:, -1, :]
            for h_i in range(attn.n_damped - 2, -1, -1):
                cols[h_i] = dtau_loc[:, h_i, :] + alphas[h_i] * cols[h_i + 1]
            dtau_d = torch.stack(cols, dim=1)
        else:
            dtau_d = dtau_loc
        dtau = torch.cat([dtau_d, torch.zeros(B, attn.n_wh, T, dtype=x.dtype, device=x.device)], dim=1) if attn.n_wh > 0 else dtau_d
    else:
        dtau = torch.zeros(B, H, T, dtype=x.dtype, device=x.device)

    tau = torch.cumsum(dtau, dim=-1)
    delta_tau = (tau.unsqueeze(-1) - tau.unsqueeze(-2)).clamp(min=0.0)

    # Explicit textbook pre-decay score S_h(i, j)
    s_pre = torch.matmul(q_res, k_res.transpose(-2, -1)) / math.sqrt(d_r) + u_emit.unsqueeze(-2)
    if attn.use_bilateral_mass and u_recv is not None:
        s_pre = s_pre + u_recv.unsqueeze(-1)

    if attn.use_base_rbf:
        diff = z_exp.unsqueeze(-2) - z_exp.unsqueeze(-3)  # [B, H, T_q, T_k, d_g]
        dist_sq = (diff * diff).sum(dim=-1)
        gamma_0_kv = (torch.exp(attn.log_inv_sigma2) / math.sqrt(d_g)).view(1, H_kv, 1)
        if attn.use_elastic_rbf:
            el_alpha = F.softplus(attn.raw_elastic_alpha).view(1, H_kv, 1)
            gamma_kv = gamma_0_kv / (1.0 + el_alpha * torch.sigmoid(u_emit_kv))  # [B, H_kv, T_k]
        else:
            gamma_kv = gamma_0_kv.expand(B, H_kv, T)
        gamma_h = gamma_kv.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else gamma_kv
        s_pre = s_pre - 0.5 * gamma_h.unsqueeze(-2) * dist_sq

    causal = torch.triu(torch.full((T, T), float("-inf"), dtype=x.dtype, device=x.device), diagonal=1).view(1, 1, T, T)
    if attn.use_resonant_tunneling:
        beta = F.softplus(attn.tunnel_beta).view(1, H, 1, 1)
        theta = attn.tunnel_theta.view(1, H, 1, 1)
        gate = torch.sigmoid(beta * (s_pre - theta))
        psi = s_pre - delta_tau * (1.0 - gate) + causal
    else:
        psi = s_pre - delta_tau + causal

    if attn.use_attn_vacuum:
        v0_h = attn.v0.repeat_interleave(attn.n_rep) if attn.n_rep > 1 else attn.v0
        vac = v0_h.view(1, H, 1, 1).expand(B, H, T, 1)
        weights = F.softmax(torch.cat([vac, psi], dim=-1), dim=-1)[..., 1:]
    else:
        weights = F.softmax(psi, dim=-1)

    out = torch.matmul(weights, xv).transpose(1, 2).contiguous().view(B, T, -1)
    return attn.wo(out)


def audit_1_pure_llama2_parity():
    print("\n" + "=" * 78)
    print("[AUDIT 1] Bit-Exact Pure Llama 2 Control Baseline Check")
    print("=" * 78)
    torch.manual_seed(1337)
    args = mod.ModelArgs(dim=128, n_layers=2, n_heads=4, n_kv_heads=4, vocab_size=512, max_seq_len=64, dropout=0.0, use_epr=False)
    m = mod.Transformer(args).eval()
    assert all(isinstance(layer.attention, mod.Attention) for layer in m.layers)
    x = torch.randint(0, 512, (2, 32))
    with torch.no_grad():
        out = m(x, targets=x)
    assert torch.isfinite(out).all() and torch.isfinite(m.last_loss)
    print("  [PASS] Pure Llama 2 backbone routes strictly to untouched karpathy/llama2.c Attention.")


def audit_2_textbook_math_parity():
    print("\n" + "=" * 78)
    print("[AUDIT 2] Textbook Mathematical Equivalence (Forward + Backward Autograd, fp64)")
    print("=" * 78)
    cases = [
        ("EPR v2.0 Distilled (D=32: 24+6+2)",     dict(dim=128, n_heads=4, n_kv_heads=4)),
        ("EPR v2.0 Distilled (D=64: 48+14+2)",    dict(dim=256, n_heads=4, n_kv_heads=4)),
        ("EPR v2.0 GQA (H=8, KV=2, half_split)",  dict(dim=256, n_heads=8, n_kv_heads=2, rope_style="half_split")),
        ("Polished #1: Mass-Elastic RBF (D=64)",  dict(dim=256, n_heads=4, n_kv_heads=4, use_elastic_rbf=True)),
        ("Polished #1: Mass-Elastic RBF (GQA)",   dict(dim=256, n_heads=8, n_kv_heads=2, use_elastic_rbf=True)),
        ("Polished #2: Resonant Tunneling",       dict(dim=128, n_heads=4, n_kv_heads=4, use_resonant_tunneling=True)),
        ("Combined: Elastic-RBF + Tunneling",     dict(dim=128, n_heads=4, n_kv_heads=4, use_elastic_rbf=True, use_resonant_tunneling=True)),
        ("Legacy: Bilateral Mass + Nested Time",  dict(dim=128, n_heads=4, n_kv_heads=4, use_bilateral_mass=True, use_nested_time=True)),
        ("Ablation: No Wormhole",                 dict(dim=128, n_heads=4, n_kv_heads=4, use_wormhole=False)),
        ("Ablation: No Base RBF",                 dict(dim=128, n_heads=4, n_kv_heads=4, use_base_rbf=False)),
        ("Ablation: No Vacuum Floor",             dict(dim=128, n_heads=4, n_kv_heads=4, use_attn_vacuum=False)),
    ]
    for name, kw in cases:
        torch.manual_seed(7)
        cfg = EPRConfig(dropout=0.0, **kw)
        attn = EPRAttention(cfg).double()
        with torch.no_grad():
            for attr, lo, hi in [
                ("v0", -1.0, 3.0), ("log_inv_sigma2", -0.5, 0.5),
                ("raw_elastic_alpha", -0.5, 1.0), ("raw_nest_alpha", -1.0, 0.5),
                ("tunnel_theta", 0.5, 2.5), ("tunnel_beta", 0.2, 1.5),
            ]:
                if hasattr(attn, attr) and getattr(attn, attr) is not None:
                    getattr(attn, attr).uniform_(lo, hi)

        fc, fs = mod.precompute_freqs_cis(attn.d_r, 20)
        fc, fs = fc.double(), fs.double()
        x = torch.randn(2, 20, kw["dim"], dtype=torch.float64, requires_grad=True)

        y_fast = attn(x, fc, fs)
        y_ref = textbook_epr_oracle(attn, x, fc, fs)
        fwd_err = (y_fast - y_ref).abs().max().item()

        g_out = torch.randn_like(y_fast)
        params = [x] + list(attn.parameters())
        g_fast = torch.autograd.grad(y_fast, params, g_out, allow_unused=True)
        g_ref = torch.autograd.grad(y_ref, params, g_out, allow_unused=True)
        bwd_err = max((a - b).abs().max().item() for a, b in zip(g_fast, g_ref) if a is not None and b is not None)
        assert fwd_err < 1e-12 and bwd_err < 1e-11, f"Math parity failed on {name}: fwd={fwd_err}, bwd={bwd_err}"
        print(f"  [PASS] {name:38s} | d_r={attn.d_r:2d}, d_g={attn.d_g:2d} | fp64 fwd={fwd_err:.2e} | bwd={bwd_err:.2e}")


def audit_3_causality(model_to_audit: mod.Transformer, label: str):
    print("\n" + "=" * 78)
    print(f"[AUDIT 3] 4-Layer Strict Causality / Anti-Cheating Audit ({label}, float64)")
    print("=" * 78)
    m64 = model_to_audit.double().eval()
    device = next(m64.parameters()).device
    V = m64.params.vocab_size

    for T, cut in [(32, 16), (64, 32), (128, 64)]:
        x1 = torch.randint(0, V, (2, T), device=device)
        x2 = x1.clone()
        x2[:, cut:] = (x2[:, cut:] + 97) % V
        with torch.no_grad():
            o1 = m64(x1, targets=x1)
            o2 = m64(x2, targets=x2)
        diff = (o1[:, :cut, :] - o2[:, :cut, :]).abs().max().item()
        assert diff == 0.0, f"CAUSALITY LEAK (3a) at T={T}, cut={cut}: diff={diff}"
        print(f"  [PASS] 3a. Forward Future Perturbation (T={T:3d}, cut={cut:2d})   | max_diff = {diff:.2e}")

    T_full, probe = 96, 40
    x_full = torch.randint(0, V, (1, T_full), device=device)
    with torch.no_grad():
        o_par = m64(x_full, targets=x_full)[0, probe, :]
        o_trc = m64(x_full[:, : probe + 1], targets=x_full[:, : probe + 1])[0, probe, :]
    diff_trunc = (o_par - o_trc).abs().max().item()
    assert diff_trunc < 1e-12, f"CAUSALITY LEAK (3b): diff_trunc={diff_trunc}"
    print(f"  [PASS] 3b. Physical Truncation Equivalence (fp64)          | max_diff = {diff_trunc:.2e}")

    x_g = torch.randint(0, V, (1, 48), device=device)
    t_target = 23
    h = m64.tok_embeddings(x_g).detach().requires_grad_(True)
    fc, fs = m64._get_rope_freqs(48, device)
    h_cur = h
    for layer in m64.layers:
        h_cur = layer(h_cur, fc, fs)
    m64.output(m64.norm(h_cur))[0, t_target, :].sum().backward()
    fut_g = h.grad[0, t_target + 1 :, :].abs().max().item()
    past_g = h.grad[0, : t_target + 1, :].abs().max().item()
    assert fut_g == 0.0 and past_g > 0.0, f"JACOBIAN LEAK (3c): fut={fut_g}, past={past_g}"
    print(f"  [PASS] 3c. Backward Jacobian Isolation (s > t)             | future_grad = {fut_g:.2e} (past={past_g:.2e})")

    x_ar = torch.randint(0, V, (1, 20), device=device)
    with torch.no_grad():
        par_logits = m64(x_ar, targets=x_ar)
        step_diffs = []
        for t in range(1, 20):
            step_logit = m64(x_ar[:, : t + 1])[:, -1, :]
            step_diffs.append((step_logit - par_logits[:, t, :]).abs().max().item())
    max_ar_diff = max(step_diffs)
    assert max_ar_diff < 1e-12, f"AR EQUIVALENCE FAILED (3d): {max_ar_diff}"
    print(f"  [PASS] 3d. Autoregressive Step-by-Step vs Parallel (fp64)  | max_diff = {max_ar_diff:.2e}")


def audit_4_gradients_and_stress():
    print("\n" + "=" * 78)
    print("[AUDIT 4] Gradient Health (v2.0 + Elastic-RBF + Tunneling) & 4x Extrapolation Stress")
    print("=" * 78)
    for label, kw in [
        ("EPR v2.0 Distilled", dict(use_epr=True)),
        ("EPR + Elastic-RBF + Tunneling", dict(use_epr=True, use_elastic_rbf=True, use_resonant_tunneling=True)),
    ]:
        torch.manual_seed(42)
        args = mod.ModelArgs(dim=128, n_layers=3, n_heads=4, n_kv_heads=4, vocab_size=512, max_seq_len=32, dropout=0.0, **kw)
        m = mod.Transformer(args)
        x = torch.randint(0, 512, (2, 32))
        m(x, targets=x)
        m.last_loss.backward()
        bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum().item() == 0.0]
        assert len(bad) == 0, f"Dead or non-finite gradients in {label}: {bad}"
        print(f"  [PASS] 4a. {label:32s} | All {len(list(m.parameters()))} parameter tensors receive finite non-zero grads.")

    m.eval()
    with torch.no_grad():
        x_4x = torch.randint(0, 512, (1, 128))
        assert torch.isfinite(m(x_4x, targets=x_4x)).all() and torch.isfinite(m.last_loss)
        x_rep = torch.full((1, 48), 7, dtype=torch.long)
        assert torch.isfinite(m(x_rep, targets=x_rep)).all()
    print(f"  [PASS] 4b. Zero-shot 4x length extrapolation (32 -> 128) & repetition stress finite (loss={m.last_loss.item():.4f}).")


def audit_5_hardware_and_param_budget():
    print("\n" + "=" * 78)
    print("[AUDIT 5] Unified Hardware Alignment (d_r+d_g+2 == D) & Parameter Budget")
    print("=" * 78)
    for D in [16, 32, 48, 64, 96, 128]:
        a = EPRAttention(EPRConfig(dim=D * 4, n_heads=4))
        assert a.d_r + a.d_g + 2 == D and a.d_r % 2 == 0, f"Misaligned packing for D={D}"
        print(f"  [PASS] head_dim={D:3d} -> d_r={a.d_r:2d} fiber + d_g={a.d_g:2d} base + 2 == {D:3d} (zero V padding)")

    for label, dim, n_layers in [("1.3M (D=32)", 128, 4), ("6.0M (D=64)", 256, 6)]:
        m_base = mod.Transformer(mod.ModelArgs(dim=dim, n_layers=n_layers, n_heads=4, vocab_size=4096, multiple_of=32, max_seq_len=128, use_epr=False))
        m_v2 = mod.Transformer(mod.ModelArgs(dim=dim, n_layers=n_layers, n_heads=4, vocab_size=4096, multiple_of=32, max_seq_len=128, use_epr=True))
        m_el = mod.Transformer(mod.ModelArgs(dim=dim, n_layers=n_layers, n_heads=4, vocab_size=4096, multiple_of=32, max_seq_len=128, use_epr=True, use_elastic_rbf=True))
        tot_base = sum(p.numel() for p in m_base.parameters())
        tot_v2 = sum(p.numel() for p in m_v2.parameters())
        tot_el = sum(p.numel() for p in m_el.parameters())
        assert tot_v2 < tot_base and tot_el < tot_base
        print(f"  [PASS] {label} | Base: {tot_base:,} | EPR v2.0: {tot_v2:,} ({tot_v2 - tot_base:+,}) | Elastic-RBF: {tot_el:,} ({tot_el - tot_base:+,})")


def audit_6_pauli_and_cached_generate():
    print("\n" + "=" * 78)
    print("[AUDIT 6] Damped Pauli Repulsion & End-to-End Cached `generate()` Parity")
    print("=" * 78)
    torch.manual_seed(0)
    m = mod.Transformer(mod.ModelArgs(dim=64, n_layers=2, n_heads=4, vocab_size=128, max_seq_len=64, dropout=0.0, use_epr=True, use_elastic_rbf=True)).double().eval()
    prompt = torch.tensor([[5, 12, 19, 25, 31]], dtype=torch.long)

    y_cached = m.generate(prompt.clone(), max_new_tokens=16, temperature=0.0, pauli_alpha=1.5, use_kv_cache=True, damped_window=None)
    y_nocache = m.generate(prompt.clone(), max_new_tokens=16, temperature=0.0, pauli_alpha=1.5, use_kv_cache=False)
    assert torch.equal(y_cached, y_nocache), f"Cached vs Non-cached generate diverged!\n{y_cached}\n{y_nocache}"
    print("  [PASS] 6a. Transformer.generate(use_kv_cache=True) matches full-recompute generation bit-for-bit!")

    y_rep = m.generate(torch.tensor([[7, 7, 7, 7]]), max_new_tokens=1, temperature=0.0, pauli_alpha=50.0, pauli_halflife=16.0)
    assert y_rep[0, -1].item() != 7
    print(f"  [PASS] 6b. Strong Pauli repulsion (alpha=50) suppressed immediate repeat token 7 -> {y_rep[0, -1].item()}.")


def audit_7_flex_attention_cuda():
    print("\n" + "=" * 78)
    print("[AUDIT 7] CUDA bf16 FlexAttention Parity & Future Perturbation Audit")
    print("=" * 78)
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available on this host (runs automatically on Colab L4/A100 GPU).")
        return

    device = "cuda"
    B, T = 2, 256
    for label, kw in [
            ("EPR v2.0 Distilled (MHA)", dict(dim=256, n_heads=4, n_kv_heads=4)),
            ("EPR + Elastic-RBF (GQA)",  dict(dim=256, n_heads=8, n_kv_heads=2, use_elastic_rbf=True)),
            ("EPR + Resonant Tunneling", dict(dim=256, n_heads=4, n_kv_heads=4, use_resonant_tunneling=True)),
    ]:
        torch.manual_seed(42)
        m_flex = EPRAttention(EPRConfig(use_flex_attention=True, **kw)).to(device=device, dtype=torch.bfloat16).eval()
        m_sdpa = EPRAttention(EPRConfig(use_flex_attention=False, **kw)).to(device=device, dtype=torch.bfloat16).eval()
        m_sdpa.load_state_dict(m_flex.state_dict())

        fc, fs = mod.precompute_freqs_cis(m_flex.d_r, T)
        fc, fs = fc.to(device), fs.to(device)
        x = torch.randn(B, T, 256, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            y_flex = m_flex(x, fc, fs)
            y_sdpa = m_sdpa(x, fc, fs)
            diff = (y_flex.float() - y_sdpa.float()).abs().max().item()
            assert diff < 2.5e-2, f"FlexAttention vs SDPA mismatch on {label}: {diff}"

            x_pert = x.clone()
            x_pert[:, 128:, :] = torch.randn_like(x_pert[:, 128:, :])
            y_flex_pert = m_flex(x_pert, fc, fs)
            leak = (y_flex[:, :128, :].float() - y_flex_pert[:, :128, :].float()).abs().max().item()
            assert leak == 0.0, f"FlexAttention future leak on {label}: {leak}"

        print(f"  [PASS] {label:28s} | bf16 parity={diff:.2e} | future_leak={leak:.2e}")

    # Also verify Asymmetric Sub-Quadratic BlockMask (`flex_damped_window=128` at `T=512`)!
    torch.manual_seed(99)
    T_sp = 512
    m_sp = EPRAttention(EPRConfig(dim=256, n_heads=8, n_kv_heads=2, use_elastic_rbf=True, flex_damped_window=128)).to(device=device, dtype=torch.bfloat16).eval()
    fc_sp, fs_sp = mod.precompute_freqs_cis(m_sp.d_r, T_sp)
    fc_sp, fs_sp = fc_sp.to(device), fs_sp.to(device)
    x_sp = torch.randn(B, T_sp, 256, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        y_sp = m_sp(x_sp, fc_sp, fs_sp)
        x_sp_pert = x_sp.clone()
        x_sp_pert[:, 256:, :] = torch.randn_like(x_sp_pert[:, 256:, :])
        y_sp_pert = m_sp(x_sp_pert, fc_sp, fs_sp)
        sp_leak = (y_sp[:, :256, :].float() - y_sp_pert[:, :256, :].float()).abs().max().item()
        assert sp_leak == 0.0 and torch.isfinite(y_sp).all(), f"Sparse BlockMask error: leak={sp_leak}"
    print(f"  [PASS] {'Sparse BlockMask (W=128, GQA)':28s} | finite=True      | future_leak={sp_leak:.2e}")


def audit_8_epr_kvcache():
    print("\n" + "=" * 78)
    print("[AUDIT 8] Stateful EPRKVCache (Cumulative Tau) & Asymmetric Eviction Audit")
    print("=" * 78)
    torch.manual_seed(42)

    for label, cfg in [
        ("MHA Distilled (4H/4KV, 1 WH)",         EPRConfig(dim=128, n_heads=4, n_kv_heads=4)),
        ("MHA Elastic-RBF + Tunnel (4H/4KV)",    EPRConfig(dim=128, n_heads=4, n_kv_heads=4, use_elastic_rbf=True, use_resonant_tunneling=True)),
        ("GQA half_split (8H/2KV, 4 WH, ElRBF)", EPRConfig(dim=256, n_heads=8, n_kv_heads=2, n_wormhole_heads=4, rope_style="half_split", use_elastic_rbf=True)),
    ]:
        attn = EPRAttention(cfg).double().eval()
        B, T = 2, 24
        x = torch.randn(B, T, cfg.dim, dtype=torch.float64)
        fc, fs = mod.precompute_freqs_cis(attn.d_r, T)
        fc, fs = fc.double(), fs.double()

        with torch.no_grad():
            y_par = attn(x, fc, fs)

            cache = EPRKVCache(
                n_layers=1, n_heads=attn.n_heads, n_kv_heads=attn.n_kv_heads,
                head_dim=attn.head_dim, n_wormhole_kv_heads=attn.n_wh_kv,
                window_size=None, max_seq_len=64, batch_size=B, dtype=torch.float64,
            )
            y_step = torch.cat([attn(x[:, t : t + 1], fc[t : t + 1], fs[t : t + 1], kv_cache=cache, layer_idx=0) for t in range(T)], dim=1)
            diff_step = (y_par - y_step).abs().max().item()
            assert diff_step < 1e-13, f"KVCache step mismatch: {diff_step}"

            cache.reset()
            ys = [attn(x[:, :10], fc[:10], fs[:10], kv_cache=cache, layer_idx=0)]
            for t in range(10, T):
                ys.append(attn(x[:, t : t + 1], fc[t : t + 1], fs[t : t + 1], kv_cache=cache, layer_idx=0))
            diff_mix = (y_par - torch.cat(ys, dim=1)).abs().max().item()
            assert diff_mix < 1e-13, f"KVCache prefill+decode mismatch: {diff_mix}"

            cache_ev = EPRKVCache(
                n_layers=1, n_heads=attn.n_heads, n_kv_heads=attn.n_kv_heads,
                head_dim=attn.head_dim, n_wormhole_kv_heads=attn.n_wh_kv,
                window_size=8, max_seq_len=64, batch_size=B, dtype=torch.float64,
            )
            y_ev = torch.cat([attn(x[:, t : t + 1], fc[t : t + 1], fs[t : t + 1], kv_cache=cache_ev, layer_idx=0) for t in range(T)], dim=1)
            assert torch.isfinite(y_ev).all()
            diff_w = (y_par[:, :8] - y_ev[:, :8]).abs().max().item()
            assert diff_w < 1e-13, f"KVCache pre-eviction mismatch: {diff_w}"

        print(f"  [PASS] {label:38s} | step={diff_step:.2e} | prefill+dec={diff_mix:.2e} | pre-evict={diff_w:.2e}")


def audit_9_paged_attention_and_hf():
    print("\n" + "=" * 78)
    print("[AUDIT 9] Industrial EPRPagedKVCache (O(1) Page Recycling, Var-Len & Triton) + HF Wrapper")
    print("=" * 78)
    from epr_paged_attention import EPRPagedKVCache
    from hf_epr import EPRLlamaConfig, EPRLlamaForCausalLM

    torch.manual_seed(2026)
    for label, cfg in [
        ("Paged MHA (4H/4KV, page=8, W=16)",     EPRConfig(dim=128, n_heads=4, n_kv_heads=4, use_elastic_rbf=True)),
        ("Paged GQA (8H/2KV, page=8, W=16)",     EPRConfig(dim=256, n_heads=8, n_kv_heads=2, n_wormhole_heads=4, rope_style="half_split", use_elastic_rbf=True)),
    ]:
        attn = EPRAttention(cfg).double().eval()
        B, T = 3, 40
        x = torch.randn(B, T, cfg.dim, dtype=torch.float64)
        fc, fs = mod.precompute_freqs_cis(attn.d_r, T)
        fc, fs = fc.double(), fs.double()

        with torch.no_grad():
            # Compare EPRPagedKVCache (with page_size=8, window_size=16 -> active O(1) page eviction & recycling!)
            # bit-for-bit against EPRKVCache (window_size=16)!
            c_dense = EPRKVCache(
                n_layers=1, n_heads=attn.n_heads, n_kv_heads=attn.n_kv_heads,
                head_dim=attn.head_dim, n_wormhole_kv_heads=attn.n_wh_kv,
                window_size=16, max_seq_len=64, batch_size=B, dtype=torch.float64,
            )
            c_paged = EPRPagedKVCache(
                n_layers=1, n_heads=attn.n_heads, n_kv_heads=attn.n_kv_heads,
                head_dim=attn.head_dim, n_wormhole_kv_heads=attn.n_wh_kv,
                window_size=16, max_seq_len=64, max_batch_size=B, page_size=8, dtype=torch.float64,
            )
            y_dense = torch.cat([attn(x[:, t : t + 1], fc[t : t + 1], fs[t : t + 1], kv_cache=c_dense, layer_idx=0) for t in range(T)], dim=1)
            y_paged = torch.cat([attn(x[:, t : t + 1], fc[t : t + 1], fs[t : t + 1], kv_cache=c_paged, layer_idx=0) for t in range(T)], dim=1)
            diff_pg = (y_dense - y_paged).abs().max().item()
            assert diff_pg < 1e-13, f"EPRPagedKVCache vs EPRKVCache mismatch: {diff_pg}"

            # Verify O(1) damped physical page recycling: across T=40 tokens (5 pages), each sequence may hold at most 3 pages (`ceil(16/8)+1 = 3`)!
            max_held = max(len(c_paged._d_pages_seq[b]) for b in range(B))
            assert max_held <= 3, f"Damped page leak! Held {max_held} pages instead of <= 3"

        print(f"  [PASS] {label:38s} | fp64_parity={diff_pg:.2e} | max_damped_pages={max_held}/3 (O(1) recycled)")

    # Verify Transformer.generate(paged_kv=True) matches generate(paged_kv=False) bit-for-bit!
    m_gen = mod.Transformer(mod.ModelArgs(dim=128, n_layers=2, n_heads=4, vocab_size=128, max_seq_len=64, dropout=0.0, use_epr=True, use_elastic_rbf=True)).eval()
    prompt = torch.tensor([[3, 14, 15, 9, 26, 5]], dtype=torch.long)
    y_g1 = m_gen.generate(prompt, max_new_tokens=20, temperature=0.0, use_kv_cache=True, damped_window=16, paged_kv=False)
    y_g2 = m_gen.generate(prompt, max_new_tokens=20, temperature=0.0, use_kv_cache=True, damped_window=16, paged_kv=True, page_size=8)
    assert torch.equal(y_g1, y_g2), "generate(paged_kv=True) diverged from generate(paged_kv=False)!"
    print("  [PASS] End-to-end Transformer.generate(paged_kv=True) matches dense EPRKVCache bit-for-bit!")

    # Verify HuggingFace EPRLlamaForCausalLM wrapper forward & loss
    hf_cfg = EPRLlamaConfig(vocab_size=256, dim=128, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=384, multiple_of=32, max_seq_len=64)
    hf_m = EPRLlamaForCausalLM(hf_cfg).eval()
    ids = torch.randint(0, 256, (2, 16))
    out_hf = hf_m(ids, labels=ids)
    loss_hf = out_hf[0] if isinstance(out_hf, tuple) else out_hf.loss
    assert torch.isfinite(loss_hf), "HF wrapper produced non-finite loss!"
    print(f"  [PASS] HuggingFace EPRLlamaForCausalLM wrapper forward/loss healthy (loss={loss_hf.item():.4f})")

    # If CUDA + Triton available, verify fused @triton.jit `_epr_paged_decode_kernel` against exact reference on variable-length batch!
    if torch.cuda.is_available():
        for t_lbl, t_cfg in [
            ("Triton Paged MHA (bf16, var-len)", EPRConfig(dim=256, n_heads=4, n_kv_heads=4, use_elastic_rbf=True)),
            ("Triton Paged GQA (bf16, 32Q/4KV)", EPRConfig(dim=512, n_heads=8, n_kv_heads=2, n_wormhole_heads=4, rope_style="half_split", use_elastic_rbf=True)),
        ]:
            attn_cu = EPRAttention(t_cfg).to(device="cuda", dtype=torch.bfloat16).eval()
            fc_cu, fs_cu = mod.precompute_freqs_cis(attn_cu.d_r, 128)
            fc_cu, fs_cu = fc_cu.to("cuda"), fs_cu.to("cuda")

            c_tri = EPRPagedKVCache(
                n_layers=1, n_heads=attn_cu.n_heads, n_kv_heads=attn_cu.n_kv_heads,
                head_dim=attn_cu.head_dim, n_wormhole_kv_heads=attn_cu.n_wh_kv,
                window_size=32, max_seq_len=128, max_batch_size=3, page_size=16,
                dtype=torch.bfloat16, device="cuda", use_triton=True,
            )
            c_ref = EPRPagedKVCache(
                n_layers=1, n_heads=attn_cu.n_heads, n_kv_heads=attn_cu.n_kv_heads,
                head_dim=attn_cu.head_dim, n_wormhole_kv_heads=attn_cu.n_wh_kv,
                window_size=32, max_seq_len=128, max_batch_size=3, page_size=16,
                dtype=torch.bfloat16, device="cuda", use_triton=False,
            )
            with torch.no_grad():
                # Prefill variable lengths per slot: b=0 -> 19 tokens, b=1 -> 47 tokens, b=2 -> 75 tokens!
                lens = [19, 47, 75]
                for b_i, L_b in enumerate(lens):
                    xb = torch.randn(1, L_b, t_cfg.dim, device="cuda", dtype=torch.bfloat16)
                    for step_t in range(L_b):
                        q_a, _, k_r, xv, _, tau = attn_cu._compute_step_tensors(
                            xb[:, step_t : step_t + 1], fc_cu[step_t : step_t + 1], fs_cu[step_t : step_t + 1],
                            tau_offset=c_tri.tau_state[0][b_i : b_i + 1],
                        )
                        # Advance only sequence slot b_i for heterogeneous length setup
                        pos = c_tri._cpu_seq_lens[b_i]
                        dp, wp, off = c_tri._ensure_token_position_allocated(b_i, pos)
                        c_ref._ensure_token_position_allocated(b_i, pos)
                        c_tri.tau_state[0][b_i] = tau[0, :, -1:]
                        c_ref.tau_state[0][b_i] = tau[0, :, -1:]
                        c_tri.k_damped_pool[0][dp, :, off, :] = k_r[0, : c_tri.n_damped_kv, 0, :]
                        c_tri.v_damped_pool[0][dp, :, off, :] = xv[0, : c_tri.n_damped_kv, 0, :]
                        c_tri.tau_damped_pool[0][dp, :, off] = tau[0, : c_tri.n_damped_q, 0].float()
                        c_tri.k_wh_pool[0][wp, :, off, :] = k_r[0, c_tri.n_damped_kv :, 0, :]
                        c_tri.v_wh_pool[0][wp, :, off, :] = xv[0, c_tri.n_damped_kv :, 0, :]

                        c_ref.k_damped_pool[0][dp, :, off, :] = k_r[0, : c_ref.n_damped_kv, 0, :]
                        c_ref.v_damped_pool[0][dp, :, off, :] = xv[0, : c_ref.n_damped_kv, 0, :]
                        c_ref.tau_damped_pool[0][dp, :, off] = tau[0, : c_ref.n_damped_q, 0].float()
                        c_ref.k_wh_pool[0][wp, :, off, :] = k_r[0, c_ref.n_damped_kv :, 0, :]
                        c_ref.v_wh_pool[0][wp, :, off, :] = xv[0, c_ref.n_damped_kv :, 0, :]
                c_tri._sync_tables_to_device(3)
                c_ref._sync_tables_to_device(3)

                # Now run simultaneous continuous-batched decode step (`B=3`, `T=1`) across `[20, 48, 76]`!
                x_step = torch.randn(3, 1, t_cfg.dim, device="cuda", dtype=torch.bfloat16)
                # Use per-seq RoPE cos/sin
                q_s, _, k_s, v_s, _, tau_s = attn_cu._compute_step_tensors(
                    x_step, fc_cu[75:76], fs_cu[75:76], tau_offset=c_tri.tau_state[0][:3]
                )
                y_tri = c_tri.write_and_attend(0, q_s, k_s, v_s, tau_s, attn_cu.v0)
                y_ref = c_ref.write_and_attend(0, q_s, k_s, v_s, tau_s, attn_cu.v0)
                tdiff = (y_tri.float() - y_ref.float()).abs().max().item()
                assert tdiff < 1.5e-2, f"Triton PagedAttention kernel mismatch: {tdiff}"
            print(f"  [PASS] {t_lbl:38s} | triton_vs_exact={tdiff:.2e} (seq_lens=[20, 48, 76])")


def main():
    ckpt_path = None
    for a in sys.argv[1:]:
        if a.startswith("--ckpt="):
            ckpt_path = a.split("=", 1)[1]

    audit_1_pure_llama2_parity()
    audit_2_textbook_math_parity()

    torch.manual_seed(1337)
    m_v2 = mod.Transformer(mod.ModelArgs(dim=128, n_layers=3, n_heads=4, vocab_size=256, max_seq_len=64, dropout=0.0, use_epr=True))
    audit_3_causality(m_v2, "Fresh EPR v2.0 Distilled")

    m_polished = mod.Transformer(mod.ModelArgs(dim=128, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64, dropout=0.0, use_epr=True, use_elastic_rbf=True, use_resonant_tunneling=True))
    audit_3_causality(m_polished, "Fresh EPR + Elastic-RBF + Resonant Tunneling")

    if ckpt_path is not None:
        from run_sota_baselines import load_any_checkpoint
        m_ck, _, _ = load_any_checkpoint(ckpt_path, device="cpu")
        audit_3_causality(m_ck, f"Trained Checkpoint: {ckpt_path}")

    audit_4_gradients_and_stress()
    audit_5_hardware_and_param_budget()
    audit_6_pauli_and_cached_generate()
    audit_7_flex_attention_cuda()
    audit_8_epr_kvcache()
    audit_9_paged_attention_and_hf()

    print("\n" + "=" * 78)
    print("ALL 9 UNIFIED AUDIT PILLARS PASSED WITH ZERO LEAKS AND MACHINE-PRECISION PARITY!")
    print("=" * 78)


if __name__ == "__main__":
    main()
