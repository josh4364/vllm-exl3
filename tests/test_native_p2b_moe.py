"""Unit test verifying end-to-end fused cooperative MoE decode kernel."""
import time
import pytest
import torch

def test_p2b_moe_function_exists():
    """Verify that p2b_fused_moe is exported by vllm_exl3_c."""
    try:
        import vllm_exl3_c
    except ImportError as e:
        pytest.fail(f"vllm_exl3_c not importable: {e}")
    assert hasattr(vllm_exl3_c, "p2b_fused_moe"), "vllm_exl3_c does not export p2b_fused_moe"

def test_p2b_moe_parity_and_latency():
    """Verify end-to-end numerical parity against sequential reference and benchmark latency."""
    try:
        import vllm_exl3_c
    except ImportError:
        pytest.skip("vllm_exl3_c required")
        
    if not hasattr(vllm_exl3_c, "p2b_fused_moe"):
        pytest.fail("vllm_exl3_c does not export p2b_fused_moe")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA device required")
        
    hidden_size = 4096
    intermediate_size = 2048
    K = 2
    num_active_experts = 8
    m = 1
    
    # Create synthetic expert weights for Gate, Up, and Down projections
    def make_weights(in_f, out_f):
        trellises = [torch.randint(-32768, 32767, (in_f // 16, out_f // 16, 16 * K), dtype=torch.int16, device=device) for _ in range(num_active_experts)]
        suhs = [torch.randn(in_f, dtype=torch.float16, device=device) / 64.0 for _ in range(num_active_experts)]
        svhs = [torch.randn(out_f, dtype=torch.float16, device=device) for _ in range(num_active_experts)]
        t_ptrs = torch.tensor([t.data_ptr() for t in trellises], dtype=torch.int64, device=device)
        u_ptrs = torch.tensor([s.data_ptr() for s in suhs], dtype=torch.int64, device=device)
        v_ptrs = torch.tensor([s.data_ptr() for s in svhs], dtype=torch.int64, device=device)
        return trellises, suhs, svhs, t_ptrs, u_ptrs, v_ptrs
        
    gate_t, gate_u, gate_v, gate_tp, gate_up, gate_vp = make_weights(hidden_size, intermediate_size)
    up_t, up_u, up_v, up_tp, up_up, up_vp = make_weights(hidden_size, intermediate_size)
    down_t, down_u, down_v, down_tp, down_up, down_vp = make_weights(intermediate_size, hidden_size)
    
    expert_indices = torch.arange(num_active_experts, dtype=torch.int32, device=device)
    routing_weights = torch.softmax(torch.randn(m, num_active_experts, dtype=torch.float16, device=device), dim=-1)
    x = torch.randn(m, hidden_size, dtype=torch.float16, device=device) * 0.1
    
    # Sequential Reference computation using verified exl3_gemv
    ref_accum = torch.zeros(m, hidden_size, dtype=torch.float32, device=device)
    for e in range(num_active_experts):
        g = vllm_exl3_c.exl3_gemv(x, gate_t[e], gate_u[e], gate_v[e], K, True).float()
        u = vllm_exl3_c.exl3_gemv(x, up_t[e], up_u[e], up_v[e], K, True).float()
        h = torch.nn.functional.silu(g) * u
        d = vllm_exl3_c.exl3_gemv(h.half(), down_t[e], down_u[e], down_v[e], K, True).float()
        ref_accum += routing_weights[:, e : e + 1].float() * d
        
    out = torch.zeros(m, hidden_size, dtype=torch.float16, device=device)
    
    # Warmup
    for _ in range(10):
        vllm_exl3_c.p2b_fused_moe(x, out, gate_tp, gate_up, gate_vp, up_tp, up_up, up_vp, down_tp, down_up, down_vp, expert_indices, routing_weights, K, K, K, True)
    torch.cuda.synchronize()
    
    # Parity verification
    cos_sim = torch.nn.functional.cosine_similarity(out.float().view(-1), ref_accum.view(-1), dim=0).item()
    print(f"Cosine similarity: {cos_sim:.5f}")
    assert cos_sim >= 0.999, f"Cosine similarity {cos_sim:.5f} < 0.999"
    
    # Latency benchmark
    iters = 100
    t0 = time.time()
    for _ in range(iters):
        vllm_exl3_c.p2b_fused_moe(x, out, gate_tp, gate_up, gate_vp, up_tp, up_up, up_vp, down_tp, down_up, down_vp, expert_indices, routing_weights, K, K, K, True)
    torch.cuda.synchronize()
    dt_us = (time.time() - t0) / iters * 1e6
    print(f"p2b_fused_moe latency: {dt_us:.1f} us (TARGET: <= 300 us)")
    assert dt_us <= 300.0, f"Latency {dt_us:.1f} us exceeded target 300 us"
