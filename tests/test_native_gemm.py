"""Unit test verifying high-throughput tiled EXL3 GEMM kernel for m > 8."""
import time
import pytest

def test_exl3_gemm_function_exists():
    """Verify that exl3_gemm is exported by vllm_exl3_c."""
    try:
        import vllm_exl3_c
    except ImportError as e:
        pytest.skip(f"vllm_exl3_c not importable: {e}")
    assert hasattr(vllm_exl3_c, "exl3_gemm"), "vllm_exl3_c does not export exl3_gemm"

def test_exl3_gemm_parity_and_throughput():
    """Verify numerical parity and >= 7.0 TFLOPS throughput for m in [16, 32, 64, 128].
    
    Note: On the 48-SM DGX Spark GB10, peak dense FP16 cuBLAS matmul at m=128 runs at 27.6 TFLOPS.
    7.0+ TFLOPS represents an 8.7x speedup over sequential row execution and 13.7x over native exl3_gemm (0.59 TFLOPS).
    """
    try:
        import torch
        import vllm_exl3_c
    except ImportError:
        pytest.skip("PyTorch and vllm_exl3_c required")
        
    if not hasattr(vllm_exl3_c, "exl3_gemm"):
        pytest.skip("vllm_exl3_c does not export exl3_gemm")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA device required")
        
    k = 4096
    n = 4096
    K = 2
    
    # Create synthetic weight matrix
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=device)
    suh = torch.randn(k, dtype=torch.float16, device=device) / 64.0
    svh = torch.randn(n, dtype=torch.float16, device=device)
    
    for m in [16, 32, 64, 128]:
        x = torch.randn(m, k, dtype=torch.float16, device=device) * 0.1
        
        # Call verified exl3_gemv row by row as ground truth
        y_ref_rows = [vllm_exl3_c.exl3_gemv(x[i : i + 1], trellis, suh, svh, K, True) for i in range(m)]
        y_ref = torch.cat(y_ref_rows, dim=0)
        
        # GEMM invocation
        y_gemm = vllm_exl3_c.exl3_gemm(x, trellis, suh, svh, K, True)
        
        assert y_gemm.shape == (m, n), f"Expected shape ({m}, {n}), got {y_gemm.shape}"
        assert torch.isfinite(y_gemm).all(), "Output contains NaN or Inf"
        
        rel_err = (y_gemm - y_ref).norm().item() / y_ref.norm().item()
        print(f"m={m:3d}: rel error = {rel_err:.5f}")
        assert rel_err <= 0.005, f"m={m}: rel error {rel_err:.5f} > 0.005"
        
    # Throughput benchmark at m = 128
    m_bench = 128
    x_bench = torch.randn(m_bench, k, dtype=torch.float16, device=device) * 0.1
    for _ in range(10):
        vllm_exl3_c.exl3_gemm(x_bench, trellis, suh, svh, K, True)
    torch.cuda.synchronize()
    
    iters = 100
    t0 = time.time()
    for _ in range(iters):
        vllm_exl3_c.exl3_gemm(x_bench, trellis, suh, svh, K, True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    
    # 2 * m * k * n FLOPs
    flops = 2.0 * m_bench * k * n
    tflops = (flops / dt) / 1e12
    print(f"exl3_gemm m={m_bench} throughput: {tflops:.2f} TFLOPS (TARGET: >= 7.0 TFLOPS)")
    assert tflops >= 7.0, f"Throughput {tflops:.2f} TFLOPS below 7.0 TFLOPS target"
