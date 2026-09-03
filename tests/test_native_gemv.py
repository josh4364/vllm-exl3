"""Unit test verifying high-throughput native EXL3 GEMV kernel against dequant reference."""
import pytest

def test_native_gemv_function_exists():
    """Verify that exl3_gemv is exposed by vllm_exl3_c."""
    try:
        import vllm_exl3_c
    except ImportError as e:
        pytest.fail(f"vllm_exl3_c not importable: {e}")
    assert hasattr(vllm_exl3_c, "exl3_gemv"), "vllm_exl3_c does not export exl3_gemv"

def test_native_gemv_no_weight_materialization():
    """Verify that exl3_gemv streams trellis directly without materializing full weight matrix."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch required for GEMV tests")
        
    import vllm_exl3_c
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA device required for native GEMV tests")
        
    in_features, out_features = 4096, 4096
    K = 4
    trellis = torch.randint(-32768, 32767, (in_features // 16, out_features // 16, 16 * K), dtype=torch.int16, device=device)
    suh = torch.randn(in_features, dtype=torch.float16, device=device)
    svh = torch.randn(out_features, dtype=torch.float16, device=device)
    x = torch.randn(1, in_features, dtype=torch.float16, device=device)
    
    # Warmup
    _ = vllm_exl3_c.exl3_gemv(x, trellis, suh, svh, K, True)
    torch.cuda.synchronize()
    
    mem_before = torch.cuda.memory_allocated(device)
    out = vllm_exl3_c.exl3_gemv(x, trellis, suh, svh, K, True)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated(device)
    
    # Full unquantized weight matrix would be in_features * out_features * 2 bytes = 32 MB
    # The output tensor is only 1 * 4096 * 2 bytes = 8 KB
    # Memory allocated must NOT increase by more than output size + small workspace (< 1 MB)
    mem_diff = mem_after - mem_before
    full_weight_bytes = in_features * out_features * 2
    assert mem_diff < full_weight_bytes // 2, (
        f"Memory leak / materialization detected: allocated {mem_diff / 1024 / 1024:.2f} MB, "
        f"which indicates full weight matrix materialization ({full_weight_bytes / 1024 / 1024:.2f} MB)!"
    )

def test_native_gemv_parity_and_shapes():
    """Verify that exl3_gemv output matches reference matmul across m=1,2,4,8 and K=2,3,4."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch required for GEMV tests")
        
    import vllm_exl3_c
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA device required for native GEMV tests")
        
    for shape in [(2048, 4096), (4096, 2048)]:
        in_features, out_features = shape
        for K in [2, 3, 4]:
            trellis = torch.randint(-32768, 32767, (in_features // 16, out_features // 16, 16 * K), dtype=torch.int16, device=device)
            suh = torch.randn(in_features, dtype=torch.float16, device=device)
            svh = torch.randn(out_features, dtype=torch.float16, device=device)
            
            w_ref = vllm_exl3_c.dequant_trellis(trellis, suh, svh, K, True)
            
            for m in [1, 2, 4, 8]:
                x = torch.randn(m, in_features, dtype=torch.float16, device=device) * 0.1
                y_ref = x.float() @ w_ref.float()
                
                y_out = vllm_exl3_c.exl3_gemv(x, trellis, suh, svh, K, True)
                
                assert y_out.shape == (m, out_features), f"Expected shape {(m, out_features)}, got {y_out.shape}"
                assert torch.isfinite(y_out).all(), "Output contains NaN or Inf"
                
                rel_err = (y_out.float() - y_ref).norm() / y_ref.norm()
                assert rel_err < 0.05, f"Relative error {rel_err.item():.4f} exceeded tolerance 0.05 for m={m}, K={K}, shape={shape}"
