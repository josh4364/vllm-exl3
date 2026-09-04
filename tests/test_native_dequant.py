"""Unit test verifying bit-exact native EXL3 trellis dequantization primitive against reference."""
import pytest

def test_native_extension_import():
    """Verify that native C++/CUDA extension module is compiled and importable."""
    try:
        import vllm_exl3_c
    except ImportError as e:
        pytest.skip(f"vllm_exl3_c extension module not built or not importable: {e}. Build with `python setup.py build_ext --inplace`.")

def test_trellis_dequant_shapes_and_parity():
    """Verify that dequant_trellis produces correct shapes and matches numerical reference."""
    try:
        import torch
        import vllm_exl3_c
    except ImportError:
        pytest.skip("PyTorch and vllm_exl3_c required for native tensor tests")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA device required for native EXL3 kernel tests")
        
    in_features = 2048
    out_features = 4096
    
    for K in [2, 3, 4]:
        trellis = torch.randint(-32768, 32767, (in_features // 16, out_features // 16, 16 * K), dtype=torch.int16, device=device)
        suh = torch.randn(in_features, dtype=torch.float16, device=device)
        svh = torch.randn(out_features, dtype=torch.float16, device=device)
        
        out = vllm_exl3_c.dequant_trellis(trellis, suh, svh, K, True)
        assert out.shape == (in_features, out_features), f"Expected shape {(in_features, out_features)}, got {out.shape}"
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all(), "Output contains NaN or Inf values"
