"""Unit test verifying vLLM production integration and native kernel dispatch."""
import pytest

def test_vllm_exl3_module_imports():
    """Verify that src.vllm_exl3.exl3 imports and exposes native kernel configuration."""
    try:
        from vllm_exl3 import exl3
    except ImportError as e:
        pytest.fail(f"Could not import vllm_exl3.exl3: {e}")
        
    assert hasattr(exl3, "native_moe_kernel_available"), "exl3 missing native_moe_kernel_available helper"
    assert hasattr(exl3, "get_moe_kernel_backend"), "exl3 missing get_moe_kernel_backend helper"

def test_vllm_exl3_backend_selection(monkeypatch):
    """Verify environment variable dispatch control (auto, native, exllamav3)."""
    from vllm_exl3 import exl3
    
    monkeypatch.setenv("VLLM_EXL3_MOE_KERNEL", "exllamav3")
    assert exl3.get_moe_kernel_backend() == "exllamav3"
    
    monkeypatch.setenv("VLLM_EXL3_MOE_KERNEL", "native")
    assert exl3.get_moe_kernel_backend() == "native"
    
    monkeypatch.delenv("VLLM_EXL3_MOE_KERNEL", raising=False)
    backend = exl3.get_moe_kernel_backend()
    assert backend in ("native", "exllamav3", "loop")
