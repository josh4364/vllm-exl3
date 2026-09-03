from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="vllm-exl3-native",
    ext_modules=[
        CppExtension(
            name="vllm_exl3_c",
            sources=[str(ROOT / "csrc" / "bindings.cpp")],
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
