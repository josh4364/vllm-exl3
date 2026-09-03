from pathlib import Path
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
ext_include = os.environ.get("EXL3_EXT_INCLUDE")
if ext_include:
    ext_include = Path(ext_include)
else:
    try:
        import exllamav3
        ext_include = Path(exllamav3.__file__).resolve().parent / "exllamav3_ext"
    except Exception:
        ext_include = None
include_dirs = [str(ROOT / "csrc")]
if ext_include and ext_include.is_dir():
    include_dirs.append(str(ext_include))

setup(
    name="vllm-exl3-native",
    ext_modules=[
        CUDAExtension(
            name="vllm_exl3_c",
            sources=[str(ROOT / "csrc" / "bindings.cpp"),
                     str(ROOT / "csrc" / "exl3_gemv.cu")],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                    "-gencode=arch=compute_100,code=sm_100",
                    "-gencode=arch=compute_120,code=sm_120",
                    "-gencode=arch=compute_121,code=sm_121",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
