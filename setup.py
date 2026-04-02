"""
ManthanQuant — TurboQuant KV cache compression for vLLM on DGX Spark.

Build: python setup.py develop
       (or: pip install -e .)

Requires: PyTorch with CUDA, nvcc matching torch.version.cuda
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Target SM 12.1 (GB10) + common architectures
cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0 9.0 12.0 12.1")
os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch

setup(
    name="manthanquant",
    version="0.2.0",
    description="TurboQuant KV cache compression: Lloyd-Max + QJL",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="manthanquant._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/turboquant_kernel.cu",
                "csrc/qjl_kernel.cu",
                "csrc/fused_attention_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
