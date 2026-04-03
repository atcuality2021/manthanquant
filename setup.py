"""
ManthanQuant — TurboQuant KV cache compression for vLLM.

Supports:
  - DGX Spark GB10 (ARM aarch64, sm_121) — pure numpy on unified memory
  - x86 discrete GPUs (RTX 4070/6000, A100, H100) — CUDA kernels on VRAM

Build: pip install -e .
       (or: python setup.py develop)

Requires: PyTorch with CUDA, nvcc matching torch.version.cuda
For GB10 (numpy-only): CUDA extension is optional, install with pip install -e . --no-build-isolation
"""

import os
from setuptools import setup, find_packages

# Try to import CUDA extension build tools
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    HAS_CUDA_BUILD = True
except ImportError:
    HAS_CUDA_BUILD = False

# Target architectures: Ampere, Ada, Hopper, Blackwell
# SM 8.0 = A100, 8.9 = RTX 4090/4070, 9.0 = H100, 12.0/12.1 = GB10 Blackwell
cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0 8.9 9.0 12.0 12.1")
os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch

ext_modules = []
cmdclass = {}

if HAS_CUDA_BUILD:
    try:
        ext_modules = [
            CUDAExtension(
                name="manthanquant._C",
                sources=[
                    "csrc/bindings.cpp",
                    "csrc/turboquant_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "-std=c++17",
                        "--expt-relaxed-constexpr",
                        "--allow-unsupported-compiler",
                        "-Xcudafe", "--diag_suppress=20012",
                    ],
                },
            ),
        ]
        cmdclass = {"build_ext": BuildExtension}
    except OSError:
        # CUDA_HOME not set or nvcc not found — skip CUDA extension
        pass

setup(
    name="manthanquant",
    version="0.4.0",
    description="TurboQuant KV cache compression: Lloyd-Max + QJL for vLLM (GB10 + x86)",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "torch>=2.0"],
)
