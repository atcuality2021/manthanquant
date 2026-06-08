"""
ManthanQuant — TurboQuant KV cache compression for vLLM.

Two install modes:

  CPU-only (default, required on DGX Spark GB10 / sm_121):
      pip install -e .
      The active path is pure numpy (manthanquant/cpu_quantize.py) — no nvcc,
      no CUDA toolkit. On GB10 the _C extension is deliberately NOT loaded
      (custom CUDA kernels collide with Triton at module import), so building
      it is pointless here.

  With CUDA kernels (x86 / datacenter QJL + fused-decode path):
      MANTHANQUANT_BUILD_CUDA=1 pip install -e .
      Requires PyTorch with CUDA and an nvcc matching torch.version.cuda.

The build also degrades gracefully: if CUDA is requested but torch/nvcc aren't
available, it falls back to a CPU-only install rather than failing.
"""

import os
from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

# Opt in to the CUDA _C extension. Off by default so `pip install` works on
# GB10 (and any box without a CUDA toolkit). MANTHANQUANT_SKIP_CUDA=1 also
# forces it off explicitly for callers that prefer that spelling.
_want_cuda = (
    os.environ.get("MANTHANQUANT_BUILD_CUDA", "0") == "1"
    and os.environ.get("MANTHANQUANT_SKIP_CUDA", "0") != "1"
)
if _want_cuda:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        # Target SM 12.1 (GB10) + common architectures
        cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0 9.0 12.0 12.1")
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch
        ext_modules = [
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
        ]
        cmdclass = {"build_ext": BuildExtension}
    except Exception as e:  # torch missing / no CUDA — fall back to CPU-only
        print(f"[manthanquant setup] CUDA build unavailable ({e}); CPU-only install.")
        ext_modules = []
        cmdclass = {}

setup(
    name="manthanquant",
    version="0.3.0",
    description="TurboQuant KV cache compression: 3-bit Lloyd-Max (CPU) + QJL (CUDA)",
    packages=find_packages(),
    install_requires=["numpy"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
)
