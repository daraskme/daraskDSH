"""Build libdsh_kitchen.so from the vendored comfy-kitchen CUDA sources (kitchen/vendor).

Uses the CUDA 13.2 toolchain in .runtime/h3-native/toolchain (PyPI nvidia-cuda-nvcc, nvidia-nvvm and
nvidia-cuda-crt 13.2.78; CUDA 12.8 nvcc cannot parse GCC 15's libstdc++) and links the CUDA 13 runtime
that ships with the PyTorch 2.15 + cu132 environment. Target: SM 12.0 (sm_120a).
Usage: python build_kitchen.py [--torch-python PATH]
"""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VENDOR = HERE / "kitchen/vendor"
BUILD = ROOT / ".runtime/h3-native/build"
NVCC = ROOT / ".runtime/h3-native/toolchain/nvidia/cu13/bin/nvcc"
TORCH_CU13 = ROOT / ".runtime/comfy-venv/lib/python3.12/site-packages/nvidia/cu13"
FLAGS = ["-Xcompiler", "-fPIC", "-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
         "--use_fast_math", "-diag-suppress", "550,128,177", "-gencode", "arch=compute_120a,code=sm_120a",
         "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
         f"-I{VENDOR}", f"-I{TORCH_CU13 / 'include'}", f"-I{TORCH_CU13 / 'include/cccl'}"]

objects = []
(BUILD / "obj").mkdir(parents=True, exist_ok=True)
for source in sorted(VENDOR.glob("*.cu")):
    obj = BUILD / "obj" / (source.stem + ".o")
    if not obj.exists() or obj.stat().st_mtime < source.stat().st_mtime:
        print("nvcc", source.name, flush=True)
        subprocess.run([str(NVCC), "-c", *FLAGS, str(source), "-o", str(obj)], check=True)
    objects.append(str(obj))
library = BUILD / "libdsh_kitchen.so"
subprocess.run([str(NVCC), "-shared", "-Xcompiler", "-fPIC", *objects, "-cudart", "none",
                "-Xlinker", str(TORCH_CU13 / "lib/libcudart.so.13"), "-Xlinker", f"-rpath={TORCH_CU13 / 'lib'}",
                "-o", str(library)], check=True)
print(library)
