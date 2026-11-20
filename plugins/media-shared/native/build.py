"""Build and load the H3 native ops (torch.ops.dsh_h3 / dsh_kitchen) with ComfyUI's venv (torch 2.15 + cu132).

libdsh_kitchen.so (the vendored comfy-kitchen kernels) comes from build_kitchen.py; ninja must be on PATH.
"""
from pathlib import Path

import torch
from torch.utils import cpp_extension

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = ROOT / ".runtime/h3-native/build"


def build():
    cu13 = Path(torch.__file__).resolve().parents[1] / "nvidia/cu13"
    cpp_extension.load(name="dsh_h3_dit", sources=[str(HERE / f) for f in ("te.cpp", "condition.cpp", "dit.cpp", "kitchen/kitchen_ops.cpp")],
                       is_python_module=False, with_cuda=False, build_directory=str(BUILD), extra_cflags=["-O3"],
                       extra_include_paths=[str(cu13 / "include"), str(ROOT / ".runtime/h3-native/toolchain/nvidia/cu13/include")],
                       extra_ldflags=[str(BUILD / "libdsh_kitchen.so"), f"-Wl,-rpath,{BUILD}",
                                      f"-L{Path(torch.__file__).parent / 'lib'}", "-lc10_cuda"], verbose=False)
