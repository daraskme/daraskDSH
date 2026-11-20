"""Build and load the CUDA LibTorch extension for Anima training."""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


SOURCE = Path(__file__).resolve().parent
WORKSPACE = SOURCE.parents[2]
BUILD_DIR = WORKSPACE / ".runtime" / "anima-native" / "extension"
RUNTIME_LIBRARY = BUILD_DIR.parent / "libdsh_anima_runtime.so"


def _compile_kernel(source_name, symbol, header_name):
    torch_path = Path(torch.__file__).resolve().parent
    wheel_nvrtc = torch_path.parent / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.12"
    try:
        nvrtc = ctypes.CDLL(str(wheel_nvrtc) if wheel_nvrtc.is_file() else "libnvrtc.so.12")
    except OSError as error:
        raise RuntimeError("Native Anima needs CUDA 12 NVRTC to build the Anima CUDA kernels") from error

    program_type = ctypes.c_void_p
    char_pointer = ctypes.c_char_p
    char_array = ctypes.POINTER(char_pointer)
    nvrtc.nvrtcGetErrorString.argtypes = (ctypes.c_int,)
    nvrtc.nvrtcGetErrorString.restype = char_pointer
    nvrtc.nvrtcCreateProgram.argtypes = (
        ctypes.POINTER(program_type), char_pointer, char_pointer, ctypes.c_int, char_array, char_array,
    )
    nvrtc.nvrtcCompileProgram.argtypes = (program_type, ctypes.c_int, char_array)
    nvrtc.nvrtcGetProgramLogSize.argtypes = (program_type, ctypes.POINTER(ctypes.c_size_t))
    nvrtc.nvrtcGetProgramLog.argtypes = (program_type, ctypes.c_void_p)
    nvrtc.nvrtcGetPTXSize.argtypes = (program_type, ctypes.POINTER(ctypes.c_size_t))
    nvrtc.nvrtcGetPTX.argtypes = (program_type, ctypes.c_void_p)
    nvrtc.nvrtcDestroyProgram.argtypes = (ctypes.POINTER(program_type),)

    def check(code):
        if code:
            raise RuntimeError(nvrtc.nvrtcGetErrorString(code).decode())

    source = (SOURCE / source_name).read_bytes()
    program = program_type()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, source_name.encode(), 0, None, None))
    try:
        options = (char_pointer * 2)(b"--gpu-architecture=compute_80", b"--std=c++17")
        code = nvrtc.nvrtcCompileProgram(program, len(options), options)
        size = ctypes.c_size_t()
        if code:
            check(nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size)))
            log = ctypes.create_string_buffer(size.value)
            check(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(f"Could not compile the Anima CUDA kernel {source_name}:\n{log.value.decode()}")
        check(nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)))
        ptx = ctypes.create_string_buffer(size.value)
        check(nvrtc.nvrtcGetPTX(program, ptx))
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    header = BUILD_DIR / header_name
    content = b'inline constexpr char ' + symbol.encode() + b'[] = R"dsh_anima(' + ptx.value + b')dsh_anima";\n'
    if not header.exists() or header.read_bytes() != content:
        with NamedTemporaryFile(dir=BUILD_DIR, delete=False) as temporary:
            temporary.write(content)
        Path(temporary.name).replace(header)


@cache
def load_extension(*, verbose=False):
    if torch.version.cuda is None:
        raise RuntimeError("Native Anima requires a CUDA build of PyTorch")

    bundled_toolkit = WORKSPACE / ".runtime" / "lora-cuda-12.8"
    if not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")) and bundled_toolkit.is_dir():
        os.environ["CUDA_HOME"] = str(bundled_toolkit)

    if not shutil.which("ninja"):
        python_bin = Path(sys.executable).absolute().parent
        os.environ["PATH"] = f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    if not shutil.which("ninja"):
        raise RuntimeError("Native Anima requires ninja on PATH or in the Python environment")
    os.environ.setdefault("MAX_JOBS", "2")

    from torch.utils import cpp_extension

    if not cpp_extension.CUDA_HOME:
        cpp_extension.CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cpp_extension.CUDA_HOME or not (Path(cpp_extension.CUDA_HOME) / "include/cuda_runtime_api.h").is_file():
        raise RuntimeError("Native Anima needs the CUDA toolkit headers. Set CUDA_HOME to the toolkit directory.")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _compile_kernel("optimizer.cu", "adamw_ptx", "optimizer_ptx.h")
    _compile_kernel("rope.cu", "rope_ptx", "rope_ptx.h")
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    toolkit = Path(cpp_extension.CUDA_HOME)
    cuda_lib = toolkit / ("lib64" if (toolkit / "lib64").is_dir() else "lib")
    return cpp_extension.load(
        name="dsh_anima_native",
        sources=[str(SOURCE / name) for name in ("engine.cpp", "model.cpp", "optimizer.cpp", "rope.cpp", "standalone.cpp")],
        extra_cflags=["-O3"],
        extra_include_paths=[str(BUILD_DIR), str(toolkit / "include")],
        extra_ldflags=["-Wl,--as-needed", f"-Wl,-rpath,{torch_lib}:{cuda_lib}", f"-L{torch_lib}", f"-L{cuda_lib}",
                       "-lc10_cuda", "-ltorch_cuda", "-lcudart"],
        build_directory=str(BUILD_DIR),
        with_cuda=False,
        is_python_module=False,
        verbose=verbose,
    )


def build_standalone():
    runtime = Path(load_extension(verbose=True))
    if not RUNTIME_LIBRARY.is_symlink() or RUNTIME_LIBRARY.resolve() != runtime.resolve():
        temporary = RUNTIME_LIBRARY.with_suffix(f".so.{os.getpid()}")
        temporary.symlink_to(runtime.relative_to(RUNTIME_LIBRARY.parent))
        temporary.replace(RUNTIME_LIBRARY)
    target = BUILD_DIR.parent / "cargo"
    subprocess.run([
        "cargo", "build", "--release", "--locked", "--manifest-path", str(SOURCE / "standalone" / "Cargo.toml"),
        "--target-dir", str(target),
    ], check=True)
    return target / "release" / "dsh-anima"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", action="store_true", help="also build the Rust command-line runtime")
    args = parser.parse_args()
    print(build_standalone() if args.standalone else load_extension(verbose=True))
