"""Native Anima text-to-image job (no ComfyUI code): prompt -> Qwen3 + LLM adapter -> ER-SDE/CFG with the DiT -> VAE.

Usage: comfy-venv/bin/python generate.py JOB_DIR   (JOB_DIR/request.json is written by jobs.mjs)
request: prompt, negative, seed, count, steps, cfg, width, height, loras [{path, strength}], fp8, output_dir.
Editing (img2img): image (path), denoise (0-1), optional mask (path, white = redraw) and mask_grow / mask_feather (px).
The picture is resized to width x height, encoded (vae_encode, bit-exact with VAEEncode) and sampled from the
denoise point (KSampler denoise); with a mask only the masked part is redrawn (SetLatentNoiseMask) and the
original pixels are pasted back outside the feathered mask.
Progress goes to JOB_DIR/status.json; images to output_dir as NNN-<seed>.png with the settings in a PNG text chunk.
With fp8 false, bit-exact with ComfyUI's UNETLoader/LoraLoaderModelOnly/KSampler(er_sde, simple)/VAEDecode graph;
fp8 (default) runs the DiT projections as FP8 GEMMs: ~1.4x faster, same composition, fine details differ
(docs/anima-generate-native-plan.md). The caller cancels by killing the process group.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.append(str(ROOT / "plugins/media-shared/native"))  # shared job.py
from job import run  # noqa: E402
MODELS = ROOT / "models"
FRONTEND = ROOT / ".runtime/anima-native/frontend"
KITCHEN_BUILD = ROOT / ".runtime/h3-native/build"  # libdsh_kitchen.so (vendored comfy-kitchen kernels)
BUILD = ROOT / ".runtime/anima-native/generate-build"
NVCC = ROOT / ".runtime/h3-native/toolchain/nvidia/cu13/bin/nvcc"  # CUDA 13.2 (12.8 cannot parse GCC 15 headers)
MAX_T5_TOKENS = 512

def build():
    """Compile fp8.cu (nvcc, sm_120a) and the C++ ops (torch cpp_extension) once; registers torch.ops.dsh_anima_gen."""
    import torch
    from torch.utils import cpp_extension
    BUILD.mkdir(parents=True, exist_ok=True)
    cu13 = Path(torch.__file__).resolve().parents[1] / "nvidia/cu13"
    kernels = BUILD / "libdsh_anima_kernels.so"
    if not kernels.exists() or kernels.stat().st_mtime < (HERE / "fp8.cu").stat().st_mtime:
        subprocess.run([str(NVCC), "-shared", "-Xcompiler", "-fPIC", "-O3", "-std=c++17", "--use_fast_math",
                        "-gencode", "arch=compute_120a,code=sm_120a", f"-I{cu13 / 'include'}", str(HERE / "fp8.cu"),
                        "-cudart", "none", "-Xlinker", str(cu13 / "lib/libcudart.so.13"), "-Xlinker", f"-rpath={cu13 / 'lib'}",
                        "-o", str(kernels)], check=True)
    cpp_extension.load(name="dsh_anima_generate", is_python_module=False, with_cuda=False, build_directory=str(BUILD),
                       sources=[str(HERE / f"{n}.cpp") for n in ("text", "dit", "vae")] +
                               [str(ROOT / "plugins/media-shared/native/kitchen/kitchen_ops.cpp")],
                       extra_cflags=["-O3"],
                       extra_include_paths=[str(cu13 / "include"), str(ROOT / ".runtime/h3-native/toolchain/nvidia/cu13/include")],
                       extra_ldflags=[str(KITCHEN_BUILD / "libdsh_kitchen.so"), f"-Wl,-rpath,{KITCHEN_BUILD}",
                                      str(kernels), f"-Wl,-rpath,{BUILD}",
                                      f"-L{Path(torch.__file__).parent / 'lib'}", "-lc10_cuda"])


def main(job):
    request, update = job.request, job.update
    count = int(request["count"])
    update(stage="loading", total=count, message="モデルを読み込んでいます")
    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from safetensors import safe_open
    from tokenizers import Tokenizer
    import torch
    import lora
    import sampler
    from tokens import prompt_tokens

    started = time.time()
    build()
    with safe_open(str(MODELS / "text_encoders/qwen_3_06b_base.safetensors"), "pt", device="cuda") as f:
        qwen_w = {k: f.get_tensor(k).float() for k in f.keys()}  # ComfyUI runs this encoder in FP32
    with safe_open(str(MODELS / "diffusion_models/anima-base-v1.0.safetensors"), "pt", device="cuda") as f:
        dit_w = {k: f.get_tensor(k) for k in f.keys()}
    with safe_open(str(MODELS / "vae/qwen_image_vae.safetensors"), "pt", device="cuda") as f:
        vae_w = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys() if k.startswith(("decoder.", "conv2."))}
        enc_w = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys() if k.startswith(("encoder.", "conv1."))} \
            if request.get("image") else None
    if request.get("loras"):
        dit_w = lora.merge(dit_w, [(lora.load(item["path"]), float(item["strength"])) for item in request["loras"]])
    fp8 = request.get("fp8", True)
    if fp8:
        dit_w = torch.ops.dsh_anima_gen.prepare_fp8(dit_w)
    qwen_tok = Tokenizer.from_file(str(FRONTEND / "qwen_tokenizer.json"))
    t5_tok = Tokenizer.from_file(str(FRONTEND / "t5_tokenizer.json"))

    def condition(text, label):
        qwen_ids, t5_ids, t5_weights = prompt_tokens(qwen_tok, t5_tok, text)
        if len(t5_ids) > MAX_T5_TOKENS:
            raise ValueError(f"{label}が長すぎます（T5 {len(t5_ids)} トークン、上限 {MAX_T5_TOKENS}）")
        return torch.ops.dsh_anima_gen.crossattn(qwen_w, dit_w, torch.tensor(qwen_ids, device="cuda"),
                                                 torch.tensor(t5_ids, device="cuda"), torch.tensor(t5_weights))

    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    saver = ThreadPoolExecutor(1)  # PNG encoding (~0.2 s) overlaps the next image
    pending = None

    def save(image, info, path, record):
        Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8)).save(path, pnginfo=info)
        job.add_image(record)

    height, width = int(request["height"]), int(request["width"])
    source = soft = None
    if request.get("image"):
        from PIL import ImageFilter
        source = Image.open(request["image"]).convert("RGB")
        if not width or not height:  # keep the picture's aspect at about 1.2 MP, multiples of 16
            scale = min(1.0, (1.2e6 / (source.width * source.height)) ** 0.5)
            width = max(256, min(2048, round(source.width * scale / 16) * 16))
            height = max(256, min(2048, round(source.height * scale / 16) * 16))
            request["width"], request["height"] = width, height
        source = source.resize((width, height), Image.LANCZOS)
        if request.get("mask"):
            m = Image.open(request["mask"]).convert("L").resize((width, height), Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
            grow = int(request.get("mask_grow", 8))
            if grow > 0:
                m = m.filter(ImageFilter.MaxFilter(grow * 2 + 1))
            if not m.getbbox():
                raise ValueError("描き直す範囲（マスク）が空です")
            feather = int(request.get("mask_feather", 12))
            soft = np.asarray(m.filter(ImageFilter.GaussianBlur(feather)) if feather > 0 else m, dtype=np.float32)[..., None] / 255.0
            hard = m
        src_np = np.asarray(source, dtype=np.float32) / 255.0

    with torch.inference_mode():
        cond, uncond = condition(request["prompt"], "プロンプト"), condition(request.get("negative", ""), "ネガティブ")
        image_latent = latent_mask = None
        if source is not None:
            image_latent = torch.ops.dsh_anima_gen.vae_encode(enc_w, torch.from_numpy(src_np).cuda())
            if soft is not None:
                # latent-size mask: the grown hard mask, bilinear to h x w (as the sampler prepares the noise mask)
                t = torch.from_numpy(np.asarray(hard, dtype=np.float32) / 255.0).cuda()[None, None]
                latent_mask = torch.nn.functional.interpolate(t, size=image_latent.shape[-2:], mode="bilinear").clamp(0, 1)[:, :, None]
        update(stage="generating", message=None, load_seconds=time.time() - started)
        for index in range(count):
            seed = int(request["seed"]) + index
            step_started = time.time()

            def progress(i, x, denoised):
                update(step=i + 1, steps=request["steps"])
            if image_latent is None:
                latent = sampler.sample(dit_w, cond, uncond, seed, steps=int(request["steps"]), cfg=float(request["cfg"]),
                                        height=height, width=width, on_step=progress, fp8=fp8)
            else:
                latent = sampler.sample_image(dit_w, cond, uncond, seed, image_latent, steps=int(request["steps"]),
                                              cfg=float(request["cfg"]), denoise=float(request.get("denoise", 0.6)),
                                              mask=latent_mask, on_step=progress, fp8=fp8)
            image = torch.ops.dsh_anima_gen.vae_decode(vae_w, latent).cpu().numpy()
            if soft is not None:  # keep the original pixels outside the redrawn part
                image = src_np * (1.0 - soft) + image * soft
            info = PngInfo()
            info.add_text("parameters", json.dumps({k: request[k] for k in ("prompt", "negative", "steps", "cfg", "width", "height")}
                                                   | ({"denoise": request.get("denoise"), "mask": bool(request.get("mask"))} if source is not None else {})
                                                   | {"seed": seed, "sampler": "er_sde", "scheduler": "simple", "fp8": fp8,
                                                      "loras": [{"name": i["name"], "strength": i["strength"]} for i in request.get("loras", [])]},
                                                   ensure_ascii=False))
            path = output / f"{index + 1:03d}-{seed}.png"
            if pending:
                pending.result()
            pending = saver.submit(save, image, info, path,
                                   {"file": path.name, "seed": seed, "seconds": round(time.time() - step_started, 2)})
        if pending:
            pending.result()
    update(stage="done", message="完了", finished=time.time())


if __name__ == "__main__":
    run(main, {"done": 0, "total": 0})
