"""Native P-H3 image text-to-image job (no ComfyUI code, no prompt upsampler): prompt -> Qwen3-VL 4B (5 hidden
states) + text_connector + token refiner -> Euler over the workflow's 8 manual sigmas with the int8 DiT -> ViT VAE.

Usage: comfy-venv/bin/python ph3.py JOB_DIR   (JOB_DIR/request.json is written by the DSH plugin)
request: prompt, seed, count, width, height (multiples of 32), output_dir. Progress goes to JOB_DIR/status.json;
images to output_dir as NNN-<seed>.png. The DiT (int8 convrot W8A8, comfy-kitchen kernels) and the sampler are
bit-exact with ComfyUI's "P-H3-Image Text to Image (int8)" graph with the upsampler switched off; the text encoder
differs ~1e-6 and the VAE runs its int8 weights dequantized to FP16 (docs/ph3-image-native-plan.md).
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
MODELS = ROOT / "plugins/dsh-comfyui/models"
DIT = MODELS / "diffusion_models/p-h3-image_int8_convrot.safetensors"
TEXT_ENCODER = MODELS / "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors"
VAE = MODELS / "vae/p-h3-image_vae_int8_convrot.safetensors"
TOKENIZER = ROOT / ".runtime/h3-native/build/qwen3vl_tokenizer.json"
TAPS = [13, 19, 25, 31]  # inputs of these layers + the final-norm output: PH3_IMAGE_HIDDEN_STATES [13, 19, 25, 31, 36]
SIGMAS = [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0]
BUFFERS = ("adaln_t_table", "rope.inv_freq")

JOB = None
status = {"stage": "starting", "done": 0, "total": 0, "images": [], "started": time.time()}


def update(**changes):
    status.update(changes, updated=time.time())
    with tempfile.NamedTemporaryFile("w", dir=JOB, delete=False, encoding="utf-8") as temporary:
        json.dump(status, temporary, ensure_ascii=False)
    os.replace(temporary.name, JOB / "status.json")


def tensors(path, keep=lambda key: True):
    from safetensors import safe_open
    with safe_open(str(path), "pt", device="cuda") as f:
        return {k: f.get_tensor(k) for k in f.keys() if keep(k) and not k.endswith("comfy_quant")}


def load_dit():
    """ComfyUI loads the file's FP32 Linear weights at the model dtype (BF16); its buffers stay FP32."""
    import torch
    return {k: v.bfloat16() if v.dtype == torch.float32 and not k.endswith("weight_scale") and k not in BUFFERS else v
            for k, v in tensors(DIT).items()}


def load_vae():
    """The decoder in FP16 with its int8 convrot projections dequantized (W = (q * scale) H, H the 256 Hadamard)."""
    import torch
    w = tensors(VAE, lambda k: k.startswith(("decoder.", "post_quant_conv.", "latents_")))
    h = torch.ops.dsh_h3.hadamard256()
    for name in [k[:-len(".weight_scale")] for k in w if k.endswith(".weight_scale")]:
        q, scale = w.pop(name + ".weight"), w.pop(name + ".weight_scale")
        rows = q.float() * scale.float().view(-1, 1)
        w[name + ".weight"] = (rows.view(q.shape[0], -1, 256) @ h).view(q.shape)
    return {k: v if k.startswith("latents_") else v.half() for k, v in w.items()}


def condition(te, dit, ids):
    """[1, T, 5376] DiT context: 5 hidden states -> text_connector (FP32) -> condition_proj + token refiner."""
    import torch
    hs = torch.ops.dsh_h3.encode_text_layers(te, "model.language_model.", torch.tensor(ids, device="cuda"), 36, TAPS)
    final = torch.nn.functional.rms_norm(hs[-1], (hs[-1].shape[-1],), te["model.language_model.norm.weight"].float(), 1e-6)
    x = torch.stack(hs[:-1] + [final], dim=1).flatten(1)  # [T, 5 * 2560]
    x = (x - te["text_connector.input_mean"]) / te["text_connector.input_std"]
    x = torch.nn.functional.linear(x, te["text_connector.proj.weight"])
    x = torch.addcmul(te["text_connector.output_mean"], x, te["text_connector.output_std"])
    x = torch.cat([te["text_connector.sink_embedding"].view(1, -1), x[1:]])
    return torch.ops.dsh_h3.refine_text(dit, x).unsqueeze(0), x


def sample(dit, context, tags, seed, width, height, on_step=None):
    """Euler over the manual sigmas, shift 1/1 (audio_scale 1), on a one-frame AV latent; returns [video, audio]."""
    import torch
    import sampler
    sigmas = torch.FloatTensor(SIGMAS).cuda()
    noise = [n.cuda() for n in sampler.initial_noise(seed, [1, 24, 1, height // 16, width // 16], [1, 32, 2, 2])]
    x = [sigmas[0] * n + (1.0 - sigmas[0]) * torch.zeros_like(n) for n in noise]
    for i in range(len(SIGMAS) - 1):
        sigma = sigmas[i]
        out = torch.ops.dsh_h3.dit_forward(dit, x[0].to(torch.bfloat16), x[1].to(torch.bfloat16), sigma.view(1) * 1000,
                                           context, tags, 1.0, 1.0, 1.0, [], False)
        denoised = [xi - o.float() * sigma for xi, o in zip(x, out)]
        if on_step:
            on_step(i, x, denoised)
        x = [xi + (xi - d) / sigma * (sigmas[i + 1] - sigma) for xi, d in zip(x, denoised)]
    return x


def decode(vae_w, latent):
    """[1, 24, 1, H, W] latent -> FP32 RGB [H * 16, W * 16, 3] in [0, 1] (256-px tiles, like ComfyUI)."""
    import vae
    pixels = vae.tiled_decode(vae_w, vae.denormalize(vae_w, latent), vae.ph3_image_pixels)
    return vae.finalize(pixels)[0, :, 0].permute(1, 2, 0)


def main():
    request = json.loads((JOB / "request.json").read_text())
    count = int(request["count"])
    update(stage="loading", total=count, message="モデルを読み込んでいます")
    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import torch
    from tokenizers import Tokenizer
    import build

    started = time.time()
    build.build()
    ids = Tokenizer.from_file(str(TOKENIZER)).encode(request["prompt"], add_special_tokens=False).ids
    with torch.inference_mode():
        dit = load_dit()
        te = tensors(TEXT_ENCODER)
        context, _ = condition(te, dit, ids)
        del te
        vae_w = load_vae()
        torch.cuda.empty_cache()
        tags = torch.ones(len(ids), dtype=torch.int64, device="cuda")  # T2I: every text token is tag 1
        update(stage="generating", message=None, load_seconds=round(time.time() - started, 1))
        output = Path(request["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        width, height = int(request["width"]), int(request["height"])
        for index in range(count):
            seed = int(request["seed"]) + index
            image_started = time.time()
            latent = sample(dit, context, tags, seed, width, height,
                            on_step=lambda i, x, d: update(step=i + 1, steps=len(SIGMAS) - 1))
            rgb = decode(vae_w, latent[0]).cpu().numpy()
            info = PngInfo()
            info.add_text("parameters", json.dumps({"prompt": request["prompt"], "seed": seed, "width": width, "height": height,
                                                    "model": "p-h3-image int8", "sampler": "euler 8 (manual sigmas)"},
                                                   ensure_ascii=False))
            path = output / f"{index + 1:03d}-{seed}.png"
            Image.fromarray(np.clip(255.0 * rgb, 0, 255).astype(np.uint8)).save(path, pnginfo=info)
            status["images"].append({"file": path.name, "seed": seed, "seconds": round(time.time() - image_started, 2)})
            update(done=len(status["images"]))
    update(stage="done", message="完了", finished=time.time())


if __name__ == "__main__":
    JOB = Path(sys.argv[1]).resolve()
    try:
        main()
    except Exception as error:  # reported to the panel; the traceback goes to stderr for the plugin
        import traceback
        traceback.print_exc()
        update(stage="error", message=str(error), finished=time.time())
        sys.exit(1)
