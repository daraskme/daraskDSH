"""Native MiniMax H3 video job (no ComfyUI code): prompt (+ pictures) -> Qwen3-VL 32B (layer 50) + token refiner ->
res_multistep 8 steps with the DiT (SLA) -> video ViT3D VAE + audio BigVGAN -> MP4 (H.264 + AAC).
Modes: text to video; image to video (first and/or last frame, MiniMaxH3ImageToVideo); reference to video
(reference pictures named <Picture i> in the prompt, MiniMaxH3ReferenceToVideo with ref_image_size=match).

Usage: comfy-venv/bin/python generate.py JOB_DIR   (JOB_DIR/request.json is written by the DSH plugin)
request: prompt, seed, width, height, length (frames at 24 fps, snapped up to 17k + 5), loras [{path, strength}],
output_dir, and optionally first_frame / last_frame (picture paths) or references [picture paths]. Progress goes to JOB_DIR/status.json; the video to output_dir as <seed>.mp4.
Bit-exact with ComfyUI's "MiniMax H3 - Eros beta7 + VBVR Pro V2" graph up to the uncompressed frames and waveform
(the Qwen output differs ~1e-6, see docs/h3-native-plan.md); pictures: docs/h3-conditioning-native-plan.md. The caller cancels by killing the process group.
"""
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
MODELS = ROOT / "plugins/dsh-comfyui/models"
TEXT_ENCODER = MODELS / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
DIT = MODELS / "diffusion_models/10Eros_Max_h3_8stepTURBO_beta7.safetensors"
VIDEO_VAE = MODELS / "vae/minimax_h3_video_vae_dully_seamless_v1_fp16.safetensors"
AUDIO_VAE = MODELS / "vae/minimax_h3_audio_vae_fp32.safetensors"
TOKENIZER = ROOT / ".runtime/h3-native/build/qwen3vl_tokenizer.json"
FPS = 24

JOB = None
status = {"stage": "starting", "images": [], "started": time.time()}


def update(**changes):
    status.update(changes, updated=time.time())
    with tempfile.NamedTemporaryFile("w", dir=JOB, delete=False, encoding="utf-8") as temporary:
        json.dump(status, temporary, ensure_ascii=False)
    os.replace(temporary.name, JOB / "status.json")


def load(path, prefixes=None, cast=None):
    from safetensors import safe_open
    with safe_open(str(path), "pt", device="cuda") as f:
        keys = [k for k in f.keys() if prefixes is None or k.startswith(prefixes)]
        return {k: f.get_tensor(k) if cast is None else f.get_tensor(k).to(cast) for k in keys}


def save_mp4(path, rgb, waveform, metadata):
    """ComfyUI CreateVideo(24 fps, sRGB) + SaveVideo(mp4, h264): libx264 CRF 18, yuv420p BT.709 limited, AAC."""
    import av
    from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
    with av.open(str(path), mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as output:
        output.metadata["parameters"] = json.dumps(metadata, ensure_ascii=False)
        rate = Fraction(FPS * 1000, 1000)
        video = output.add_stream("h264", rate=rate)
        video.width, video.height, video.pix_fmt = rgb.shape[2], rgb.shape[1], "yuv420p"
        video.options = {"crf": "18"}

        def color(target):
            target.color_primaries, target.color_trc = ColorPrimaries.BT709, ColorTrc.IEC61966_2_1
            target.colorspace, target.color_range = 1, ColorRange.MPEG
        color(video.codec_context)
        wave = waveform[0, :, :math.ceil((32000 / rate) * rgb.shape[0])]
        audio = output.add_stream("aac", rate=32000, layout="stereo")
        frames = (rgb * 255).clamp(0, 255).byte().numpy()
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p", dst_colorspace=1)
            color(frame)
            output.mux(video.encode(frame))
        output.mux(video.encode(None))
        frame = av.AudioFrame.from_ndarray(wave.float().contiguous().numpy(), format="fltp", layout="stereo")
        frame.sample_rate, frame.pts = 32000, 0
        output.mux(audio.encode(frame))
        output.mux(audio.encode(None))


def main():
    request = json.loads((JOB / "request.json").read_text())
    update(stage="loading", message="テキストエンコーダーを読み込んでいます")
    import torch
    from tokenizers import Tokenizer
    import audio
    import build
    import cond
    import pictures
    import sampler
    import vae

    started = time.time()
    build.build()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    seed, width, height, length = (int(request[k]) for k in ("seed", "width", "height", "length"))
    frames = sampler.frame_count(length)
    # (pixel frame index, picture) keyframes: the first frame stretched to the canvas, the last one center-cropped
    keyframes = [(0, pictures.resize(pictures.load(request["first_frame"]), width, height))] if request.get("first_frame") else []
    if request.get("last_frame"):
        keyframes.append((frames - 1, pictures.resize(pictures.load(request["last_frame"]), width, height, crop=True)))
    references = [pictures.h3_reference(pictures.load(path), width, height) for path in request.get("references") or []]
    shown = [image for _, image in keyframes] + references  # what Qwen sees, <Picture 1>, <Picture 2>, ...
    with torch.inference_mode():
        te = load(TEXT_ENCODER, ("model.", "visual.") if shown else ("model.",))
        te = {k: v for k, v in te.items() if not k.endswith("comfy_quant")}
        hidden_states, tags = cond.encode(te, "model.", "visual.", tokenizer, request["prompt"], shown, 50, [])
        hidden = hidden_states[-1]
        del te, hidden_states
        torch.cuda.empty_cache()
        update(message="DiT を読み込んでいます")
        dit = load(DIT)
        for item in request.get("loras", []):
            torch.ops.dsh_h3.merge_loras(dit, load(item["path"]), float(item["strength"]))
        context = torch.ops.dsh_h3.refine_text(dit, hidden).unsqueeze(0)
        video_w = load(VIDEO_VAE, ("decoder.", "post_quant_conv.", "latents_") + (("encoder.", "quant_conv.") if shown else ()),
                       torch.float16)
        keyframes = [(index, vae.encode_image(video_w, image)) for index, image in keyframes]
        references = [vae.encode_image(video_w, image) for image in references]
        audio_w = load(AUDIO_VAE, ("decoder.", "dec_in_proj.", "latents_"))
        update(stage="generating", message=None, load_seconds=round(time.time() - started, 1), step=0, steps=8)

        sample_started = time.time()
        latent = sampler.sample(dit, context, tags, seed, width, height, length,
                                on_step=lambda i, x, d: update(step=i + 1), keyframes=keyframes, refs=references)
        del dit
        torch.cuda.empty_cache()
        update(stage="decoding", message="映像と音声を作っています", sample_seconds=round(time.time() - sample_started, 1))
        rgb = vae.decode_video(video_w, latent[0]).cpu()
        wave = audio.decode(audio_w, latent[1])
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{seed}.mp4"
    save_mp4(path, rgb, wave, {"prompt": request["prompt"], "seed": seed, "width": width, "height": height,
                               "frames": rgb.shape[0], "loras": [{"name": i.get("name"), "strength": i["strength"]}
                                                                for i in request.get("loras", [])]})
    status["images"].append({"file": path.name, "seed": seed, "frames": rgb.shape[0]})
    update(stage="done", message="完了", finished=time.time(), total_seconds=round(time.time() - started, 1))


if __name__ == "__main__":
    JOB = Path(sys.argv[1]).resolve()
    try:
        main()
    except Exception as error:  # reported to the panel; the traceback goes to stderr for the plugin
        import traceback
        traceback.print_exc()
        update(stage="error", message=str(error), finished=time.time())
        sys.exit(1)
