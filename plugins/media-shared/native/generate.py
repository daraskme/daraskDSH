"""Native MiniMax H3 video job (no ComfyUI code): prompt (+ pictures) -> Qwen3-VL 32B (layer 50) + token refiner ->
res_multistep 8 steps with the DiT (SLA) -> video ViT3D VAE + audio BigVGAN -> MP4 (H.264 + AAC).
Modes: text to video; image to video (first and/or last frame, MiniMaxH3ImageToVideo); reference to video
(reference pictures named <Picture i> in the prompt, MiniMaxH3ReferenceToVideo with ref_image_size=match).
Long videos (length > SEGMENT): segments of at most SEGMENT frames, each continuing the previous one like
ComfyUI-H3-Long-Video: the previous latent's last 7 steps (22 frames) as 1-step keyframes at the head and its last second
of audio latent as an audio keyframe ending at the join; seed + i per segment; the overlap is cut after decoding
(docs/h3-long-video-native-plan.md).

Usage: comfy-venv/bin/python generate.py JOB_DIR   (JOB_DIR/request.json is written by the DSH plugin)
request: prompt, seed, width, height, length (frames at 24 fps, snapped up to 17k + 5), loras [{path, strength}],
output_dir, and optionally first_frame / last_frame (picture paths) or references [picture paths]. Progress goes to JOB_DIR/status.json; the video to output_dir as <seed>.mp4.
Bit-exact with ComfyUI's "MiniMax H3 - Eros beta7 + VBVR Pro V2" graph up to the uncompressed frames and waveform
(the Qwen output differs ~1e-6, see docs/h3-native-plan.md); pictures: docs/h3-conditioning-native-plan.md. The caller cancels by killing the process group.
"""
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from job import run  # noqa: E402
MODELS = ROOT / "models"
TEXT_ENCODER = MODELS / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
DIT = MODELS / "diffusion_models/10Eros_Max_h3_8stepTURBO_beta7.safetensors"
VIDEO_VAE = MODELS / "vae/minimax_h3_video_vae_dully_seamless_v1_fp16.safetensors"
AUDIO_VAE = MODELS / "vae/minimax_h3_audio_vae_fp32.safetensors"
TOKENIZER = ROOT / ".runtime/h3-native/build/qwen3vl_tokenizer.json"
FPS = 24
SEGMENT = 362        # frames per segment: the top of H3's trained range (~15 s)
OVERLAP = 22         # frames carried into the next segment (7 latent steps)
OVERLAP_OFFSETS = (0, 1, 5, 9, 13, 17, 18)  # pixel frame of each carried latent step (1, 4, 4, 4, 4 frames per step)
AUDIO_CONTEXT = 40   # audio latent steps carried (1 s at 40 Hz)
SAMPLE_RATE = 32000

def load(path, prefixes=None, cast=None):
    from safetensors import safe_open
    with safe_open(str(path), "pt", device="cuda") as f:
        keys = [k for k in f.keys() if prefixes is None or k.startswith(prefixes)]
        return {k: f.get_tensor(k) if cast is None else f.get_tensor(k).to(cast) for k in keys}


def plan(frames):
    """[(segment frames, overlap)]: the fewest segments of at most SEGMENT frames, all the same length, whose kept frames
    (the first segment whole, the others without the overlap) add up to at least `frames`."""
    if frames <= SEGMENT:
        return [(frames, 0)]
    import sampler
    count = math.ceil((frames - OVERLAP) / (SEGMENT - OVERLAP))
    length = sampler.frame_count(math.ceil((frames - OVERLAP) / count) + OVERLAP)
    return [(length, 0)] + [(length, OVERLAP)] * (count - 1)


def join_keyframes(previous):
    """The next segment's head from the previous segment's latent [video, audio] (CPU):
    -> (video keyframes [(frame, 1-step latent)], audio keyframe (frame, audio latent))."""
    video, sound = previous
    tail = video[:, :, video.shape[2] - len(OVERLAP_OFFSETS):]
    keyframes = [(offset, tail[:, :, k:k + 1]) for k, offset in enumerate(OVERLAP_OFFSETS)]
    frames = sum((1, 4, 4, 4, 4)[k % 5] for k in range(video.shape[2]))
    rescale = 5.0 / 3.0
    overhang = sound.shape[-1] - rescale * frames  # the audio track's rounding against the video's length
    if not -0.5 < overhang < 0.5:
        overhang = 0.0
    end = round(rescale * (OVERLAP + overhang / rescale)) / rescale  # the join, on the 40 Hz grid
    return keyframes, (end - AUDIO_CONTEXT / rescale, sound[..., -AUDIO_CONTEXT:])


def save_mp4(path, rgb, waveform, metadata):
    """ComfyUI CreateVideo(24 fps, sRGB) + SaveVideo(mp4, h264): libx264 CRF 18, yuv420p BT.709 limited, AAC.
    rgb: [F, H, W, 3] in [0, 1], or a list of callables returning such chunks (decoded and written one after another);
    metadata["frames"] is the frame count."""
    import av
    from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
    with av.open(str(path), mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as output:
        output.metadata["parameters"] = json.dumps(metadata, ensure_ascii=False)
        rate = Fraction(FPS * 1000, 1000)
        chunks = rgb if isinstance(rgb, list) else [lambda: rgb]
        frame_total = metadata["frames"]
        video = output.add_stream("h264", rate=rate)
        video.width, video.height, video.pix_fmt = metadata["width"], metadata["height"], "yuv420p"
        video.options = {"crf": "18"}

        def color(target):
            target.color_primaries, target.color_trc = ColorPrimaries.BT709, ColorTrc.IEC61966_2_1
            target.colorspace, target.color_range = 1, ColorRange.MPEG
        color(video.codec_context)
        wave = waveform[0, :, :math.ceil((32000 / rate) * frame_total)]
        audio = output.add_stream("aac", rate=32000, layout="stereo")
        for chunk in chunks:
            for image in chunk().mul(255).clamp(0, 255).byte().numpy():
                frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p", dst_colorspace=1)
                color(frame)
                output.mux(video.encode(frame))
        output.mux(video.encode(None))
        frame = av.AudioFrame.from_ndarray(wave.float().contiguous().numpy(), format="fltp", layout="stereo")
        frame.sample_rate, frame.pts = 32000, 0
        output.mux(audio.encode(frame))
        output.mux(audio.encode(None))


def main(job):
    request, update = job.request, job.update
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
    segments = plan(sampler.frame_count(length))
    # (pixel frame index, picture) keyframes: the first frame stretched to the canvas (first segment), the last one
    # center-cropped (last frame of the last segment)
    keyframes = [(0, pictures.resize(pictures.load(request["first_frame"]), width, height))] if request.get("first_frame") else []
    if request.get("last_frame"):
        keyframes.append((-1, pictures.resize(pictures.load(request["last_frame"]), width, height, crop=True)))
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
        steps = 8 * len(segments)
        update(stage="generating", message=None, load_seconds=round(time.time() - started, 1), step=0, steps=steps,
               segments=len(segments))

        sample_started = time.time()
        latents = []  # per segment [video, audio] on the CPU (tens of MB each)
        for i, (frames, overlap) in enumerate(segments):
            last = i == len(segments) - 1
            head, sound = join_keyframes(latents[-1]) if i else ([], [])
            pinned = [(frames - 1 if index < 0 else index, z) for index, z in keyframes
                      if (index == 0 and i == 0) or (index < 0 and last)]
            if len(segments) > 1:
                update(message=f"区間 {i + 1}/{len(segments)}")
            latent = sampler.sample(dit, context, tags, seed + i, width, height, frames,
                                    on_step=lambda k, x, d, base=8 * i: update(step=base + k + 1),
                                    keyframes=head + pinned, refs=references, audio_keyframes=[sound] if i else [])
            latents.append([z.cpu() for z in latent])
        del dit
        torch.cuda.empty_cache()
        update(stage="decoding", message="映像と音声を作っています", sample_seconds=round(time.time() - sample_started, 1))
        # the audio decodes per segment: the overlap's samples are cut, the rest fitted to the kept frames
        waves = []
        for (frames, overlap), latent in zip(segments, latents):
            wave = audio.decode(audio_w, latent[1].cuda()).cpu()
            if overlap:
                wave = wave[..., round(overlap / FPS * SAMPLE_RATE):]
                want = round((frames - overlap) / FPS * SAMPLE_RATE)
                wave = wave[..., :want] if wave.shape[-1] >= want else torch.nn.functional.pad(wave, (0, want - wave.shape[-1]))
            waves.append(wave)
        wave = torch.cat(waves, -1)

        def chunk(latent, overlap):
            def run():
                with torch.inference_mode():
                    return vae.decode_video(video_w, latent[0].cuda()).cpu()[overlap:]
            return run
        total = sum(frames - overlap for frames, overlap in segments)
        output = Path(request["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{seed}.mp4"
        save_mp4(path, [chunk(latent, overlap) for (_, overlap), latent in zip(segments, latents)] if len(segments) > 1
                 else vae.decode_video(video_w, latents[0][0].cuda()).cpu(), wave,
                 {"prompt": request["prompt"], "seed": seed, "width": width, "height": height, "frames": total,
                  "segments": len(segments), "loras": [{"name": i.get("name") or Path(i["path"]).stem, "strength": i["strength"]}
                                                       for i in request.get("loras", [])]})
    job.status["images"].append({"file": path.name, "seed": seed, "frames": total})
    update(stage="done", message="完了", finished=time.time(), total_seconds=round(time.time() - started, 1))


if __name__ == "__main__":
    run(main)
