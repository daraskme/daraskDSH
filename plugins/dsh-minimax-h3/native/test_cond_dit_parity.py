"""DiT with keyframes / references against ComfyUI: packed positions, then dit_forward on the captured inputs
(context, tags, condition latents, x) at steps 0 and 1 (H3 with SLA statistics carried over)."""
import json
import sys
from pathlib import Path
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build
import sampler

ROOT = HERE.parents[2]
build.build()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def run(case, weights, keyframe_index=None, sparse=True, shifts=(10.0, 3.0), seed=42, guide=None):
    ref = safe_open(str(ROOT / ".runtime/h3-native" / case / "reference.safetensors"), "pt", device="cuda")
    keys = ref.keys()
    latents = []
    while f"dit.call0.payload.cond_video_latents.{len(latents)}" in keys:
        latents.append(ref.get_tensor(f"dit.call0.payload.cond_video_latents.{len(latents)}"))
    context, tags = ref.get_tensor("dit.call0.context"), ref.get_tensor("dit.call0.payload.text_token_tags")
    video, audio = ref.get_tensor("dit.call0.x.0"), ref.get_tensor("dit.call0.x.1")
    audio_keyframes = []
    if guide:  # long-video join: 1-step video keyframes, then one audio keyframe, then the references
        from safetensors.torch import load_file
        data = load_file(str(guide))
        keyframe_index = sorted(int(k[6:]) for k in data if k.startswith("video."))
        audio_keyframes = [(float(k[6:]), ref.get_tensor("dit.call0.payload.cond_audio_latents.0")) for k in data if k.startswith("audio.")]
    keyframes = list(zip(keyframe_index, latents)) if keyframe_index else []
    refs = latents[len(keyframes):]
    rows, audio_rows, positions = sampler.conditioning(context.shape[1], list(video.shape), audio.shape[-1], keyframes, refs, seed,
                                                       audio_keyframes)
    n_key = sum(z.shape[2] for _, z in keyframes) * (video.shape[3] // 2) * (video.shape[4] // 2)
    want = ref.get_tensor("layout.position_ids").cpu()
    print(case, "positions", "exact" if torch.equal(positions, want) else f"max {(positions - want).abs().max():.2e}")
    stats = [torch.zeros(56, 128, device="cuda") for _ in range(100)] if sparse else []
    scale = shifts[0] / shifts[1]
    for call in (0, 1):
        x = [ref.get_tensor(f"dit.call{call}.x.0"), ref.get_tensor(f"dit.call{call}.x.1")]
        with torch.inference_mode():
            out = torch.ops.dsh_h3.dit_forward(weights, x[0], x[1], ref.get_tensor(f"dit.call{call}.timestep"), context, tags,
                                               shifts[0], shifts[1], scale, stats, call == 0, rows, positions,
                                               audio_rows, n_key)
        for name, got, k in (("video", out[0], f"dit.call{call}.out.0"), ("audio", out[1], f"dit.call{call}.out.1")):
            w = ref.get_tensor(k)
            print(f"  call{call} {name}", "exact" if torch.equal(got, w) else f"rel {rel(got, w):.2e}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["h3", "ph3"]
    if "h3" in which:
        import generate
        weights = generate.load(generate.DIT)
        torch.ops.dsh_h3.merge_loras(weights, generate.load(ROOT / "plugins/dsh-comfyui/models/loras/H3_VBVR_Pro-V2.safetensors"), 1.0)
        run("cond-h3-i2v", weights, keyframe_index=[0, 38])
        run("cond-h3-ref", weights)
        guide = ROOT / ".runtime/h3-native/cond-h3-long/guide.safetensors"
        if guide.exists():
            run("cond-h3-long", weights, guide=guide)
        del weights
        torch.cuda.empty_cache()
    if "ph3" in which:
        import ph3
        weights = ph3.load_dit()
        run("cond-ph3-edit", weights, sparse=False, shifts=(1.0, 1.0))
        run("cond-ph3-editref", weights, sparse=False, shifts=(1.0, 1.0))
