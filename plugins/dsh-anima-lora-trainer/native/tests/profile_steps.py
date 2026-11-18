"""Profile the native Anima engine. Usage: profile_steps.py RESOLUTION STEPS OUT_DIR"""
import json, sys, statistics, collections
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE))
import torch
from torch.profiler import profile, ProfilerActivity
import train as T

res, steps, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
warm = 5
args = T.parse_args(['--folder', str(NATIVE.parents[2] / '.runtime/folder-lora-validation/media'),
                     '--output', str(out / 'unused.safetensors'), '--resolution', str(res)])
paths = T._model_paths(args)
device = torch.device('cuda')
samples = T.find_samples(args.folder, False)
latents, contexts, _ = T.prepare_dataset(samples, res, paths, device, out / 'cache')
torch.cuda.empty_cache()
weights = T.load_dit_weights(paths.diffusion)
adapters = T.initialize_adapters(weights, 16, 42)
print('latent', [tuple(l.shape) for l in latents], 'context', [tuple(c.shape) for c in contexts], flush=True)

T.train_native(weights, adapters, latents, contexts, 3, 2e-5, 42)  # build/load + allocator warmup
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    _, losses, elapsed = T.train_native(weights, adapters, latents, contexts, steps, 2e-5, 42)
    torch.cuda.synchronize()
trace = out / f'trace-{res}.json'
prof.export_chrome_trace(str(trace))

events = json.load(open(trace))['traceEvents']
kernels = sorted((e for e in events if e.get('cat') == 'kernel'), key=lambda e: e['ts'])
opt = [e for e in kernels if 'adamw' in e['name'].lower() or 'optimizer' in e['name'].lower()]
assert len(opt) == steps, (len(opt), steps)
# Steady window: from end of optimizer kernel of step `warm` to end of last optimizer kernel.
t0 = opt[warm - 1]['ts'] + opt[warm - 1]['dur']
t1 = opt[-1]['ts'] + opt[-1]['dur']
n = steps - warm
window = [e for e in kernels if t0 <= e['ts'] < t1]
busy, cur_s, cur_e = 0.0, None, None
for e in window:
    s, f = e['ts'], min(e['ts'] + e['dur'], t1)
    if cur_e is None or s > cur_e:
        if cur_e is not None: busy += cur_e - cur_s
        cur_s, cur_e = s, f
    else:
        cur_e = max(cur_e, f)
busy += cur_e - cur_s
wall = t1 - t0

def bucket(name):
    l = name.lower()
    if 'flash' in l or 'fmha' in l or 'attention' in l or 'sdpa' in l or 'cudnn' in l and 'mha' in l: return 'attention'
    if 'gemm' in l or 'cutlass' in l or 'sm90' in l or 'sm100' in l or 'sm120' in l or 'xmma' in l or 'cublas' in l or 'nvjet' in l: return 'gemm'
    if 'adamw' in l or 'optimizer' in l: return 'optimizer'
    if 'reduce' in l or 'norm' in l: return 'reduce/norm'
    if 'copy' in l or 'cast' in l or 'direct_copy' in l: return 'copy/cast'
    return 'elementwise/other'

per_bucket, per_name = collections.Counter(), collections.Counter()
count_bucket, count_name = collections.Counter(), collections.Counter()
for e in window:
    b = bucket(e['name']); per_bucket[b] += e['dur']; count_bucket[b] += 1
    per_name[e['name'][:110]] += e['dur']; count_name[e['name'][:110]] += 1
memcpy = sum(e['dur'] for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset') and t0 <= e['ts'] < t1)

el = elapsed.tolist()[warm:]
result = {
    'resolution': res, 'steps_measured': n,
    'event_step_ms_median': statistics.median(el), 'event_step_ms_p95': sorted(el)[int(0.95 * (len(el) - 1))],
    'wall_ms_per_step': wall / n / 1000, 'gpu_busy_ms_per_step': busy / n / 1000,
    'gpu_idle_pct': 100 * (1 - busy / wall), 'kernels_per_step': len(window) / n,
    'memcpy_ms_per_step': memcpy / n / 1000,
    'buckets_ms_per_step': {k: round(v / n / 1000, 3) for k, v in per_bucket.most_common()},
    'bucket_launches_per_step': {k: round(v / n, 1) for k, v in count_bucket.items()},
    'top_kernels': [(k, round(v / n / 1000, 3), round(count_name[k] / n, 1)) for k, v in per_name.most_common(15)],
    'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
}
print(json.dumps(result, indent=1))
(out / f'profile-{res}.json').write_text(json.dumps(result, indent=1))
