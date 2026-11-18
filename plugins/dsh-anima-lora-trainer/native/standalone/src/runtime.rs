use crate::dataset::{read_pixels, CaptionTokenizers, Pixels, PromptTokens, Sample};
use crate::settings::Settings;
use crate::Result;
use libloading::Library;
use memmap2::Mmap;
use safetensors::tensor::{serialize_to_file, SafeTensors, TensorView};
use safetensors::Dtype;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs::{self, File};
use std::os::unix::ffi::OsStrExt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;
use std::slice;
use tempfile::NamedTempFile;

const BF16: u32 = 1;
const F32: u32 = 2;
const F16: u32 = 3;

// These structures and callback signatures mirror plugins/dsh-anima-lora-trainer/native/standalone.h.
#[repr(C)]
struct RawTensor {
    name: *const c_char,
    data: *const u8,
    bytes: usize,
    shape: *const i64,
    dimensions: usize,
    dtype: u32,
}

impl RawTensor {
    fn empty() -> Self {
        Self {
            name: ptr::null(),
            data: ptr::null(),
            bytes: 0,
            shape: ptr::null(),
            dimensions: 0,
            dtype: 0,
        }
    }

    unsafe fn view(&self) -> Result<TensorView<'_>> {
        let dtype = match self.dtype {
            BF16 => Dtype::BF16,
            F32 => Dtype::F32,
            F16 => Dtype::F16,
            _ => {
                return Err(format!(
                    "Unknown tensor dtype {} from native runtime",
                    self.dtype
                ))
            }
        };
        let shape = slice::from_raw_parts(self.shape, self.dimensions)
            .iter()
            .map(|&dimension| usize::try_from(dimension).map_err(|err| err.to_string()))
            .collect::<Result<Vec<_>>>()?;
        TensorView::new(dtype, shape, slice::from_raw_parts(self.data, self.bytes))
            .map_err(|err| err.to_string())
    }
}

#[repr(C)]
struct RawSample {
    rgb: *const u8,
    height: i64,
    width: i64,
    qwen_ids: *const i64,
    qwen_mask: *const i64,
    t5_ids: *const i64,
    t5_mask: *const i64,
    latent: RawTensor,
    context: RawTensor,
}

#[repr(C)]
struct RawConfig {
    weights: *const RawTensor,
    weight_count: usize,
    sample_count: usize,
    frontend_dir: *const c_char,
    steps: i64,
    rank: i64,
    seed: i64,
    log_every: i64,
    lr: f64,
    epochs: i64,
    self_attn_lr: f64,
    cross_attn_lr: f64,
    mlp_lr: f64,
    adaln_lr: f64,
    weight_decay: f64,
    timestep_scale: f64,
    warmup: f64,
    lr_cycles: i64,
}

#[repr(C)]
struct RawMetrics {
    losses: *const f32,
    step_ms: *const f32,
    step_count: usize,
    cache_hits: usize,
    prepare_seconds: f64,
    train_seconds: f64,
    peak_allocated_bytes: u64,
    peak_reserved_bytes: u64,
    prepare_peak_allocated_bytes: u64,
    prepare_peak_reserved_bytes: u64,
    device_name: *const c_char,
}

type ReadSample = extern "C" fn(*mut c_void, usize, *mut RawSample) -> c_int;
type WriteCache = extern "C" fn(*mut c_void, usize, *const RawTensor, *const RawTensor) -> c_int;
type WriteOutput = extern "C" fn(*mut c_void, *const RawTensor, usize, *const RawMetrics) -> c_int;
type OnError = extern "C" fn(*mut c_void, *const c_char);
type Run = unsafe extern "C" fn(
    *const RawConfig,
    *mut c_void,
    ReadSample,
    Option<WriteCache>,
    WriteOutput,
    OnError,
) -> c_int;

struct Tensor<D> {
    name: CString,
    data: D,
    shape: Vec<i64>,
    dtype: u32,
}

impl<D: AsRef<[u8]>> Tensor<D> {
    fn raw(&self) -> RawTensor {
        RawTensor {
            name: self.name.as_ptr(),
            data: self.data.as_ref().as_ptr(),
            bytes: self.data.as_ref().len(),
            shape: self.shape.as_ptr(),
            dimensions: self.shape.len(),
            dtype: self.dtype,
        }
    }
}

fn tensor<'a>(name: &str, view: TensorView<'a>) -> Result<Tensor<&'a [u8]>> {
    let dtype = match view.dtype() {
        Dtype::BF16 => BF16,
        Dtype::F32 => F32,
        Dtype::F16 => F16,
        dtype => {
            return Err(format!(
                "Expected an unquantized Anima tensor for {name}, found {dtype:?}"
            ))
        }
    };
    let shape = view
        .shape()
        .iter()
        .map(|&dim| i64::try_from(dim).map_err(|err| err.to_string()))
        .collect::<Result<_>>()?;
    let name = CString::new(name).map_err(|err| err.to_string())?;
    Ok(Tensor {
        name,
        data: view.data(),
        shape,
        dtype,
    })
}

fn checkpoint_tensors<'a>(checkpoint: &SafeTensors<'a>) -> Result<Vec<Tensor<&'a [u8]>>> {
    let mut records = BTreeMap::new();
    for (key, view) in checkpoint.iter() {
        let name = key.strip_prefix("net.").unwrap_or(key);
        let name = name.strip_prefix("model.diffusion_model.").unwrap_or(name);
        if name.ends_with(".scale_weight") || name.ends_with(".scale_input") {
            return Err(format!(
                "Expected unquantized Anima weights, found quantization scale {key}"
            ));
        }
        if name.starts_with("llm_adapter.") {
            continue;
        }
        if records.insert(name, tensor(name, view)?).is_some() {
            return Err(format!("Duplicate Anima tensor {name}"));
        }
    }
    if !records.contains_key("x_embedder.proj.1.weight")
        || !records.contains_key("blocks.0.self_attn.q_proj.weight")
    {
        return Err("The checkpoint does not contain Anima DiT weights".into());
    }
    Ok(records.into_values().collect())
}

struct CachedSample {
    latent: Tensor<Vec<u8>>,
    context: Tensor<Vec<u8>>,
}

fn read_cache(path: &Path) -> Option<CachedSample> {
    let bytes = fs::read(path).ok()?;
    let saved = SafeTensors::deserialize(&bytes).ok()?;
    let latent = tensor("latents", saved.tensor("latents").ok()?).ok()?;
    let context = tensor("context", saved.tensor("context").ok()?).ok()?;
    if latent.dtype != BF16 || context.dtype != BF16 || context.shape != [1, 512, 1024] {
        return None;
    }
    if latent.shape.len() != 5
        || latent.shape[..3] != [1, 16, 1]
        || !latent.shape[3..]
            .iter()
            .all(|dim| (2..=1024).contains(dim) && dim % 2 == 0)
    {
        return None;
    }
    fn own(tensor: Tensor<&[u8]>) -> Tensor<Vec<u8>> {
        Tensor {
            name: tensor.name,
            data: tensor.data.to_vec(),
            shape: tensor.shape,
            dtype: tensor.dtype,
        }
    }
    Some(CachedSample {
        latent: own(latent),
        context: own(context),
    })
}

enum PreparedInput {
    Image(Pixels, PromptTokens),
    Cached(CachedSample),
}

impl PreparedInput {
    fn as_raw(&self) -> RawSample {
        match self {
            Self::Image(pixels, tokens) => RawSample {
                rgb: pixels.rgb.as_ptr(),
                height: i64::from(pixels.height),
                width: i64::from(pixels.width),
                qwen_ids: tokens.qwen_ids.as_ptr(),
                qwen_mask: tokens.qwen_mask.as_ptr(),
                t5_ids: tokens.t5_ids.as_ptr(),
                t5_mask: tokens.t5_mask.as_ptr(),
                latent: RawTensor::empty(),
                context: RawTensor::empty(),
            },
            Self::Cached(sample) => RawSample {
                rgb: ptr::null(),
                height: 0,
                width: 0,
                qwen_ids: ptr::null(),
                qwen_mask: ptr::null(),
                t5_ids: ptr::null(),
                t5_mask: ptr::null(),
                latent: sample.latent.raw(),
                context: sample.context.raw(),
            },
        }
    }
}

struct CallbackState<'a> {
    settings: &'a Settings,
    samples: &'a [Sample],
    tokenizers: Option<CaptionTokenizers>,
    input: Option<PreparedInput>,
    error: Option<String>,
    wrote_output: bool,
}

fn callback(
    context: *mut c_void,
    call: impl FnOnce(&mut CallbackState<'_>) -> Result<()>,
) -> c_int {
    // `run` lends this state exclusively to the synchronous C++ trainer.
    let state = unsafe { &mut *context.cast::<CallbackState<'_>>() };
    let result = catch_unwind(AssertUnwindSafe(|| call(state)));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            state.error.get_or_insert(error);
            1
        }
        Err(_) => {
            state
                .error
                .get_or_insert_with(|| "Anima dataset/output callback panicked".into());
            1
        }
    }
}

extern "C" fn read_sample(context: *mut c_void, index: usize, result: *mut RawSample) -> c_int {
    callback(context, |state| {
        let sample = state
            .samples
            .get(index)
            .ok_or("Anima runtime requested an invalid image index")?;
        sample.verify_image()?;
        state.input = if let Some(cached) = sample.cache.as_deref().and_then(read_cache) {
            Some(PreparedInput::Cached(cached))
        } else {
            if state.tokenizers.is_none() {
                state.tokenizers = Some(CaptionTokenizers::load(&state.settings.frontend)?);
            }
            let pixels = read_pixels(&sample.path, state.settings.resolution)?;
            sample.verify_image()?;
            let tokens = state
                .tokenizers
                .as_ref()
                .expect("loaded tokenizers")
                .encode(sample)?;
            Some(PreparedInput::Image(pixels, tokens))
        };
        // The backing input vectors remain in state until the next read callback.
        unsafe { result.write(state.input.as_ref().expect("prepared sample").as_raw()) };
        Ok(())
    })
}

fn publish(path: &Path, replace: bool, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let write_err = |error: &dyn std::fmt::Display| format!("Writing {}: {error}", path.display());
    fs::create_dir_all(parent).map_err(|err| write_err(&err))?;
    let temporary = NamedTempFile::new_in(parent).map_err(|err| write_err(&err))?;
    write(temporary.path())?;
    if replace {
        temporary.persist(path).map_err(|err| write_err(&err))?;
    } else {
        temporary
            .as_file()
            .sync_all()
            .map_err(|err| write_err(&err))?;
        temporary
            .persist_noclobber(path)
            .map_err(|err| write_err(&err))?;
    }
    Ok(())
}

extern "C" fn write_cache(
    context: *mut c_void,
    index: usize,
    latent: *const RawTensor,
    conditioning: *const RawTensor,
) -> c_int {
    callback(context, |state| {
        let sample = state
            .samples
            .get(index)
            .ok_or("Anima runtime returned an invalid cache index")?;
        if let Some(path) = sample.cache.as_ref() {
            let items = unsafe {
                [
                    ("latents", (*latent).view()?),
                    ("context", (*conditioning).view()?),
                ]
            };
            publish(path, true, |temporary| {
                serialize_to_file(items, None, temporary).map_err(|err| err.to_string())
            })?;
        }
        Ok(())
    })
}

#[derive(Serialize)]
struct Summary<'a> {
    output: &'a Path,
    backend: &'static str,
    gpu: &'a str,
    steps: usize,
    images: usize,
    cache_hits: usize,
    resolution: u32,
    rank: i64,
    learning_rate: f64,
    /// Self-attention, cross-attention, MLP, adaLN; null means --learning-rate.
    module_learning_rates: [Option<f64>; 4],
    warmup: f64,
    lr_cycles: i64,
    timestep_scale: f64,
    weight_decay: f64,
    seed: i64,
    preprocessing_seconds: f64,
    preprocessing_peak_vram_bytes: u64,
    preprocessing_peak_reserved_vram_bytes: u64,
    training_seconds: f64,
    training_peak_vram_bytes: u64,
    training_peak_reserved_vram_bytes: u64,
    first_loss: f32,
    last_loss: f32,
    first_step_ms: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    step_loss: Option<&'a [f32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    step_ms: Option<&'a [f32]>,
    median_step_ms: f64,
    median_step_ms_after_first: Option<f64>,
}

fn median(values: &[f32]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f32::total_cmp);
    let center = values.len() / 2;
    if values.len() % 2 == 0 {
        (f64::from(values[center - 1]) + f64::from(values[center])) / 2.0
    } else {
        f64::from(values[center])
    }
}

extern "C" fn write_output(
    context: *mut c_void,
    weights: *const RawTensor,
    count: usize,
    metrics: *const RawMetrics,
) -> c_int {
    callback(context, |state| {
        if count == 0 || state.wrote_output {
            return Err("Anima runtime returned an invalid adapter".into());
        }
        let stats = unsafe { &*metrics };
        if stats.step_count == 0 {
            return Err("Anima runtime returned no training steps".into());
        }
        let losses = unsafe { slice::from_raw_parts(stats.losses, stats.step_count) };
        let elapsed = unsafe { slice::from_raw_parts(stats.step_ms, stats.step_count) };
        if losses.iter().chain(elapsed).any(|value| !value.is_finite()) {
            return Err("Anima training returned a non-finite loss or step timing".into());
        }
        let mut adapters = BTreeMap::new();
        for raw in unsafe { slice::from_raw_parts(weights, count) } {
            let key = unsafe { CStr::from_ptr(raw.name) }
                .to_str()
                .map_err(|err| err.to_string())?;
            if !key.starts_with("blocks.")
                || !(key.ends_with(".lora_A.weight") || key.ends_with(".lora_B.weight"))
            {
                return Err(format!("Unexpected Anima adapter key {key}"));
            }
            let view = unsafe { raw.view()? };
            if view.dtype() != Dtype::BF16 || view.shape().len() != 2 {
                return Err(format!("Unexpected shape or dtype for {key}"));
            }
            if adapters
                .insert(format!("diffusion_model.{key}"), view)
                .is_some()
            {
                return Err(format!("Duplicate Anima adapter weight {key}"));
            }
        }
        let settings = state.settings;
        let gpu = unsafe { CStr::from_ptr(stats.device_name) }.to_string_lossy();
        let mut summary = Summary {
            output: &settings.output,
            backend: "native",
            gpu: &gpu,
            steps: stats.step_count,
            images: state.samples.len(),
            cache_hits: stats.cache_hits,
            resolution: settings.resolution,
            rank: settings.rank,
            learning_rate: settings.lr,
            module_learning_rates: settings.module_lr.map(|rate| (rate >= 0.0).then_some(rate)),
            warmup: settings.warmup,
            lr_cycles: settings.lr_cycles,
            timestep_scale: settings.timestep_scale,
            weight_decay: settings.weight_decay,
            seed: settings.seed,
            preprocessing_seconds: stats.prepare_seconds,
            preprocessing_peak_vram_bytes: stats.prepare_peak_allocated_bytes,
            preprocessing_peak_reserved_vram_bytes: stats.prepare_peak_reserved_bytes,
            training_seconds: stats.train_seconds,
            training_peak_vram_bytes: stats.peak_allocated_bytes,
            training_peak_reserved_vram_bytes: stats.peak_reserved_bytes,
            first_loss: losses[0],
            last_loss: losses[losses.len() - 1],
            first_step_ms: elapsed[0],
            step_loss: Some(losses),
            step_ms: Some(elapsed),
            median_step_ms: median(elapsed),
            median_step_ms_after_first: (elapsed.len() > 1).then(|| median(&elapsed[1..])),
        };
        let full_metrics = settings
            .metrics
            .as_ref()
            .map(|_| serde_json::to_string(&summary))
            .transpose()
            .map_err(|err| err.to_string())?;
        if settings.summary_only {
            summary.step_loss = None;
            summary.step_ms = None;
        }
        let rendered = serde_json::to_string(&summary).map_err(|err| err.to_string())?;
        let metadata = Some(HashMap::from([("format".to_string(), "pt".to_string())]));
        publish(&settings.output, false, |temporary| {
            serialize_to_file(adapters, metadata, temporary).map_err(|err| err.to_string())
        })?;
        if let Some((path, contents)) = settings.metrics.as_ref().zip(full_metrics) {
            publish(path, false, |temporary| {
                fs::write(temporary, format!("{contents}\n")).map_err(|err| err.to_string())
            })?;
        }
        state.wrote_output = true;
        println!("{rendered}");
        Ok(())
    })
}

extern "C" fn on_error(context: *mut c_void, error: *const c_char) {
    callback(context, |state| {
        let error = unsafe { CStr::from_ptr(error) }.to_string_lossy();
        state.error.get_or_insert_with(|| error.into_owned());
        Ok(())
    });
}

pub fn run(settings: &Settings, samples: &[Sample]) -> Result<()> {
    let library = unsafe { Library::new(&settings.runtime) }.map_err(|err| {
        format!(
            "Opening native Anima runtime {}: {err}",
            settings.runtime.display()
        )
    })?;
    let abi_version =
        unsafe { library.get::<unsafe extern "C" fn() -> u32>(b"dsh_anima_abi_version\0") }
            .map_err(|_| {
                "The native Anima runtime is out of date. Rebuild it with dsh anima build"
                    .to_string()
            })?;
    if unsafe { abi_version() } != 3 {
        return Err("Unsupported native Anima ABI. Rebuild it with dsh anima build".into());
    }
    let entry = unsafe { library.get::<Run>(b"dsh_anima_run\0") }.map_err(|err| err.to_string())?;
    let file = File::open(&settings.diffusion)
        .map_err(|err| format!("Opening checkpoint {}: {err}", settings.diffusion.display()))?;
    // Checkpoints are stable during training. The memory map stays live through the last callback.
    let map = unsafe { Mmap::map(&file) }
        .map_err(|err| format!("Mapping checkpoint {}: {err}", settings.diffusion.display()))?;
    let checkpoint = SafeTensors::deserialize(&map)
        .map_err(|err| format!("Reading checkpoint {}: {err}", settings.diffusion.display()))?;
    let weights = checkpoint_tensors(&checkpoint)?;
    let raw_weights: Vec<RawTensor> = weights.iter().map(Tensor::raw).collect();
    let frontend =
        CString::new(settings.frontend.as_os_str().as_bytes()).map_err(|err| err.to_string())?;
    let (steps, epochs) = settings.duration();
    let config = RawConfig {
        weights: raw_weights.as_ptr(),
        weight_count: raw_weights.len(),
        sample_count: samples.len(),
        frontend_dir: frontend.as_ptr(),
        steps,
        rank: settings.rank,
        seed: settings.seed,
        log_every: settings.log_every,
        lr: settings.lr,
        epochs,
        self_attn_lr: settings.module_lr[0],
        cross_attn_lr: settings.module_lr[1],
        mlp_lr: settings.module_lr[2],
        adaln_lr: settings.module_lr[3],
        weight_decay: settings.weight_decay,
        timestep_scale: settings.timestep_scale,
        warmup: settings.warmup,
        lr_cycles: settings.lr_cycles,
    };
    let mut state = CallbackState {
        settings,
        samples,
        tokenizers: None,
        input: None,
        error: None,
        wrote_output: false,
    };
    let cache: Option<WriteCache> = settings
        .cache_dir
        .as_ref()
        .map(|_| write_cache as WriteCache);
    if steps > 0 {
        eprintln!(
            "Encoding {} image(s) at {}px and training {steps} step(s) with the native backend",
            samples.len(),
            settings.resolution,
        );
    } else {
        eprintln!(
            "Encoding {} image(s) at {}px and training {epochs} epoch(s) with the native backend",
            samples.len(),
            settings.resolution,
        );
    }
    let result = unsafe {
        entry(
            &config,
            (&mut state as *mut CallbackState<'_>).cast(),
            read_sample,
            cache,
            write_output,
            on_error,
        )
    };
    if result == 0 && state.wrote_output {
        Ok(())
    } else {
        Err(state.error.unwrap_or_else(|| {
            format!("Native Anima runtime exited without saving an adapter (status {result})")
        }))
    }
}
