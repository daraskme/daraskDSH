use crate::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

const HELP: &str = "Train a ComfyUI-compatible Anima LoRA from a folder of images.

Usage: dsh anima train --folder DIR --output PATH [OPTIONS]

  --folder DIR           Image directory with same-stem UTF-8 captions
  --output PATH          New .safetensors file
  --recursive            Include images in subdirectories
  --resolution N         Nominal aspect ratio bucket resolution [default: 512]
  --rank N               LoRA rank [default: 16]
  --steps N              Number of optimizer steps
  --epochs N             Passes over the dataset [default: 1]
  --learning-rate LR     Constant AdamW learning rate per image [default: 0.00002]

Same-size images up to about 572px are trained two per step (the step uses
twice the learning rate); larger images train one per step.
  --seed N               Random seed [default: 42]
  --backend BACKEND      native, eager, or compile [default: native]
  --config PATH          Folder LoRA config.json
  --diffusion-pipe DIR   Match an explicit diffusion-pipe frontend checkout
  --diffusion PATH       Anima base DiT safetensors
  --vae PATH             Anima VAE safetensors used to prepare the frontend pack
  --text-encoder PATH    Qwen3-0.6B safetensors used to prepare the frontend pack
  --frontend DIR         TorchScript and tokenizer pack [default: managed runtime]
  --cache-dir DIR        Reuse encoded samples across runs
  --metrics PATH         Save the JSON run summary to a new file
  --summary-only         Omit individual steps from stdout JSON
  --log-every N          Print progress every N steps (0 disables) [default: 1]
  -h, --help             Show this help
";

pub struct Settings {
    pub folder: PathBuf,
    pub output: PathBuf,
    pub recursive: bool,
    pub resolution: u32,
    pub rank: i64,
    steps: Option<i64>,
    epochs: i64,
    pub lr: f64,
    pub seed: i64,
    pub log_every: i64,
    pub diffusion: PathBuf,
    pub frontend: PathBuf,
    pub runtime: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub metrics: Option<PathBuf>,
    pub summary_only: bool,
    _frontend_lock: Option<File>,
}

#[derive(Default, Deserialize)]
struct FolderConfig {
    #[serde(default)]
    models: BTreeMap<String, AnimaModels>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnimaModels {
    diffusion: Option<PathBuf>,
    vae: Option<PathBuf>,
    text_encoder: Option<PathBuf>,
}

#[derive(Deserialize)]
struct FrontendMetadata {
    format: u32,
    inputs: BTreeMap<String, FileIdentity>,
}

#[derive(Deserialize)]
struct FileIdentity {
    path: PathBuf,
    size: u64,
    mtime_ns: i128,
}

fn expand_user(path: PathBuf) -> PathBuf {
    match (path.strip_prefix("~"), env::var_os("HOME")) {
        (Ok(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => path,
    }
}

fn default_config() -> PathBuf {
    env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(|| PathBuf::from(".config"))
        .join("comfy-folder-lora/config.json")
}

fn read_folder_config(explicit: Option<PathBuf>) -> Result<FolderConfig> {
    let requested = explicit.or_else(|| {
        env::var_os("COMFY_FOLDER_LORA_CONFIG")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    });
    let path = requested
        .clone()
        .map(expand_user)
        .unwrap_or_else(default_config);
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|err| format!("Reading configuration {}: {err}", path.display())),
        Err(err) if requested.is_none() && err.kind() == std::io::ErrorKind::NotFound => {
            Ok(FolderConfig::default())
        }
        Err(err) => Err(format!("Opening configuration {}: {err}", path.display())),
    }
}

fn checkpoint(path: Option<PathBuf>, flag: &str) -> Result<PathBuf> {
    let path =
        expand_user(path.ok_or_else(|| format!("Set {flag} or configure the Anima checkpoint"))?);
    if !path.is_file() {
        return Err(format!(
            "Anima checkpoint for {flag} is missing: {}",
            path.display()
        ));
    }
    path.canonicalize()
        .map_err(|err| format!("Opening {}: {err}", path.display()))
}

fn check_frontend(
    directory: &Path,
    models: [(&str, &Path); 3],
    pipe: Option<PathBuf>,
) -> Result<()> {
    let metadata_path = directory.join("metadata.json");
    let metadata: FrontendMetadata = serde_json::from_slice(
        &fs::read(&metadata_path)
            .map_err(|err| format!("Opening frontend pack {}: {err}", metadata_path.display()))?,
    )
    .map_err(|err| {
        format!(
            "Reading frontend metadata {}: {err}",
            metadata_path.display()
        )
    })?;
    if metadata.format != 1 {
        return Err(format!(
            "Unsupported frontend pack format {}",
            metadata.format
        ));
    }
    for (name, path) in models {
        let expected = metadata
            .inputs
            .get(name)
            .ok_or_else(|| format!("Frontend metadata is missing {name}"))?;
        let actual =
            fs::metadata(path).map_err(|err| format!("Reading {}: {err}", path.display()))?;
        let actual_mtime =
            i128::from(actual.mtime()) * 1_000_000_000 + i128::from(actual.mtime_nsec());
        if path != expected.path
            || actual.len() != expected.size
            || actual_mtime != expected.mtime_ns
        {
            return Err(format!("Frontend pack {} was prepared with a different {name} checkpoint. Rebuild the frontend for {}.", directory.display(), path.display()));
        }
    }
    if let Some(pipe) = pipe {
        let pipe = expand_user(pipe)
            .canonicalize()
            .map_err(|err| format!("Opening diffusion-pipe: {err}"))?;
        let expected = metadata
            .inputs
            .get("qwen_tokenizer")
            .ok_or("Frontend metadata is missing qwen_tokenizer")?;
        if expected.path != pipe.join("configs/qwen3_06b/tokenizer.json") {
            return Err(format!(
                "Frontend pack {} was prepared from a different diffusion-pipe checkout",
                directory.display()
            ));
        }
    }
    for filename in [
        "vae.pt",
        "conditioner.pt",
        "qwen_tokenizer.json",
        "qwen_tokenizer_config.json",
        "t5_tokenizer.json",
    ] {
        if !directory.join(filename).is_file() {
            return Err(format!(
                "Frontend pack is missing {}",
                directory.join(filename).display()
            ));
        }
    }
    Ok(())
}

fn integer(value: OsString, flag: &str) -> Result<i64> {
    value
        .to_str()
        .and_then(|text| text.parse().ok())
        .ok_or_else(|| format!("{flag} requires an integer"))
}

fn lock_frontend(directory: &Path) -> Result<Option<File>> {
    let mut filename = directory
        .file_name()
        .ok_or("Invalid frontend pack directory")?
        .to_os_string();
    filename.push(".lock");
    let path = directory.with_file_name(filename);
    match File::open(&path) {
        Ok(file) => {
            file.lock_shared()
                .map_err(|err| format!("Locking frontend pack {}: {err}", directory.display()))?;
            Ok(Some(file))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(format!(
            "Opening frontend pack lock {}: {err}",
            path.display()
        )),
    }
}

impl Settings {
    /// `(steps, epochs)` for the runtime: exactly one is positive. With epochs,
    /// the runtime counts steps after pairing same-size images into batches.
    pub fn duration(&self) -> (i64, i64) {
        match self.steps {
            Some(steps) => (steps, 0),
            None => (0, self.epochs),
        }
    }

    pub fn parse() -> Result<Option<Self>> {
        let mut args = env::args_os().skip(1).peekable();
        if args.peek().is_none() {
            print!("{HELP}");
            return Ok(None);
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(4)
            .expect("workspace root");
        let managed = root.join(".runtime/anima-native");
        let mut folder = None;
        let mut output = None;
        let mut recursive = false;
        let mut resolution = 512;
        let mut rank = 16;
        let mut steps = None;
        let mut epochs = None;
        let mut lr = 2e-5_f64;
        let mut seed = 42;
        let mut log_every = 1;
        let mut cfg = None;
        let mut pipe = None;
        let mut diffusion = None;
        let mut vae = None;
        let mut text_encoder = None;
        let mut frontend = env::var_os("DSH_ANIMA_FRONTEND").map(PathBuf::from);
        let mut cache_dir = None;
        let mut metrics = None;
        let mut summary_only = false;
        while let Some(arg) = args.next() {
            let original = arg.to_str().ok_or("Expected a UTF-8 option name")?;
            let (flag, attached) = original
                .split_once('=')
                .map_or((original, None), |(k, v)| (k, Some(v)));
            let mut value = || {
                attached
                    .map(OsString::from)
                    .or_else(|| args.next())
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            match flag {
                "-h" | "--help" if attached.is_none() => {
                    print!("{HELP}");
                    return Ok(None);
                }
                "--folder" => folder = Some(value()?.into()),
                "--output" => output = Some(value()?.into()),
                "--recursive" if attached.is_none() => recursive = true,
                "--resolution" => resolution = integer(value()?, flag)?,
                "--rank" => rank = integer(value()?, flag)?,
                "--steps" => steps = Some(integer(value()?, flag)?),
                "--epochs" => epochs = Some(integer(value()?, flag)?),
                "--learning-rate" => {
                    lr = value()?
                        .to_str()
                        .and_then(|text| text.parse().ok())
                        .ok_or("Invalid learning rate")?
                }
                "--seed" => seed = integer(value()?, flag)?,
                "--log-every" => log_every = integer(value()?, flag)?,
                "--backend" if value()? == "native" => (),
                "--config" => cfg = Some(value()?.into()),
                "--diffusion-pipe" => pipe = Some(value()?.into()),
                "--diffusion" => diffusion = Some(value()?.into()),
                "--vae" => vae = Some(value()?.into()),
                "--text-encoder" => text_encoder = Some(value()?.into()),
                "--frontend" => frontend = Some(value()?.into()),
                "--cache-dir" => cache_dir = Some(value()?.into()),
                "--metrics" => metrics = Some(value()?.into()),
                "--summary-only" if attached.is_none() => summary_only = true,
                _ => {
                    return Err(format!(
                        "Unsupported native Anima option {original}; see --help"
                    ))
                }
            }
        }

        if resolution <= 0 || resolution > 5792 || resolution % 16 != 0 {
            return Err(
                "--resolution must be a positive multiple of 16 no greater than 5792".into(),
            );
        }
        if rank <= 0 || steps.is_some_and(|n| n <= 0) || epochs.is_some_and(|n| n <= 0) {
            return Err("--rank, --steps, and --epochs must be positive".into());
        }
        if steps.is_some() && epochs.is_some() {
            return Err("Use either --steps or --epochs".into());
        }
        if !lr.is_finite() || lr <= 0.0 || seed < 0 || log_every < 0 {
            return Err(
                "Use a positive finite --learning-rate and nonnegative --seed and --log-every"
                    .into(),
            );
        }
        let folder = expand_user(folder.ok_or("--folder is required")?);
        let output = std::path::absolute(expand_user(output.ok_or("--output is required")?))
            .map_err(|err| err.to_string())?;
        if !output
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        {
            return Err("--output must end in .safetensors".into());
        }
        let metrics = metrics
            .map(expand_user)
            .map(std::path::absolute)
            .transpose()
            .map_err(|err| err.to_string())?;
        if metrics.as_ref() == Some(&output) {
            return Err("--output and --metrics must use different paths".into());
        }
        for path in std::iter::once(&output).chain(metrics.iter()) {
            match fs::symlink_metadata(path) {
                Ok(_) => return Err(format!("Refusing to overwrite {}", path.display())),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => (),
                Err(err) => return Err(format!("Checking {}: {err}", path.display())),
            }
        }
        let mut config = read_folder_config(cfg)?;
        let models = config.models.remove("anima").unwrap_or(AnimaModels {
            diffusion: None,
            vae: None,
            text_encoder: None,
        });
        let diffusion = checkpoint(diffusion.or(models.diffusion), "--diffusion")?;
        let vae = checkpoint(vae.or(models.vae), "--vae")?;
        let text_encoder = checkpoint(text_encoder.or(models.text_encoder), "--text-encoder")?;
        let frontend = frontend
            .map(expand_user)
            .unwrap_or_else(|| managed.join("frontend"));
        let frontend_lock = lock_frontend(&frontend)?;
        check_frontend(
            &frontend,
            [
                ("diffusion", &diffusion),
                ("vae", &vae),
                ("text_encoder", &text_encoder),
            ],
            pipe,
        )?;
        let runtime = env::var_os("DSH_ANIMA_RUNTIME")
            .map(PathBuf::from)
            .unwrap_or_else(|| managed.join("libdsh_anima_runtime.so"));

        Ok(Some(Self {
            folder,
            output,
            recursive,
            resolution: resolution as u32,
            rank,
            steps,
            epochs: epochs.unwrap_or(1),
            lr,
            seed,
            log_every,
            diffusion,
            frontend,
            runtime,
            cache_dir: cache_dir.map(expand_user),
            metrics,
            summary_only,
            _frontend_lock: frontend_lock,
        }))
    }
}
