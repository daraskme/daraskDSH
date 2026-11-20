use crate::Result;
use image::{DynamicImage, ImageReader};
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const TOKEN_COUNT: usize = 512;
const COEFFICIENT_SCALE: i64 = 1 << 22;

pub struct Sample {
    pub path: PathBuf,
    caption: String,
    pub cache: Option<PathBuf>,
    image_stamp: Option<[u8; 32]>,
}

impl Sample {
    pub fn verify_image(&self) -> Result<()> {
        if let Some(expected) = self.image_stamp {
            if file_stamp(&self.path)? != expected {
                return Err(format!(
                    "Image changed during preparation: {}",
                    self.path.display()
                ));
            }
        }
        Ok(())
    }
}

pub struct Pixels {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct PromptTokens {
    pub qwen_ids: Vec<i64>,
    pub qwen_mask: Vec<i64>,
    pub t5_ids: Vec<i64>,
    pub t5_mask: Vec<i64>,
}

pub struct CaptionTokenizers {
    qwen: Tokenizer,
    t5: Tokenizer,
}

fn image_paths(directory: &Path, recursive: bool, paths: &mut Vec<PathBuf>) -> Result<()> {
    let entries = fs::read_dir(directory)
        .map_err(|err| format!("Opening image directory {}: {err}", directory.display()))?;
    for entry in entries {
        let entry = entry
            .map_err(|err| format!("Reading image directory {}: {err}", directory.display()))?;
        let path = entry.path();
        if recursive && entry.file_type().map_err(|err| err.to_string())?.is_dir() {
            image_paths(&path, recursive, paths)?;
        } else if path.extension().and_then(OsStr::to_str).is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff"
            )
        }) && path.is_file()
        {
            paths.push(path);
        }
    }
    Ok(())
}

fn cache_fingerprint(frontend: &Path, runtime: &Path, resolution: u32) -> Result<Sha256> {
    let mut hash = Sha256::new();
    hash.update(include_bytes!("dataset.rs"));
    hash.update(include_bytes!("../Cargo.lock"));
    hash.update(resolution.to_le_bytes());
    hash.update(fs::read(frontend.join("metadata.json")).map_err(|err| err.to_string())?);
    for artifact in [
        "vae.pt",
        "conditioner.pt",
        "qwen_tokenizer.json",
        "qwen_tokenizer_config.json",
        "t5_tokenizer.json",
    ] {
        hash.update(file_stamp(&frontend.join(artifact))?);
    }
    hash.update(file_stamp(runtime)?);
    Ok(hash)
}

fn file_stamp(path: &Path) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    let canonical = path
        .canonicalize()
        .map_err(|err| format!("Opening {}: {err}", path.display()))?;
    let metadata =
        fs::metadata(&canonical).map_err(|err| format!("Reading {}: {err}", path.display()))?;
    hash.update((canonical.as_os_str().as_bytes().len() as u64).to_le_bytes());
    hash.update(canonical.as_os_str().as_bytes());
    hash.update(metadata.dev().to_le_bytes());
    hash.update(metadata.ino().to_le_bytes());
    hash.update(metadata.len().to_le_bytes());
    hash.update(metadata.mtime().to_le_bytes());
    hash.update(metadata.mtime_nsec().to_le_bytes());
    hash.update(metadata.ctime().to_le_bytes());
    hash.update(metadata.ctime_nsec().to_le_bytes());
    Ok(hash.finalize().into())
}

pub fn find_samples(
    folder: &Path,
    recursive: bool,
    resolution: u32,
    frontend: &Path,
    runtime: &Path,
    cache_directory: Option<&Path>,
) -> Result<Vec<Sample>> {
    let mut paths = Vec::new();
    image_paths(folder, recursive, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err(format!("No supported images in {}", folder.display()));
    }
    let shared_fingerprint = cache_directory
        .map(|directory| {
            fs::create_dir_all(directory).map_err(|err| {
                format!("Creating cache directory {}: {err}", directory.display())
            })?;
            cache_fingerprint(frontend, runtime, resolution).map(|hash| (directory, hash))
        })
        .transpose()?;
    let mut samples = Vec::with_capacity(paths.len());
    for path in paths {
        let caption_path = path.with_extension("txt");
        let caption_text = fs::read_to_string(&caption_path).map_err(|err| {
            format!(
                "Image requires a nonempty UTF-8 caption at {} ({err})",
                caption_path.display()
            )
        })?;
        let caption = caption_text
            .strip_prefix('\u{feff}')
            .unwrap_or(&caption_text)
            .trim_matches(|ch: char| ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch));
        if caption.is_empty() {
            return Err(format!(
                "Image requires a nonempty UTF-8 caption: {}",
                caption_path.display()
            ));
        }
        // Match Python's universal newlines before tokenization.
        let caption = caption.replace("\r\n", "\n").replace('\r', "\n");
        let (cache, image_stamp) = if let Some((directory, base_hash)) = &shared_fingerprint {
            let mut hash = base_hash.clone();
            let stamp = file_stamp(&path)?;
            hash.update(stamp);
            hash.update(caption.as_bytes());
            (
                Some(directory.join(format!("{:x}.safetensors", hash.finalize()))),
                Some(stamp),
            )
        } else {
            (None, None)
        };
        samples.push(Sample {
            path,
            caption,
            cache,
            image_stamp,
        });
    }
    Ok(samples)
}

fn load_tokenizer(path: &Path, pad_token: &str) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|err| format!("Loading tokenizer {}: {err}", path.display()))?;
    let pad_id = tokenizer.token_to_id(pad_token).ok_or_else(|| {
        format!(
            "Tokenizer {} is missing the pad token {pad_token}",
            path.display()
        )
    })?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: TOKEN_COUNT,
            ..Default::default()
        }))
        .map_err(|err| format!("Configuring tokenizer {}: {err}", path.display()))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(TOKEN_COUNT),
        pad_id,
        pad_token: pad_token.to_string(),
        ..Default::default()
    }));
    Ok(tokenizer)
}

impl CaptionTokenizers {
    pub fn load(frontend: &Path) -> Result<Self> {
        let config = frontend.join("qwen_tokenizer_config.json");
        let data =
            fs::read(&config).map_err(|err| format!("Reading {}: {err}", config.display()))?;
        let config: serde_json::Value =
            serde_json::from_slice(&data).map_err(|err| err.to_string())?;
        let pad = ["pad_token", "eos_token"]
            .into_iter()
            .find_map(|name| {
                let value = config.get(name)?;
                value
                    .as_str()
                    .or_else(|| value.get("content").and_then(|x| x.as_str()))
            })
            .ok_or("Qwen tokenizer configuration has no pad or EOS token")?;
        Ok(Self {
            qwen: load_tokenizer(&frontend.join("qwen_tokenizer.json"), pad)?,
            t5: load_tokenizer(&frontend.join("t5_tokenizer.json"), "<pad>")?,
        })
    }

    pub fn encode(&self, sample: &Sample) -> Result<PromptTokens> {
        let convert = |tokenizer: &Tokenizer| -> Result<(Vec<i64>, Vec<i64>)> {
            let encoding = tokenizer
                .encode(sample.caption.as_str(), true)
                .map_err(|err| {
                    format!("Tokenizing caption for {}: {err}", sample.path.display())
                })?;
            if encoding.len() != TOKEN_COUNT {
                return Err(format!(
                    "Expected {TOKEN_COUNT} padded tokens for {}",
                    sample.path.display()
                ));
            }
            Ok((
                encoding.get_ids().iter().map(|id| i64::from(*id)).collect(),
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|x| i64::from(*x))
                    .collect(),
            ))
        };
        let (qwen_ids, qwen_mask) = convert(&self.qwen)?;
        let (t5_ids, t5_mask) = convert(&self.t5)?;
        Ok(PromptTokens {
            qwen_ids,
            qwen_mask,
            t5_ids,
            t5_mask,
        })
    }
}

fn bucket_size(width: u32, height: u32, resolution: u32) -> (u32, u32) {
    let aspect_log = (f64::from(width) / f64::from(height)).ln();
    let aspect = (-4..=4)
        .map(|index| 2_f64.powf(f64::from(index) / 4.0))
        .min_by(|left, right| {
            (left.ln() - aspect_log)
                .abs()
                .total_cmp(&(right.ln() - aspect_log).abs())
        })
        .expect("aspect ratio buckets");
    let scale = aspect.sqrt();
    let align = |side: f64| (side / 16.0).round_ties_even() as u32 * 16;
    (
        align(f64::from(resolution) * scale),
        align(f64::from(resolution) / scale),
    )
}

fn bicubic(distance: f64) -> f64 {
    let x = distance.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn coefficients(source: u32, dest: u32, start: f64, end: f64) -> Vec<(usize, Vec<i64>)> {
    // Pillow passes the crop box to its resampler in single precision.
    let crop_width = (end as f32) - (start as f32);
    let start = f64::from(start as f32);
    let scale = f64::from(crop_width) / f64::from(dest);
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let inverse_scale = 1.0 / filter_scale;
    (0..dest)
        .map(|pixel| {
            let center = start + (f64::from(pixel) + 0.5) * scale;
            let first = ((center - support + 0.5) as i64).max(0) as usize;
            let last = ((center + support + 0.5) as i64).min(i64::from(source)) as usize;
            let weights: Vec<_> = (first..last)
                .map(|index| bicubic((index as f64 - center + 0.5) * inverse_scale))
                .collect();
            let total: f64 = weights.iter().sum();
            let fixed = weights
                .into_iter()
                .map(|weight| (weight / total * COEFFICIENT_SCALE as f64).round() as i64)
                .collect();
            (first, fixed)
        })
        .collect()
}

fn rounded_channel(accumulator: i64) -> u8 {
    ((accumulator + COEFFICIENT_SCALE / 2) / COEFFICIENT_SCALE).clamp(0, 255) as u8
}

fn resize_rows(rgb: &[u8], row_size: usize, weights: &[(usize, Vec<i64>)]) -> Vec<u8> {
    let mut output = Vec::with_capacity(weights.len() * 3 * (rgb.len() / row_size));
    for row in rgb.chunks_exact(row_size) {
        for (offset, weights) in weights {
            let mut channels = [0i64; 3];
            for (index, weight) in weights.iter().enumerate() {
                let start = (offset + index) * 3;
                for channel in 0..3 {
                    channels[channel] += i64::from(row[start + channel]) * weight;
                }
            }
            output.extend(channels.map(rounded_channel));
        }
    }
    output
}

fn resize_columns(rgb: &[u8], row_size: usize, weights: &[(usize, Vec<i64>)]) -> Vec<u8> {
    let mut output = Vec::with_capacity(row_size * weights.len());
    for (offset, weights) in weights {
        for x in 0..row_size {
            output.push(rounded_channel(
                weights
                    .iter()
                    .enumerate()
                    .map(|(index, weight)| i64::from(rgb[(offset + index) * row_size + x]) * weight)
                    .sum(),
            ));
        }
    }
    output
}

fn resize_rgb(rgb: Vec<u8>, width: u32, height: u32, out_width: u32, out_height: u32) -> Vec<u8> {
    if (width, height) == (out_width, out_height) {
        return rgb;
    }
    let source_aspect = f64::from(width) / f64::from(height);
    let target_aspect = f64::from(out_width) / f64::from(out_height);
    let (crop_width, crop_height) = if source_aspect == target_aspect {
        (f64::from(width), f64::from(height))
    } else if source_aspect > target_aspect {
        (f64::from(height) * target_aspect, f64::from(height))
    } else {
        (f64::from(width), f64::from(width) / target_aspect)
    };
    let left = (f64::from(width) - crop_width) / 2.0;
    let top = (f64::from(height) - crop_height) / 2.0;
    let x_weights = coefficients(width, out_width, left, left + crop_width);
    let y_weights = coefficients(height, out_height, top, top + crop_height);
    let (width_reduction, height_reduction) = (
        i64::from(width) - i64::from(out_width),
        i64::from(height) - i64::from(out_height),
    );
    // Pillow changes the axis order for large vertical reductions. Pixels are
    // rounded after each pass, so the order affects the result by up to a byte.
    if height_reduction > 0 && height_reduction > 2 * width_reduction {
        let vertical = resize_columns(&rgb, width as usize * 3, &y_weights);
        resize_rows(&vertical, width as usize * 3, &x_weights)
    } else {
        let horizontal = resize_rows(&rgb, width as usize * 3, &x_weights);
        resize_columns(&horizontal, out_width as usize * 3, &y_weights)
    }
}

pub fn read_pixels(path: &Path, resolution: u32) -> Result<Pixels> {
    let fail = |err: &dyn std::fmt::Display| format!("Reading image {}: {err}", path.display());
    let image = ImageReader::open(path)
        .map_err(|err| fail(&err))?
        .with_guessed_format()
        .map_err(|err| fail(&err))?
        .decode()
        .map_err(|err| fail(&err))?;
    let (width, height) = (image.width(), image.height());
    let (out_width, out_height) = bucket_size(width, height, resolution);
    let rgb = match image {
        DynamicImage::ImageRgba8(image) => image
            .pixels()
            .flat_map(|pixel| {
                let [r, g, b, a] = pixel.0;
                let a = u16::from(a);
                [r, g, b].map(|value| ((u16::from(value) * a + 255 * (255 - a) + 127) / 255) as u8)
            })
            .collect(),
        DynamicImage::ImageLuma16(image) => image
            .pixels()
            .flat_map(|pixel| [pixel.0[0].min(255) as u8; 3])
            .collect(),
        image => image.into_rgb8().into_raw(),
    };
    Ok(Pixels {
        rgb: resize_rgb(rgb, width, height, out_width, out_height),
        width: out_width,
        height: out_height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_resize_matches_pillow_center_crop() {
        for (width, height, out_width, out_height, sha) in [
            (
                13,
                7,
                9,
                5,
                "f1f1ff847b36706c2b0792ed2e932dd1f18db57c9d261518cd5c3a9afecdbad2",
            ),
            (
                7,
                13,
                10,
                16,
                "51247f94a17f7688ad01d865ffc0380bdfe0f077682229d1e5f19a19ee0b5a28",
            ),
            (
                7,
                10,
                10,
                10,
                "3f73c795ae00bfbe133b103aa6d23e4c131276cadda706c96f0cb0b644b4342a",
            ),
            (
                701,
                329,
                720,
                368,
                "6b52254ad6f5b5b91cb4615d22fa6abde8a967f2f728804bf182caaf8b4074c6",
            ),
        ] {
            let rgb = (0..u64::from(width * height * 3))
                .map(|i| ((i * 179 + 13 * i * i) % 251) as u8)
                .collect();
            let actual = resize_rgb(rgb, width, height, out_width, out_height);
            assert_eq!(format!("{:x}", Sha256::digest(actual)), sha);
        }
    }
}
