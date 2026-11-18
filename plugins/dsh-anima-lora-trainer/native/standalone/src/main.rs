mod dataset;
mod runtime;
mod settings;

use std::process::ExitCode;

type Result<T> = std::result::Result<T, String>;

fn run() -> Result<()> {
    let Some(settings) = settings::Settings::parse()? else {
        return Ok(());
    };
    let samples = dataset::find_samples(
        &settings.folder,
        settings.recursive,
        settings.resolution,
        &settings.frontend,
        &settings.runtime,
        settings.cache_dir.as_deref(),
    )?;
    runtime::run(&settings, &samples)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Anima training failed: {error}");
            ExitCode::FAILURE
        }
    }
}
