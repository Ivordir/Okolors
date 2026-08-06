#![allow(clippy::print_stdout, clippy::print_stderr, reason = "CLI program")]

mod cli;

use clap::Parser as _;
use cli::{Colorize, Format, LIGHTNESS_SCALE, Options, Sort};
use colored::{ColoredString, Colorize as _};
use image::{
    ImageError, RgbImage,
    error::{LimitError, LimitErrorKind},
};
use okolors::{KmeansOptions, Okolors};
use palette::{FromColor as _, Okhsl, Oklab, Oklch, Srgb};
use std::{
    io::{self, StdoutLock, Write as _},
    process::ExitCode,
    time::Instant,
};

macro_rules! time {
    ($name: literal, $verbose: expr, $func_call: expr) => {{
        let start = Instant::now();
        let result = $func_call;
        if $verbose {
            eprintln!("{} took {}ms", $name, start.elapsed().as_millis());
        }
        result
    }};
}

fn main() -> ExitCode {
    #[cfg(unix)]
    {
        use nix::sys::signal::{SaFlags, SigAction, SigHandler, SigSet, Signal, sigaction};

        // Currently, the Rust runtime unconditionally sets the SIGPIPE handler to ignore.
        // This means writes to a broken stdout pipe can return an `Err` instead of exiting the process.
        // Since we bubble up io errors, this will cause:
        //   1. an error message to be printed
        //   2. the shell will see that the process exited instead of being terminated by a signal
        //
        // The first issue can be solved by manually checking if the io error is due to a broken pipe.
        // However, the second issue is more annoying. We would have to first set the SIGPIPE handler
        // to something else (e.g., the default) and only then re-raise the SIGPIPE signal.
        // Instead, let's just "restore" the signal handler to the default action from the start,
        // so that the desired behavior happens automatically.
        //
        // Note that this is still not a fully robust solution, since the SIGPIPE handler
        // inherited from the parent process is dropped in favor of the system default handler.
        // This could be solved if the `unix_sigpipe` attribute were to somehow land.
        // See: https://github.com/rust-lang/rust/issues/97889

        let default = SigAction::new(SigHandler::SigDfl, SaFlags::empty(), SigSet::empty());
        #[expect(unsafe_code, clippy::expect_used, reason = "loud crash is desired")]
        // Safety: setting default handler on valid signal
        unsafe { sigaction(Signal::SIGPIPE, &default) }.expect("set default SIGPIPE handler");
    }

    let options = Options::parse();

    #[cfg(feature = "threads")]
    let result = {
        #[expect(clippy::expect_used, reason = "loud crash is desired")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(usize::from(options.threads))
            .build()
            .expect("initialized thread pool");

        pool.install(|| generate_and_print_palette(&options))
    };

    #[cfg(not(feature = "threads"))]
    let result = generate_and_print_palette(&options);

    // Returning Result<_> uses Debug printing instead of Display
    if let Err(e) = result {
        eprintln!("{e}");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Load an image, generate its palette, and print the result using the given options
fn generate_and_print_palette(options: &Options) -> Result<(), ImageError> {
    // Input
    let img = time!(
        "Image loading",
        options.verbose,
        image::open(&options.image)
    )?;
    let img = img.into_rgb8();

    // Processing
    let result = {
        if options.verbose {
            eprintln!("Starting palette generation...");
        }
        time!(
            "Palette generation",
            options.verbose,
            generate_palette(&img, options)?
        )
    };

    // Output
    let mut colors = sorted_colors(result, options);
    print_palette(&mut colors, options)?;

    Ok(())
}

/// Generate a palette from the given image and options
fn generate_palette(image: &RgbImage, options: &Options) -> Result<Vec<Oklab>, ImageError> {
    let Options {
        lightness_weight,
        k,
        sampling_factor,
        max_samples,
        seed,
        #[cfg(feature = "threads")]
        batch_size,
        #[cfg(feature = "threads")]
        threads,
        ..
    } = *options;

    let kmeans_options = KmeansOptions::new()
        .sampling_factor(sampling_factor)
        .max_samples(max_samples)
        .seed(seed);

    #[cfg(feature = "threads")]
    let kmeans_options = kmeans_options.batch_size(batch_size);

    let okolors = Okolors::try_from(image)
        .map_err(|_| ImageError::Limits(LimitError::from_kind(LimitErrorKind::DimensionError)))?
        .lightness_weight(lightness_weight)
        .palette_size(k)
        .kmeans_options(kmeans_options)
        .sort_by_frequency(true);

    #[cfg(feature = "threads")]
    let okolors = okolors.parallel(threads != 1);

    Ok(okolors.oklab_palette())
}

/// Convert [`Oklab`] colors to [`Okhsl`], sorting by the given metric.
fn sorted_colors(palette: Vec<Oklab>, options: &Options) -> Vec<Okhsl> {
    fn sort_by_component(palette: &mut [Okhsl], component: impl Fn(&Okhsl) -> f32) {
        palette.sort_by(|x, y| f32::total_cmp(&component(x), &component(y)));
    }

    let Options { sort, reverse, .. } = *options;

    let mut palette = palette
        .into_iter()
        .map(Okhsl::from_color)
        .collect::<Vec<_>>();

    match sort {
        Sort::H => sort_by_component(&mut palette, |c| c.hue.into()),
        Sort::S => sort_by_component(&mut palette, |c| c.saturation),
        Sort::L => sort_by_component(&mut palette, |c| c.lightness),
        Sort::N => (),
    }

    let reverse = if sort == Sort::N { !reverse } else { reverse };
    if reverse {
        palette.reverse();
    }

    palette
}

/// Print the given colors based off the provided options
fn print_palette(colors: &mut [Okhsl], options: &Options) -> io::Result<()> {
    let (colorize, delimiter) = if matches!(options.format, Format::Swatch) {
        (Some(Colorize::Bg), "")
    } else {
        (options.colorize, " ")
    };

    let colorize = match colorize {
        Some(Colorize::Fg) => ColoredString::truecolor,
        Some(Colorize::Bg) => ColoredString::on_truecolor,
        None => |s, _, _, _| s,
    };

    let format: fn(Okhsl, Srgb<u8>) -> _ = match options.format {
        Format::Hex => |_, color| format!("{color:X}"),
        Format::Rgb => |_, color| format!("({},{},{})", color.red, color.green, color.blue),
        Format::Oklab => |color, _| {
            let color = Oklab::from_color(color);
            format!("({},{},{})", color.l, color.a, color.b)
        },
        Format::Oklch => |color, _| {
            let color = Oklch::from_color(color);
            format!(
                "({},{},{})",
                color.l,
                color.chroma,
                color.hue.into_positive_degrees(),
            )
        },
        Format::Okhsl => |color, _| {
            format!(
                "({},{},{})",
                color.hue.into_positive_degrees(),
                color.saturation,
                color.lightness,
            )
        },
        Format::Swatch => |_, _| "   ".into(),
    };

    let stdout = &mut io::stdout().lock();
    if !options.no_avg_lightness {
        print_colors_line(stdout, colors, delimiter, format, colorize)?;
    }
    for &l in &options.lightness_levels {
        for color in &mut *colors {
            color.lightness = l / LIGHTNESS_SCALE;
        }
        print_colors_line(stdout, colors, delimiter, format, colorize)?;
    }
    Ok(())
}

/// Format and colorize the given colors, printing them as a line of text output
fn print_colors_line(
    stdout: &mut StdoutLock<'_>,
    colors: &[Okhsl],
    delimiter: &str,
    format: fn(Okhsl, Srgb<u8>) -> String,
    colorize: fn(ColoredString, u8, u8, u8) -> ColoredString,
) -> io::Result<()> {
    let str = colors
        .iter()
        .map(|&okhsl| {
            let rgb = Srgb::from_color(okhsl).into_format();
            let text = format(okhsl, rgb).into();
            colorize(text, rgb.red, rgb.green, rgb.blue).to_string()
        })
        .collect::<Vec<_>>()
        .join(delimiter);

    writeln!(stdout, "{str}")
}

#[cfg(test)]
mod tests {}
