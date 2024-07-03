//! Generate a color palette from an image by performing k-means clustering in the Oklab color space.

#![deny(unsafe_code, unsafe_op_in_unsafe_fn)]
#![warn(
    clippy::pedantic,
    clippy::cargo,
    clippy::use_debug,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    clippy::unneeded_field_pattern,
    clippy::unnecessary_self_imports,
    clippy::str_to_string,
    clippy::string_to_string,
    clippy::string_slice,
    missing_docs,
    clippy::missing_docs_in_private_items,
    rustdoc::all,
    clippy::float_cmp_const,
    clippy::lossy_float_literal
)]
#![allow(
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::unreadable_literal
)]

mod cli;

#[allow(clippy::wildcard_imports)]
use cli::*;

use clap::Parser;
use colored::{ColoredString, Colorize as _};
use image::{DynamicImage, GenericImageView, ImageError};
use okolors::{
    internal::{self, QuantizeOutput},
    ColorSlice,
};
use palette::{FromColor, Okhsl, Oklab, Srgb};
use std::{
    io::{self, StdoutLock, Write},
    process::ExitCode,
    time::Instant,
};

/// Record the running time of a function and print the elapsed time
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
        use nix::sys::signal::{sigaction, SaFlags, SigAction, SigHandler, SigSet, Signal};

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
        #[allow(unsafe_code)] // setting default handler on valid signal
        unsafe { sigaction(Signal::SIGPIPE, &default) }.expect("set default SIGPIPE handler");
    }

    let options = Options::parse();

    let result = run_generate_and_print_palette(&options);

    // Returning Result<_> uses Debug printing instead of Display
    if let Err(e) = result {
        eprintln!("{e}");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Builds a thread pool and then runs `generate_and_print_palette`
#[cfg(feature = "threads")]
fn run_generate_and_print_palette(options: &Options) -> Result<(), ImageError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(usize::from(options.threads))
        .build()
        .expect("initialized thread pool");

    pool.install(|| generate_and_print_palette(options))
}

/// Runs `generate_and_print_palette` on a single thread
#[cfg(not(feature = "threads"))]
fn run_generate_and_print_palette(options: &Options) -> Result<(), ImageError> {
    generate_and_print_palette(options)
}

/// Load an image, generate its palette, and print the result using the given options
fn generate_and_print_palette(options: &Options) -> Result<(), ImageError> {
    // Input
    let img = time!(
        "Image loading",
        options.verbose,
        image::open(&options.image)
    )?;
    let img = generate_thumbnail(img, options.max_pixels, options.verbose);
    let img = img.into_rgb8();
    let slice = ColorSlice::try_from(&img).expect("less than u32::MAX pixels"); // because of thumbnail

    // Processing
    let result = {
        if options.verbose {
            eprintln!("Starting palette generation...");
        }
        let start = Instant::now();
        let result = get_palette_counts(slice, options);
        if options.verbose {
            eprintln!(
                "Palette generation took {}ms in total",
                start.elapsed().as_millis()
            );
        }
        result
    };

    // Output
    let mut colors = sorted_colors(result, options);
    print_palette(&mut colors, options)?;

    Ok(())
}

/// Create a thumbnail with at most `max_pixels` pixels if the image has more than `max_pixels` pixels
fn generate_thumbnail(image: DynamicImage, max_pixels: u32, verbose: bool) -> DynamicImage {
    // The number of pixels should be < u64::MAX, since image dimensions are (u32, u32)
    let (width, height) = image.dimensions();
    let pixels = u64::from(width) * u64::from(height);
    if pixels <= u64::from(max_pixels) {
        if verbose {
            eprintln!("Skipping image thumbnail since pixels was below max pixels");
        }

        image
    } else {
        // (u64 as f64) only gives innaccurate results for very large u64
        // I.e, only when pixels is in the order of quintillions
        #[allow(clippy::cast_precision_loss)]
        let scale = (f64::from(max_pixels) / pixels as f64).sqrt();

        // multiplying by a positive factor < 1
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (thumb_width, thumb_height) = (
            (f64::from(width) * scale) as u32,
            (f64::from(height) * scale) as u32,
        );

        if verbose {
            eprintln!("Creating a thumbnail with dimensions {thumb_width}x{thumb_height}");
        }

        time!(
            "Image thumbnail",
            verbose,
            image.thumbnail(thumb_width, thumb_height)
        )
    }
}

/// Generate a palette from the given image and options
fn palette_counts(colors: ColorSlice<Srgb<u8>>, options: &Options) -> QuantizeOutput<Oklab> {
    let Options {
        lightness_weight,
        k,
        sampling_factor,
        seed,
        verbose,
        ..
    } = *options;

    let unique = time!(
        "Preprocessing",
        verbose,
        internal::unique_oklab_counts(colors, lightness_weight)
    );

    if verbose {
        eprintln!("Reduced image to {} unique colors", unique.num_colors());
    }

    let centroids = time!(
        "Initial centroids",
        verbose,
        internal::wu_palette(&unique, k, lightness_weight)
    );

    let samples = internal::num_samples(&unique, sampling_factor);

    let mut result = if samples == 0 {
        if verbose {
            eprintln!("Skipping k-means since samples was 0");
        }

        centroids
    } else {
        if verbose {
            eprintln!("Running k-means for {samples} samples");
        }

        time!(
            "k-means",
            verbose,
            internal::kmeans_palette(&unique, samples, centroids.palette, seed)
        )
    };

    internal::restore_lightness(&mut result.palette, lightness_weight);

    result
}

/// Generate a palette from the given image and options
#[cfg(not(feature = "threads"))]
fn get_palette_counts(colors: ColorSlice<Srgb<u8>>, options: &Options) -> QuantizeOutput<Oklab> {
    palette_counts(colors, options)
}

/// Generate a palette from the given image and options
#[cfg(feature = "threads")]
fn get_palette_counts(colors: ColorSlice<Srgb<u8>>, options: &Options) -> QuantizeOutput<Oklab> {
    let Options {
        lightness_weight,
        k,
        sampling_factor,
        seed,
        verbose,
        batch_size,
        threads,
        ..
    } = *options;

    if threads == 1 {
        palette_counts(colors, options)
    } else {
        let unique = time!(
            "Preprocessing",
            verbose,
            internal::unique_oklab_counts_par(colors, lightness_weight)
        );

        if verbose {
            eprintln!("Reduced image to {} unique colors", unique.num_colors());
        }

        let centroids = time!(
            "Initial centroids",
            verbose,
            internal::wu_palette_par(&unique, k, lightness_weight)
        );

        let samples = internal::num_samples(&unique, sampling_factor);

        let mut result = if samples < batch_size {
            if verbose {
                eprintln!("Skipping k-means since the number of samples was too low");
            }

            centroids
        } else {
            if verbose {
                eprintln!("Running k-means for {samples} samples with batch size {batch_size}");
            }

            time!(
                "k-means",
                verbose,
                internal::kmeans_palette_par(&unique, samples, batch_size, centroids.palette, seed)
            )
        };

        internal::restore_lightness(&mut result.palette, lightness_weight);

        result
    }
}

/// Convert [`Oklab`] colors to [`Okhsl`], sorting by the given metric.
fn sorted_colors(result: QuantizeOutput<Oklab>, options: &Options) -> Vec<Okhsl> {
    /// Convert `Oklab` to `Okhsl`
    fn to_okhsl(oklab: Vec<Oklab>) -> Vec<Okhsl> {
        oklab.into_iter().map(Okhsl::from_color).collect()
    }

    /// Convert `Oklab` to `Okhsl`, sorting by the given `Okhsl` component
    fn sort_by_component(
        result: QuantizeOutput<Oklab>,
        reverse: bool,
        component: impl Fn(&Okhsl) -> f32,
    ) -> Vec<Okhsl> {
        let mut colors = to_okhsl(result.palette);

        colors.sort_by(|x, y| f32::total_cmp(&component(x), &component(y)));

        if reverse {
            colors.reverse();
        }

        colors
    }

    let reverse = options.reverse;

    match options.sort {
        Sort::H => sort_by_component(result, reverse, |c| c.hue.into()),
        Sort::S => sort_by_component(result, reverse, |c| c.saturation),
        Sort::L => sort_by_component(result, reverse, |c| c.lightness),
        Sort::N => {
            let result = QuantizeOutput {
                palette: to_okhsl(result.palette),
                counts: result.counts,
                indices: result.indices,
            };

            let mut colors = internal::sort_by_frequency(result);

            if !reverse {
                colors.reverse();
            }

            colors
        }
    }
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

    let format: fn(Srgb<u8>) -> _ = match options.format {
        Format::Hex => |color| format!("{color:X}"),
        Format::Rgb => |color| format!("({},{},{})", color.red, color.green, color.blue),
        Format::Swatch => |_| "   ".into(),
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
    stdout: &mut StdoutLock,
    colors: &[Okhsl],
    delimiter: &str,
    format: fn(Srgb<u8>) -> String,
    colorize: fn(ColoredString, u8, u8, u8) -> ColoredString,
) -> io::Result<()> {
    let str = colors
        .iter()
        .map(|&color| {
            let color = Srgb::from_color(color).into_format();
            let text = format(color).into();
            colorize(text, color.red, color.green, color.blue).to_string()
        })
        .collect::<Vec<_>>()
        .join(delimiter);

    writeln!(stdout, "{str}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn load_img(image: &str) -> DynamicImage {
        image::open(image).unwrap()
    }

    #[test]
    fn thumbnail_has_at_most_max_pixels() {
        // Use scaled down image for reduced running time
        let img = load_img("../img/formats/img/Jewel Changi.jpg");
        let (img_width, img_height) = img.dimensions();

        assert!(img_width % 10 == 0 && img_height % 10 == 0);
        let (width, height) = (img_width / 10, img_height / 10);

        for dw in 0..5 {
            for dh in 0..5 {
                let width = width - dw;
                let height = height - dh;
                let max_pixels = width * height;
                let thumb = generate_thumbnail(img.clone(), max_pixels, false);
                let pixels = thumb.width() * thumb.height();

                if dw == 0 && dh == 0 {
                    assert_eq!(pixels, max_pixels);
                } else {
                    let max_d = u32::max(dw, dh);
                    let min_pixels = (width - max_d) * (height - max_d);
                    assert!(
                        min_pixels <= pixels && pixels <= max_pixels,
                        "{img_width}x{img_height} => {width}x{height}: {min_pixels} <= {pixels} <= {max_pixels}"
                    );
                }
            }
        }
    }

    fn test_format(ext: &str) {
        let _img = load_img(&format!("../img/formats/img/Jewel Changi.{ext}"));
    }

    #[test]
    fn load_jpeg() {
        test_format("jpg");
    }

    #[test]
    fn load_png() {
        test_format("png");
    }

    #[test]
    fn load_gif() {
        let _img = load_img("../img/formats/img/kmeans.gif");
    }

    #[test]
    fn load_qoi() {
        test_format("qoi");
    }

    #[test]
    fn load_webp() {
        test_format("webp");
    }
}
