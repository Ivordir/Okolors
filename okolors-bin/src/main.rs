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
    clippy::rest_pat_in_fully_bound_structs,
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

use std::{
    fmt::{self, Display},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use image::{DynamicImage, GenericImageView, RgbImage};
use palette::{FromColor, Okhsl, Oklab, Srgb};
use quantette::{
    kmeans::{self, Centroids},
    wu::{self, FloatBinner},
    ColorSpace, UniqueColorCounts,
};

/// Record the running time of a function and print the elapsed time
macro_rules! time {
    ($name: literal, $verbose: expr, $func_call: expr) => {{
        let start = Instant::now();
        let result = $func_call;
        if $verbose {
            println!("{} took {}ms", $name, start.elapsed().as_millis());
        }
        result
    }};
}

/// Error cases for loading and decoding an image
#[derive(Debug)]
enum ImageLoadError {
    /// Failed to read or decode the image file
    ImageLoad(image::ImageError),
    /// Failed to read the avif file
    #[cfg(feature = "avif")]
    AvifRead(std::io::Error),
    /// Failed to decode the avif file
    #[cfg(feature = "avif")]
    AvifDecode(libavif_image::Error),
}

impl Display for ImageLoadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ImageLoadError::ImageLoad(e) => write!(f, "Failed to load the image file: {e}"),
            #[cfg(feature = "avif")]
            ImageLoadError::AvifRead(e) => write!(f, "Failed to read the avif file: {e}"),
            #[cfg(feature = "avif")]
            ImageLoadError::AvifDecode(e) => write!(f, "Failed to decode the avif file: {e}"),
        }
    }
}

fn main() -> ExitCode {
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

/// Builds a thread pool and then runs `get_print_palette`
#[cfg(feature = "threads")]
fn run_generate_and_print_palette(options: &Options) -> Result<(), ImageLoadError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(usize::from(options.threads))
        .build()
        .expect("initialized thread pool");

    pool.install(|| generate_and_print_palette(options))
}

/// Runs `get_print_palette` on a single thread
#[cfg(not(feature = "threads"))]
fn run_generate_and_print_palette(options: &Options) -> Result<(), ImageLoadError> {
    generate_and_print_palette(options)
}

/// Load an image, generate its palette, and print the result using the given options
fn generate_and_print_palette(options: &Options) -> Result<(), ImageLoadError> {
    // Input
    let img = time!("Image loading", options.verbose, load_image(&options.image))?;
    let img = generate_thumbnail(img, options.max_pixels, options.verbose);
    let img = img.into_rgb8();

    // Processing
    let (palette, counts) = {
        let start = Instant::now();
        let result = get_palette_counts(&img, options);
        if options.verbose {
            println!(
                "Palette generation took {}ms in total",
                start.elapsed().as_millis()
            );
        }
        result
    };

    // Output
    let mut colors = sorted_colors(&palette, &counts, options);
    print_palette(&mut colors, options);

    Ok(())
}

/// Load the image at the given path
#[cfg(feature = "avif")]
fn load_image(path: &PathBuf) -> Result<DynamicImage, ImageLoadError> {
    if path.extension().map_or(false, |ext| ext == "avif") {
        let buf = std::fs::read(path).map_err(ImageLoadError::AvifRead)?;
        libavif_image::read(&buf).map_err(ImageLoadError::AvifDecode)
    } else {
        image::open(path).map_err(ImageLoadError::ImageLoad)
    }
}

/// Load the image at the given path
#[cfg(not(feature = "avif"))]
fn load_image(path: &PathBuf) -> Result<DynamicImage, ImageLoadError> {
    image::open(path).map_err(ImageLoadError::ImageLoad)
}

/// Create a thumbnail with at most `max_pixels` pixels if the image has more than `max_pixels` pixels
fn generate_thumbnail(image: DynamicImage, max_pixels: u32, verbose: bool) -> DynamicImage {
    // The number of pixels should be < u64::MAX, since image dimensions are (u32, u32)
    let (width, height) = image.dimensions();
    let pixels = u64::from(width) * u64::from(height);
    if pixels <= u64::from(max_pixels) {
        if verbose {
            println!("Skipping image thumbnail since pixels was below max pixels");
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
            println!("Creating a thumbnail with dimensions {thumb_width}x{thumb_height}");
        }

        time!(
            "Image thumbnail",
            verbose,
            image.thumbnail(thumb_width, thumb_height)
        )
    }
}

/// Create an [`Oklab`] binner for the given lightness weight.
fn binner(lightness_weight: f32) -> FloatBinner<f32, 32> {
    let mut ranges = ColorSpace::OKLAB_F32_COMPONENT_RANGES_FROM_SRGB;
    ranges[0].1 *= lightness_weight;
    FloatBinner::new(ranges)
}

/// Returns the number of samples to make based off the given sampling factor and unique colors.
fn num_samples(color_counts: &UniqueColorCounts<Oklab, f32, 3>, sampling_factor: f32) -> u32 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (f64::from(sampling_factor) * f64::from(color_counts.num_colors())) as u32
    }
}

/// Generate a palette from the given image and options
fn palette_counts(image: &RgbImage, options: &Options) -> (Vec<Oklab>, Vec<u32>) {
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
        UniqueColorCounts::try_from_rgbimage(image, |srgb| {
            let mut oklab = Oklab::from_color(srgb.into_linear());
            oklab.l *= lightness_weight;
            oklab
        })
        .expect("less than u32::MAX pixels") // because of thumbnail
    );

    if verbose {
        println!("Reduced image to {} unique colors", unique.num_colors());
    }

    let centroids = time!(
        "Initial centroids",
        verbose,
        wu::palette(&unique, k, &binner(lightness_weight))
    );

    let samples = num_samples(&unique, sampling_factor);

    let mut result = if samples == 0 {
        if verbose {
            println!("Skipping k-means since samples was 0");
        }

        centroids
    } else {
        if verbose {
            println!("Running k-means for {samples} samples");
        }

        let centroids = Centroids::from_truncated(centroids.palette);

        time!(
            "k-means",
            verbose,
            kmeans::palette(&unique, samples, centroids, seed)
        )
    };

    for color in &mut result.palette {
        color.l /= lightness_weight;
    }

    (result.palette, result.counts)
}

/// temp
#[cfg(not(feature = "threads"))]
fn get_palette_counts(image: &RgbImage, options: &Options) -> (Vec<Oklab>, Vec<u32>) {
    palette_counts(image, options)
}

/// Generate a palette from the given image and options
#[cfg(feature = "threads")]
fn get_palette_counts(image: &RgbImage, options: &Options) -> (Vec<Oklab>, Vec<u32>) {
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
        palette_counts(image, options)
    } else {
        let unique = time!(
            "Preprocessing",
            verbose,
            UniqueColorCounts::try_from_rgbimage_par(image, |srgb| {
                let mut oklab = Oklab::from_color(srgb.into_linear());
                oklab.l *= lightness_weight;
                oklab
            })
            .expect("less than u32::MAX pixels")
        );

        if verbose {
            println!("Reduced image to {} unique colors", unique.num_colors());
        }

        let centroids = time!(
            "Initial centroids",
            verbose,
            wu::palette_par(&unique, k, &binner(lightness_weight))
        );

        let samples = num_samples(&unique, sampling_factor);

        let mut result = if samples < batch_size {
            if verbose {
                println!("Skipping k-means since the number of samples was too low");
            }

            centroids
        } else {
            if verbose {
                println!("Running k-means for {samples} samples with batch size {batch_size}");
            }

            let centroids = Centroids::from_truncated(centroids.palette);

            time!(
                "k-means",
                verbose,
                kmeans::palette_par(&unique, samples, batch_size, centroids, seed)
            )
        };

        for color in &mut result.palette {
            color.l /= lightness_weight;
        }

        (result.palette, result.counts)
    }
}

/// Convert [`Oklab`] colors to [`Okhsl`], sorting by the given metric.
fn sorted_colors(palette: &[Oklab], counts: &[u32], options: &Options) -> Vec<Okhsl> {
    let mut avg_colors = palette
        .iter()
        .map(|&color| Okhsl::from_color(color))
        .zip(counts)
        .collect::<Vec<_>>();

    match options.sort {
        SortOutput::H => {
            avg_colors.sort_by(|(x, _), (y, _)| f32::total_cmp(&x.hue.into(), &y.hue.into()));
        }
        SortOutput::S => {
            avg_colors.sort_by(|(x, _), (y, _)| f32::total_cmp(&x.saturation, &y.saturation));
        }
        SortOutput::L => {
            avg_colors.sort_by(|(x, _), (y, _)| f32::total_cmp(&x.lightness, &y.lightness));
        }
        SortOutput::N => avg_colors.sort_by_key(|&(_, count)| std::cmp::Reverse(count)),
    }

    if options.reverse {
        avg_colors.reverse();
    }

    avg_colors.into_iter().map(|(color, _)| color).collect()
}

/// Print the given colors based off the provided options
fn print_palette(colors: &mut [Okhsl], options: &Options) {
    match options.output {
        FormatOutput::Hex => color_format_print(colors, options, " ", |color| format!("{color:X}")),

        FormatOutput::Rgb => color_format_print(colors, options, " ", |color| {
            format!("({},{},{})", color.red, color.green, color.blue)
        }),

        FormatOutput::Swatch => format_print(colors, options, "", |color| {
            "   "
                .on_truecolor(color.red, color.green, color.blue)
                .to_string()
        }),
    }
}

/// Print a line of colors using the given format
fn print_colors(colors: &[Okhsl], delimiter: &str, format: impl Fn(Srgb<u8>) -> String) {
    println!(
        "{}",
        colors
            .iter()
            .map(|&color| format(Srgb::from_color(color).into_format::<u8>()))
            .collect::<Vec<_>>()
            .join(delimiter)
    );
}

/// Print all colors using the given format
fn format_print(
    colors: &mut [Okhsl],
    options: &Options,
    delimiter: &str,
    format: impl Fn(Srgb<u8>) -> String,
) {
    if !options.no_avg_lightness {
        print_colors(colors, delimiter, &format);
    }
    for &l in &options.lightness_levels {
        for color in &mut *colors {
            color.lightness = l / LIGHTNESS_SCALE;
        }
        print_colors(colors, delimiter, &format);
    }
}

/// Format, colorize, and then print the text for all colors
fn color_format_print(
    colors: &mut [Okhsl],
    options: &Options,
    delimiter: &str,
    format: impl Fn(Srgb<u8>) -> String,
) {
    match options.colorize {
        Some(ColorizeOutput::Fg) => format_print(colors, options, delimiter, |color| {
            format(color)
                .truecolor(color.red, color.green, color.blue)
                .to_string()
        }),

        Some(ColorizeOutput::Bg) => format_print(colors, options, delimiter, |color| {
            format(color)
                .on_truecolor(color.red, color.green, color.blue)
                .to_string()
        }),

        None => format_print(colors, options, delimiter, format),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn load_img(image: &str) -> DynamicImage {
        load_image(&PathBuf::from(image)).unwrap()
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
    #[cfg(any(feature = "jpeg", feature = "threads"))]
    fn load_jpeg() {
        test_format("jpg");
    }

    #[test]
    #[cfg(feature = "png")]
    fn load_png() {
        test_format("png");
    }

    #[test]
    #[cfg(feature = "gif")]
    fn load_gif() {
        let _img = load_img("../img/formats/img/kmeans.gif");
    }

    #[test]
    #[cfg(feature = "qoi")]
    fn load_qoi() {
        test_format("qoi");
    }

    #[test]
    #[cfg(feature = "webp")]
    fn load_webp() {
        test_format("webp");
    }

    #[test]
    #[cfg(feature = "avif")]
    fn load_avif() {
        test_format("avif");
    }

    #[test]
    #[cfg(feature = "bmp")]
    fn load_bmp() {
        test_format("bmp");
    }

    #[test]
    #[cfg(feature = "tiff")]
    fn load_tiff() {
        test_format("tiff");
    }
}
