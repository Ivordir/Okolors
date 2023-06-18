//! Generates a color palette for an image by performing k-means clustering in the Oklab color space.
//! Also supports outputting the resulting colors in multiple Okhsl lightness levels.

#![deny(unsafe_code)]
#![warn(clippy::pedantic, clippy::cargo)]
#![warn(clippy::use_debug, clippy::dbg_macro, clippy::todo, clippy::unimplemented)]
#![warn(clippy::unwrap_used, clippy::unwrap_in_result)]
#![warn(clippy::unneeded_field_pattern, clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::str_to_string, clippy::string_to_string, clippy::string_slice)]
#![warn(missing_docs, clippy::missing_docs_in_private_items, rustdoc::all)]
#![warn(clippy::float_cmp_const, clippy::lossy_float_literal)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::unreadable_literal)]

use colored::Colorize;
use image::{DynamicImage, GenericImageView};
use palette::{FromColor, Okhsl, Srgb};
use std::{fmt, path::PathBuf, process::ExitCode};

mod cli;
use cli::{ColorizeOutput, FormatOutput, Options, SortOutput, LIGHTNESS_SCALE};

/// Record the running time of a function and print the elapsed time
#[cfg(feature = "time")]
macro_rules! time {
	($name: ident, $func_call: expr) => {{
		use std::time::Instant;
		let start = Instant::now();
		let result = $func_call;
		let end = Instant::now();
		println!("{} took {}ms", stringify!($name), end.duration_since(start).as_millis());
		result
	}};
}

/// No-op when the time feature is disabled
#[cfg(not(feature = "time"))]
macro_rules! time {
	($name: ident, $func_call: expr) => {
		$func_call
	};
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

impl fmt::Display for ImageLoadError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		use ImageLoadError::*;
		match self {
			ImageLoad(e) => write!(f, "Failed to load the image file: {e}"),
			#[cfg(feature = "avif")]
			AvifRead(e) => write!(f, "Failed to read the avif file: {e}"),
			#[cfg(feature = "avif")]
			AvifDecode(e) => write!(f, "Failed to decode the avif file: {e}"),
		}
	}
}

fn main() -> ExitCode {
	use clap::Parser;

	let options = Options::parse();

	let result = run_get_print_palette(&options);

	// Returning Result<_> uses Debug printing instead of Display
	if let Err(e) = result {
		eprintln!("{e}");
		ExitCode::FAILURE
	} else {
		ExitCode::SUCCESS
	}
}

/// Builds a thread pool and then runs `get_print_palette`
#[cfg(any(feature = "okolors_threads", feature = "jpeg_rayon"))]
fn run_get_print_palette(options: &Options) -> Result<(), ImageLoadError> {
	let pool = rayon::ThreadPoolBuilder::new()
		.num_threads(usize::from(options.threads))
		.build()
		.expect("initialized thread pool");

	pool.install(|| get_print_palette(options))
}

/// Runs `get_print_palette` on a single thread
#[cfg(not(any(feature = "okolors_threads", feature = "jpeg_rayon")))]
fn run_get_print_palette(options: &Options) -> Result<(), ImageLoadError> {
	get_print_palette(options)
}

/// Load an image, generate its palette, and print the result using the given options
fn get_print_palette(options: &Options) -> Result<(), ImageLoadError> {
	// Input
	let img = get_pixels(options)?;

	// Processing
	let mut colors = get_palette(&img, options);

	// Output
	print_palette(&mut colors, options);

	Ok(())
}

/// Load an image from disk, generating an thumbnail if needed, and converting to [`Srgb<u8>`]
fn get_pixels(options: &Options) -> Result<DynamicImage, ImageLoadError> {
	time!(loading, load_image(&options.image))
		.map(|img| time!(thumbnail, get_thumbnail(img, options.max_pixels, options.verbose)))
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
fn get_thumbnail(image: DynamicImage, max_pixels: u32, verbose: bool) -> DynamicImage {
	// The number of pixels should be < u64::MAX, since image dimensions are (u32, u32)
	let (width, height) = image.dimensions();
	let pixels = u64::from(width) * u64::from(height);
	if pixels <= u64::from(max_pixels) {
		image
	} else {
		// (u64 as f64) only gives innaccurate results for very large u64
		// I.e, only when pixels is in the order of quintillions
		#[allow(clippy::cast_precision_loss)]
		let scale = (f64::from(max_pixels) / pixels as f64).sqrt();

		// multiplying by a positive factor < 1
		#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
		let (thumb_width, thumb_height) = ((f64::from(width) * scale) as u32, (f64::from(height) * scale) as u32);

		if verbose {
			println!("Creating a thumbnail with dimensions {thumb_width}x{thumb_height}");
		}

		image.thumbnail(thumb_width, thumb_height)
	}
}

/// Generate a palette from the given image and options
fn get_palette(image: &DynamicImage, options: &Options) -> Vec<Okhsl> {
	let data = time!(
		preprocessing,
		okolors::OklabCounts::try_from_image(image, options.alpha_threshold)
			.expect("less than u32::MAX pixels")
			.with_lightness_weight(options.lightness_weight)
	);

	if options.verbose {
		println!("Reduced image to {} unique colors", data.num_colors());
	}

	let kmeans = time!(
		kmeans,
		okolors::run(
			&data,
			options.trials,
			options.k,
			options.convergence_threshold,
			options.max_iter,
			options.seed,
		)
	);

	if options.verbose {
		println!(
			"k-means took {} iterations with a final MSE of {}",
			kmeans.iterations, kmeans.mse
		);
	}

	sorted_colors(&kmeans, options)
}

/// Convert [`Oklab`] colors from k-means to [`Okhsl`], sorting by the given metric.
fn sorted_colors(kmeans: &okolors::KmeansResult, options: &Options) -> Vec<Okhsl> {
	let mut avg_colors = kmeans
		.centroids
		.iter()
		.map(|&color| Okhsl::from_color(color))
		.zip(&kmeans.counts)
		.collect::<Vec<_>>();

	match options.sort {
		SortOutput::H => avg_colors.sort_by(|(x, _), (y, _)| f32::total_cmp(&x.hue.into(), &y.hue.into())),
		SortOutput::S => avg_colors.sort_by(|(x, _), (y, _)| f32::total_cmp(&x.saturation, &y.saturation)),
		SortOutput::L => avg_colors.sort_by(|(x, _), (y, _)| f32::total_cmp(&x.lightness, &y.lightness)),
		SortOutput::N => avg_colors.sort_by_key(|&(_, count)| std::cmp::Reverse(count)),
	}

	if options.reverse {
		avg_colors.reverse();
	}

	avg_colors.into_iter().map(|(color, _)| color).collect()
}

/// Print the given colors based off the provided options
fn print_palette(colors: &mut [Okhsl], options: &Options) {
	use FormatOutput::*;
	match options.output {
		Hex => color_format_print(colors, options, " ", |color| format!("{color:X}")),

		Rgb => color_format_print(colors, options, " ", |color| {
			format!("({},{},{})", color.red, color.green, color.blue)
		}),

		Swatch => format_print(colors, options, "", |color| {
			"   ".on_truecolor(color.red, color.green, color.blue).to_string()
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
fn format_print(colors: &mut [Okhsl], options: &Options, delimiter: &str, format: impl Fn(Srgb<u8>) -> String) {
	if !options.no_avg_lightness {
		print_colors(colors, delimiter, &format);
	}
	for &l in &options.lightness_levels {
		for color in colors.iter_mut() {
			color.lightness = l / LIGHTNESS_SCALE;
		}
		print_colors(colors, delimiter, &format);
	}
}

/// Format, colorize, and then print the text for all colors
fn color_format_print(colors: &mut [Okhsl], options: &Options, delimiter: &str, format: impl Fn(Srgb<u8>) -> String) {
	use ColorizeOutput::*;
	match options.colorize {
		Some(Fg) => format_print(colors, options, delimiter, |color| {
			format(color).truecolor(color.red, color.green, color.blue).to_string()
		}),

		Some(Bg) => format_print(colors, options, delimiter, |color| {
			format(color)
				.on_truecolor(color.red, color.green, color.blue)
				.to_string()
		}),

		None => format_print(colors, options, delimiter, format),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn load_img(image: &str) -> DynamicImage {
		load_image(&PathBuf::from(image)).expect("loaded image")
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
				let thumb = get_thumbnail(img.clone(), max_pixels, false);
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
	#[cfg(any(feature = "jpeg", feature = "jpeg_rayon"))]
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
