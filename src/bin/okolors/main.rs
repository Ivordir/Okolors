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
#![allow(clippy::enum_glob_use)]
#![allow(clippy::unreadable_literal)]

use clap::Parser;
use colored::Colorize;
use image::{ImageBuffer, Rgb};
use palette::{FromColor, Okhsl, Srgb};
use std::path::PathBuf;

mod cli;
use cli::{ColorizeOutput, FormatOutput, Options, SortOutput, LIGHTNESS_SCALE};

/// Load Srgb pixels from an image path
fn load_image(path: &PathBuf) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
	// TODO: handle errors more gracefully, providing helpful messages
	if path.extension().map_or(false, |ext| ext == "avif") {
		let buf = std::fs::read(path).expect("reading avif file");
		libavif_image::read(&buf).expect("decoding avif file").into_rgb8()
	} else {
		image::open(path).expect("opening image file").into_rgb8()
	}
}

/// Create a thumbnail with `max_pixels` pixels if the image has more than `max_pixels` pixels
fn get_thumbnail(img: ImageBuffer<Rgb<u8>, Vec<u8>>, max_pixels: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
	// The number of pixels should be < u64::MAX, since image dimensions are (u32, u32)
	let pixels = img.pixels().len() as u64;
	if pixels <= u64::from(max_pixels) {
		img
	} else if max_pixels == 0 {
		ImageBuffer::new(0, 0)
	} else {
		// (u64 as f64) only gives innaccurate results for very large u64
		// I.e, only when pixels is in the order of quintillions
		#[allow(clippy::cast_precision_loss)]
		let scale = (f64::from(max_pixels) / pixels as f64).sqrt();
		let (width, height) = img.dimensions();

		// multiplying by a positive factor < 1
		#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
		image::imageops::thumbnail(
			&img,
			(f64::from(width) * scale) as u32,
			(f64::from(height) * scale) as u32,
		)
	}
}

/// Record the running time of a function and print the elapsed time
#[cfg(feature = "time")]
macro_rules! time {
	($name: ident, $func_call: expr) => {{
		use std::time::Instant;
		let start = Instant::now();
		let result = $func_call;
		let end = Instant::now();
		println!("{} took {}", stringify!($name), end.duration_since(start).as_millis());
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

fn main() {
	print_palette(&Options::parse());
}

/// Generate and print a palette using the given options
fn print_palette(options: &Options) {
	let img = time!(loading, load_image(&options.image));

	let img = time!(thumbnail, get_thumbnail(img, options.max_pixels));

	let data = time!(
		preprocessing,
		okolors::srgb_to_oklab_counts(palette::cast::from_component_slice(img.as_raw()))
	);

	let kmeans = time!(
		kmeans,
		okolors::from_oklab_counts(
			&data,
			options.trials,
			options.k,
			options.convergence_threshold,
			options.max_iter,
			options.seed,
			options.ignore_lightness,
		)
	);

	if options.verbose {
		println!(
			"k-means took {} iterations with a final variance of {}",
			kmeans.iterations, kmeans.variance
		);
	}

	let mut colors = sorted_colors(&kmeans, options);

	match options.output {
		FormatOutput::Hex => color_format_print(&mut colors, options, " ", |color| format!("{color:X}")),

		FormatOutput::Rgb => color_format_print(&mut colors, options, " ", |color| {
			format!("({},{},{})", color.red, color.green, color.blue)
		}),

		FormatOutput::Swatch => format_print(&mut colors, options, "", |color| {
			"   ".on_truecolor(color.red, color.green, color.blue).to_string()
		}),
	}
}

/// Shorthand for `vec.iter().map().collect::<Vec<_>>()`
fn vec_map<T, U>(vec: &[T], mapping: impl FnMut(&T) -> U) -> Vec<U> {
	vec.iter().map(mapping).collect::<Vec<_>>()
}

/// Convert an Okhsl color to an Srgb color with u8 components
fn to_srgb(okhsl: Okhsl) -> Srgb<u8> {
	Srgb::from_color(okhsl).into_format::<u8>()
}

/// Convert Oklab colors from k-means to Okhsl, sorting by the given metric.
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

	vec_map(&avg_colors, |&(color, _)| color)
}

/// Print a line of colors using the given format
fn print_colors(colors: &[Okhsl], delimiter: &str, format: impl Fn(Srgb<u8>) -> String) {
	println!(
		"{}",
		colors
			.iter()
			.copied()
			.map(|color| format(to_srgb(color)))
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
	match options.colorize {
		Some(ColorizeOutput::Fg) => format_print(colors, options, delimiter, |color| {
			format(color).truecolor(color.red, color.green, color.blue).to_string()
		}),

		Some(ColorizeOutput::Bg) => format_print(colors, options, delimiter, |color| {
			format(color)
				.on_truecolor(color.red, color.green, color.blue)
				.to_string()
		}),

		None => format_print(colors, options, delimiter, format),
	}
}
