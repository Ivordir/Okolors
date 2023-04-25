//! Generates a color palette for an image by performing k-means clustering in the Oklab color space.
//! Also supports outputting the resulting colors in multiple Okhsl lightness levels.

#![deny(unsafe_code)]
#![warn(clippy::cargo, clippy::nursery, clippy::pedantic)]
#![warn(clippy::use_debug, clippy::dbg_macro, clippy::todo, clippy::unimplemented)]
#![warn(clippy::unwrap_used, clippy::unwrap_in_result)]
#![warn(clippy::unneeded_field_pattern, clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::str_to_string, clippy::string_to_string, clippy::string_slice)]
#![warn(missing_docs, clippy::missing_docs_in_private_items, rustdoc::all)]
#![warn(clippy::float_cmp_const, clippy::lossy_float_literal)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::unreadable_literal)]
// Clippy issue? This lint triggers for `derive(StructOfArray)` macro
#![allow(clippy::range_plus_one)]

use clap::Parser;
use colored::Colorize;
use image::{ImageBuffer, Rgb};
use palette::{FromColor, Okhsl, Oklab, Srgb};
use soa_derive::StructOfArray;
use std::{collections::HashMap, path::PathBuf};

mod cli;
use cli::{ColorizeOutput, FormatOutput, Options, SortOutput, LIGHTNESS_SCALE};

mod kmeans;
use kmeans::KmeansResult;

/// Processed image pixel data
#[derive(StructOfArray)]
pub struct PixelData {
	/// The Oklab color for this data point
	pub color: Oklab,
	/// The number of duplicate Srgb pixels
	pub count: u32,
}

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

/// Process Srgb pixels to Oklab colors and `PixelDataVec`, scaling pixel counts by the log base, if provided
fn process_pixels(pixels: &[Srgb<u8>]) -> PixelDataVec {
	let mut data = PixelDataVec::new();

	// Converting from Srgb to Oklab is expensive.
	// Memoizing the results almost halves the time needed.
	// This also groups identical pixels, speeding up k-means.

	// Packed Srgb -> data index
	let mut memo: HashMap<u32, u32> = HashMap::new();

	// Convert to Oklab color, merging entries as necessary
	for srgb in pixels {
		let key = srgb.into_u32::<palette::rgb::channels::Rgba>();
		let index = *memo.entry(key).or_insert_with(|| {
			let color = Oklab::from_color(srgb.into_format());

			// pixels.len() < u32::MAX because of `get_thumbnail`
			// Also, there are only (2^8)^3 < u32::MAX possible sRGB colors and we are grouping the same colors together
			#[allow(clippy::cast_possible_truncation)]
			let index = data.len() as u32;

			data.push(PixelData { color, count: 0 });
			index
		});

		data.count[index as usize] += 1;
	}

	data
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

	assert!(img.pixels().len() > 0, "The provided image is empty!");

	let img = time!(thumbnail, get_thumbnail(img, options.max_pixels));

	let data = time!(
		preprocessing,
		process_pixels(palette::cast::from_component_slice(img.as_raw()))
	);

	let kmeans = time!(
		kmeans,
		kmeans::run_trials(
			&data,
			options.trials,
			options.k,
			options.max_iter,
			options.convergence_threshold,
			options.ignore_lightness,
			options.seed,
		)
	);

	if options.verbose {
		println!(
			"k-means took {} iterations with a final variance of {}",
			kmeans.iterations, kmeans.variance
		);
	}

	let colors_by_lightness = sorted_colors_grouped_by_lightness(&kmeans, options);

	match options.output {
		FormatOutput::Hex => color_format_print(&colors_by_lightness, options.colorize, " ", |color| {
			format!("#{color:X}")
		}),

		FormatOutput::Rgb => color_format_print(&colors_by_lightness, options.colorize, " ", |color| {
			format!("({},{},{})", color.red, color.green, color.blue)
		}),

		FormatOutput::Palette => format_print(&colors_by_lightness, "", |color| {
			"  ".on_truecolor(color.red, color.green, color.blue).to_string()
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
/// Then, create rows for each lightness and convert into Srgb.
fn sorted_colors_grouped_by_lightness(kmeans: &KmeansResult, options: &Options) -> Vec<Vec<Srgb<u8>>> {
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

	let mut colors_by_lightness = Vec::new();

	if !options.no_avg_lightness {
		colors_by_lightness.push(vec_map(&avg_colors, |&(color, _)| to_srgb(color)));
	}

	for &l in &options.lightness_levels {
		colors_by_lightness.push(vec_map(&avg_colors, |&(color, _)| {
			to_srgb(Okhsl { lightness: l / LIGHTNESS_SCALE, ..color })
		}));
	}

	colors_by_lightness
}

/// Print all colors using the given format
fn format_print(colors_by_lightness: &Vec<Vec<Srgb<u8>>>, delimiter: &str, format: impl Fn(Srgb<u8>) -> String) {
	for colors in colors_by_lightness {
		println!(
			"{}",
			colors.iter().copied().map(&format).collect::<Vec<_>>().join(delimiter)
		);
	}
}

/// Format, colorize, and then print the text for all colors
fn color_format_print(
	colors_by_lightness: &Vec<Vec<Srgb<u8>>>,
	colorize: Option<ColorizeOutput>,
	delimiter: &str,
	format: impl Fn(Srgb<u8>) -> String,
) {
	match colorize {
		Some(ColorizeOutput::Fg) => format_print(colors_by_lightness, delimiter, |color| {
			format(color).truecolor(color.red, color.green, color.blue).to_string()
		}),

		Some(ColorizeOutput::Bg) => format_print(colors_by_lightness, delimiter, |color| {
			format(color)
				.on_truecolor(color.red, color.green, color.blue)
				.to_string()
		}),

		None => format_print(colors_by_lightness, delimiter, format),
	}
}
