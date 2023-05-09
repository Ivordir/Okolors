//! Specifies the CLI and handles arg parsing

use clap::{Parser, ValueEnum};
use palette::Okhsl;
use std::{
	fmt::{Debug, Display},
	num::ParseFloatError,
	ops::RangeBounds,
	path::PathBuf,
	str::FromStr,
};

/// Supported output formats for the final colors
#[derive(Copy, Clone, ValueEnum)]
pub enum FormatOutput {
	/// sRGB hexcode
	Hex,
	/// sRGB (r,g,b) triple
	Rgb,
	/// Whitespace with true color background
	Swatch,
}

/// Sort orders for the final colors
#[derive(Copy, Clone, ValueEnum)]
pub enum SortOutput {
	/// Ascending hue
	H,
	/// Ascending saturation
	S,
	/// Ascending lightness
	L,
	/// Descending number of pixels
	N,
}

/// Ways to colorize the output text
#[derive(Copy, Clone, ValueEnum)]
pub enum ColorizeOutput {
	/// Foreground
	Fg,
	/// Background
	Bg,
}

/// Generate a color palette for an image by performing k-means clustering in the Oklab color space.
///
/// Okolors also supports outputting the resulting colors in multiple Okhsl lightness levels.
#[allow(clippy::struct_excessive_bools)]
#[derive(Parser)]
#[command(version)]
pub struct Options {
	/// The path to the input image
	pub image: PathBuf,

	/// The format to print the colors in
	#[arg(short, long, default_value = "hex")]
	pub output: FormatOutput,

	/// Color the foreground or background for each printed color
	#[arg(short, long)]
	pub colorize: Option<ColorizeOutput>,

	/// The order to print the colors in
	///
	/// The h, s, and l options below refer to Okhsl component values and not the HSL color space.
	#[arg(short, long, default_value = "n")]
	pub sort: SortOutput,

	/// Reverse the printed order of the colors
	#[arg(short, long)]
	pub reverse: bool,

	/// A comma separated list of additional lightness levels that each color should be printed in
	///
	/// Lightness refers to Okhsl lightness with values in the range [0, 100].
	/// A separate line is used for printing the colors at each lightness level.
	#[arg(short, long, value_delimiter = ',', value_parser = parse_valid_lightness)]
	pub lightness_levels: Vec<f32>,

	/// Do not print each color with its average lightness
	///
	/// This is useful if you only care about colors resulting from the --lightness-levels option.
	#[arg(long)]
	pub no_avg_lightness: bool,

	/// The value used to scale down the influence of the lightness component on color difference
	///
	/// Lower weights have the effect of bringing out more distinct hues,
	/// but the resulting colors will technically not be accurate, perceptual averages for the colors in the image.
	/// I.e., this is a subjective option you can tweak to get different kind of color palettes.
	/// A value around 0.325 seems to provide somewhat similar results to the CIELAB color space,
	/// whereas a value of 1.0 indicates to provide results while staying true to the Oklab color space.
	/// Provided values should be in the range [0.0, 1.0].
	#[arg(short = 'w', long, default_value_t = 0.325, value_parser = parse_valid_lightness_weight)]
	pub lightness_weight: f32,

	/// The (maximum) number of colors to find
	#[arg(short, default_value_t = 8)]
	pub k: u8,

	/// The number of trials of k-means to run
	///
	/// k-means can get stuck in a local minimum, so you may want to run a few or more trials to get better results.
	/// The trial with the lowest variance is picked.
	#[arg(short = 'n', long, default_value_t = 1)]
	pub trials: u32,

	/// The threshold number used to determine k-means convergence
	///
	/// The recommended range for the convergence threshold is [0.01, 0.1].
	///
	/// A value of 0.1 is very fast (often only a few iterations are needed for regular sized images),
	/// and this should be enough to get a decent looking palette.
	///
	/// A value of 0.01 is the lowest sensible value for maximum accuracy.
	/// Convergence thresholds should probably be not too far lower than this,
	/// as any iterations after this either do not or barely effect the final sRGB colors.
	/// I.e., don't use 0.0 as the convergence threshold,
	/// as that may require many more iterations and would just be wasting time.
	///
	/// Of course, any values between 0.01 and 0.1 will be some compromise between accuracy and speed.
	#[arg(short = 'e', long, default_value_t = 0.05, value_parser = parse_valid_convergence)]
	pub convergence_threshold: f32,

	/// The maximum number of iterations for all k-means trials
	///
	/// If you have a very large image and are not using the --max-pixels options, you may have to set this option higher.
	/// You can use the --verbose option to see how many iterations the best k-means trial took.
	#[arg(short = 'i', long, default_value_t = 128)]
	pub max_iter: u32,

	/// The maximum image size, in number of pixels, before a thumbnail is created
	///
	/// Unfortunately, this option may reduce the color accuracy,
	/// as multiple pixels in the original image are interpolated to form a pixel in the thumbnail.
	/// This option is intended for reducing the time needed for large images,
	/// but it can also be used to provide fast, inaccurate results for any image.
	#[arg(short = 'p', long, default_value_t = u32::MAX)]
	pub max_pixels: u32,

	/// The seed value used for the random number generator
	#[arg(long, default_value_t = 0)]
	pub seed: u64,

	/// Print additional information, such as the number of k-means iterations
	#[arg(long)]
	pub verbose: bool,
}

/// Parse a float value and ensure it in the provided, valid range
fn parse_float_in_range<T>(s: &str, range: impl RangeBounds<T> + Debug) -> Result<T, String>
where
	T: FromStr<Err = ParseFloatError> + Display + PartialOrd,
{
	let value: T = s.parse().map_err(|e| format!("{e}"))?;
	if range.contains(&value) {
		Ok(value)
	} else {
		Err(format!("{value} is not in {range:?}"))
	}
}

/// Parse the convergence number and ensure it is >= `0.0`
fn parse_valid_convergence(s: &str) -> Result<f32, String> {
	parse_float_in_range(s, 0.0..)
}

/// The factor used to scale the lightness values for a better interface,
/// as Okhsl lightness values are in the range `0.0..=1.0`
pub const LIGHTNESS_SCALE: f32 = 100.0;

/// Parse a lumiance value and ensure it is in `0.0..=100.0`
fn parse_valid_lightness(s: &str) -> Result<f32, String> {
	let min = LIGHTNESS_SCALE * Okhsl::<f32>::min_lightness();
	let max = LIGHTNESS_SCALE * Okhsl::<f32>::max_lightness();
	parse_float_in_range(s, min..=max)
}

/// Parse the lightness weight and ensure it is in `0.0..=1.0`
fn parse_valid_lightness_weight(s: &str) -> Result<f32, String> {
	parse_float_in_range(s, 0.0..=1.0)
}
