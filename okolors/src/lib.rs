//! Perform k-means clustering in the Oklab color space.
//!
//! # Overview
//!
//! Okolors takes an image in the `sRGB` color space or a slice of [`Srgb`] colors and returns `k` average [`Oklab`] colors.
//!
//! See the [parameters](#parameters) section for information about and recommended values for each of [`run`]'s parameters.
//!
//! For visual examples and more information (e.g., features, performance, or the Okolors binary)
//! see the [README](https://github.com/Ivordir/Okolors#readme).
//!
//! # Examples
//!
//! ## Read an image file and get 5 average colors.
//!
//! The example below first opens an image and then processes it into a [`OklabCounts`].
//! This counts duplicate pixels and converts them to the [`Oklab`] color space.
//! The resulting [`OklabCounts`] can be reused for multiple k-means runs with different arguments.
//! In this example, only one k-means run is done with k = 5.
//!
//! ```no_run
//! # fn main() -> Result<(), image::ImageError> {
//! let img = image::open("some image")?;
//! // For large images, you can create a thumbnail here to reduce the running time
//! // use image::GenericImageView;
//! // let img = img.thumbnail(width, height);
//!
//! let oklab = okolors::OklabCounts::from_image(&img, u8::MAX);
//! let result = okolors::run(&oklab, 1, 5, 0.05, 64, 0);
//! #
//! # Ok(())
//! # }
//! ```
//!
//! ## Run with different lightness weights.
//!
//! This example reuses an [`OklabCounts`], changing its `lightness_weight` to get different [`KmeansResult`]s.
//!
//! ```no_run
//! # fn main() -> Result<(), image::ImageError> {
//! let img = image::open("some image")?;
//!
//! let mut oklab = okolors::OklabCounts::from_image(&img, u8::MAX)
//!     .with_lightness_weight(0.325);
//!
//! let resultA = okolors::run(&oklab, 1, 5, 0.05, 64, 0);
//!
//! oklab.set_lightness_weight(0.01);
//! let resultB = okolors::run(&oklab, 1, 5, 0.05, 64, 0);
//! #
//! # Ok(())
//! # }
//! ```
//!
//! # Parameters
//!
//! Here are explanations for the various parameters of [`run`] and more.
//!
//! In short, the `trials`, `convergence_threshold`, and `max_iter` parameters
//! control the color accuracy at the expense of running time.
//! `k` indicates the number of colors to return,
//! and `lightness_weight` is a subjective parameter that affects how the colors are clustered.
//!
//! The [README](https://github.com/Ivordir/Okolors#readme) contains some visual examples of the effects of the parameters below.
//!
//! Note that if `trials = 0`, `k = 0`, or an empty slice of Srgb colors is provided,
//! then the returned [`KmeansResult`] will have no centroids and a variance of `0.0`.
//!
//! ## Lightness Weight
//!
//! This is used to scale down the lightness component of the [`Oklab`] colors when performing color difference.
//!
//! A value around `0.325` seems to provide similar results to the CIELAB color space.
//!
//! Lightness weights should be in the range `0.0..=1.0`.
//! A value of `1.0` indicates no scaling and performs color difference in the [`Oklab`] color space using standard euclidean distance.
//! Otherwise, lower weights have the effect of merging similar colors together, possibly bringing out more distinct hues.
//! Note that for weights near `0.0`, if the image contains black and white, then they will be averaged into a shade of gray.
//! Also, the lightness weight affects the final variance, so it does not make sense to compare two [`KmeansResult`]s using their variance
//! if the results came from different lightness weights.
//!
//! ## Trials
//!
//! This is the number of times to run k-means, taking the trial with the lowest variance.
//!
//! 1 to 3 or 1 to 5 trials is recommended, depending on how much you value accuracy.
//!
//! k-means is an approximation algorithm that can get stuck in a local minimum.
//! So, there is no guarantee that a single run may give a "good enough" result.
//! Doing multiple runs increases your chance of getting a more optimal result.
//! However, k-means is also a somewhat expensive operation,
//! so you probably do not want to set the number of runs too high.
//!
//! ## K
//!
//! This is the (maximum) number of average colors to find.
//!
//! 4 to 16 is likely the range you want for a palette.
//!
//! The ideal number of clusters is hard to know in advance, if there even is an "ideal" number.
//! Lower k values give faster runtime but also typically lower color accuracy.
//! Higher k values provide higher accuracy, but can potentially give more colors than you need/want.
//! The [`KmeansResult`] will provide the number of pixels in each centroid,
//! so this can be used to filter out centroids that make less than a certain percentage of the image.
//!
//! ## Convergence Threshold
//!
//! After a certain point, the centroids change so little that there is no longer a visual, perceivable difference.
//! This is the cutoff value used to determine whether any change is significant.
//!
//! 0.01 to 0.1 is the recommended range, with lower values indicating higher accuracy.
//!
//! A value of `0.1` is very fast (often only a few iterations are needed),
//! and this should be enough to get a decent looking palette.
//!
//! A value of `0.01` is the lowest sensible value for maximum accuracy.
//! Convergence thresholds should probably be not too far lower than this,
//! as any iterations after this either do not or barely effect the final colors once converted to [`Srgb`].
//! I.e., don't use `0.0` as the convergence threshold,
//! as that may require many more iterations and would just be wasting time.
//!
//! ## Max Iterations
//!
//! This is the maximum number of iterations allowed for each k-means trial.
//!
//! Use 16, 64, 256, or more iterations depending on k, the image size, and the convergence threshold.
//!
//! Ideally, k-means should stop due to the convergence threshold,
//! so you want to choose a high enough maximum iterations that will prevent k-means from stopping early.
//! But, the number of iterations to reach convergence might be high in some cases,
//! and the maximum iterations is there to prevent k-means from taking too long.
//!
//! ## Seed
//!
//! This is the value used to seed the random number generator which is used to choose the initial centroids.
//!
//! Provide any arbitrary value like `0`, `42`, or `123456789`.

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

use hashbrown::HashMap;
use image::{DynamicImage, RgbImage, RgbaImage};
use palette::{IntoColor, Oklab, Srgb, Srgba, WithAlpha};
#[cfg(feature = "threads")]
use rayon::prelude::*;

mod kmeans;
pub use kmeans::KmeansResult;

/// The format used to convert an [`Srgb`] color into a `u32` for hashing
type Packed = palette::rgb::channels::Rgba;

/// Deduplicated [`Oklab`] colors converted from [`Srgb`] colors
#[derive(Debug, Clone)]
pub struct OklabCounts {
	/// [`Oklab`] colors and the corresponding number of duplicate [`Srgb`] pixels
	pub(crate) color_counts: Vec<(Oklab, u32)>,
	/// The value used to scale down the lightness of each color
	pub(crate) lightness_weight: f32,
}

impl OklabCounts {
	/// Gets the underlying Vec of [`Oklab`] colors and each corresponding [`u32`] count that indicates the number of duplicate [`Srgb`] pixels for the color.
	/// Each [`Oklab`] color's lightness component is scaled down according to the current `lightness_weight`.
	#[must_use]
	pub fn weighted_pairs(&self) -> &Vec<(Oklab, u32)> {
		&self.color_counts
	}

	/// Get the number of unique colors which is less than or equal to `2.pow(24)`
	#[must_use]
	#[allow(clippy::cast_possible_truncation)]
	pub fn num_colors(&self) -> u32 {
		// Only 2^8^3 = 2^24 possible Srgb colors
		debug_assert!(self.color_counts.len() <= usize::pow(2, 24));
		self.color_counts.len() as u32
	}

	/// Get the current lightness weight
	#[must_use]
	pub fn lightness_weight(&self) -> f32 {
		self.lightness_weight
	}

	/// Set the lightness weight to the provided value which should be in the range `0.0..=1.0`
	///
	/// `lightness_weight` is used to scale down each [`Oklab`] color's lightness when performing color difference.
	pub fn set_lightness_weight(&mut self, weight: f32) {
		// Values outside this range do not make sense but will technically work, so this is a debug assert
		debug_assert!((0.0..=1.0).contains(&weight));

		let lightness_weight = self.lightness_weight;

		#[allow(clippy::float_cmp)]
		if !(weight == lightness_weight
			|| (weight == 0.0 && lightness_weight == 1.0)
			|| (weight == 1.0 && lightness_weight == 0.0))
		{
			let new_weight = if weight == 0.0 { 1.0 } else { weight };
			let old_weight = if lightness_weight == 0.0 { 1.0 } else { lightness_weight };

			for color_count in &mut self.color_counts {
				color_count.0.l = (color_count.0.l / old_weight) * new_weight;
			}
		}

		self.lightness_weight = weight;
	}

	/// Set the lightness weight to the provided value which should be in the range `0.0..=1.0`
	///
	/// `lightness_weight` is used to scale down each [`Oklab`] color's lightness when performing color difference.
	#[must_use]
	pub fn with_lightness_weight(mut self, lightness_weight: f32) -> Self {
		self.set_lightness_weight(lightness_weight);
		self
	}

	/// Create an [`OklabCounts`] from a Vec of color counts
	#[cfg(feature = "threads")]
	fn from_thread_counts(mut thread_counts: Vec<HashMap<u32, u32>>) -> Self {
		// Merge counts from each thread
		let mut counts = thread_counts.pop().expect("one thread");
		for other_counts in thread_counts {
			for (key, add_count) in other_counts {
				*counts.entry(key).or_insert(0) += add_count;
			}
		}

		let color_counts = counts
			.into_par_iter()
			.map(|(key, count)| {
				let srgb = Srgb::from_u32::<Packed>(key);
				let oklab: Oklab = srgb.into_format().into_color();
				(oklab, count)
			})
			.collect::<Vec<_>>();

		OklabCounts { color_counts, lightness_weight: 1.0 }
	}

	/// Create an [`OklabCounts`] from a slice of [`Srgb`] colors
	#[cfg(feature = "threads")]
	#[must_use]
	pub fn from_srgb(pixels: &[Srgb<u8>]) -> Self {
		// Converting from Srgb to Oklab is expensive, so let's group identical pixels.
		// This will also have the effect of speeding up k-means, since there will be less data points.

		// We use hashbrown::HashMap instead of std::collections::HashMap, since:
		// - AHash is faster than SipHash (we do not need the DDoS protection)
		// - the standard HashMap uses thead-local random state which causes non-deterministic output with rayon,
		//   so we would have to sort the final colors/counts to restore determinism.
		let thread_counts = pixels
			.par_iter()
			// setting min_len reduces the number of intermediate HashMaps (needed to be merged at the end, etc.)
			.with_min_len(pixels.len() / rayon::current_num_threads())
			.fold(HashMap::new, |mut counts, srgb| {
				let key = srgb.into_u32::<Packed>();
				*counts.entry(key).or_insert(0) += 1_u32;
				counts
			})
			.collect::<Vec<_>>();

		Self::from_thread_counts(thread_counts)
	}

	/// Create an [`OklabCounts`] from a slice of [`Srgba`] colors
	///
	/// Colors with an alpha value less than `alpha_threshold` are excluded from the resulting [`OklabCounts`].
	#[cfg(feature = "threads")]
	#[must_use]
	pub fn from_srgba(pixels: &[Srgba<u8>], alpha_threshold: u8) -> Self {
		// Converting from Srgb to Oklab is expensive, so let's group identical pixels.
		// This will also have the effect of speeding up k-means, since there will be less data points.

		// We use hashbrown::HashMap instead of std::collections::HashMap, since:
		// - AHash is faster than SipHash (we do not need the DDoS protection)
		// - the standard HashMap uses thead-local random state which causes non-deterministic output with rayon,
		//   so we would have to sort the final colors/counts to restore determinism.
		let thread_counts = pixels
			.par_iter()
			// setting min_len reduces the number of intermediate HashMaps (needed to be merged at the end, etc.)
			.with_min_len(pixels.len() / rayon::current_num_threads())
			.fold(HashMap::new, |mut counts, srgb| {
				if srgb.alpha >= alpha_threshold {
					let key = srgb.with_alpha(0).into_u32::<Packed>();
					*counts.entry(key).or_insert(0) += 1_u32;
				}
				counts
			})
			.collect::<Vec<_>>();

		Self::from_thread_counts(thread_counts)
	}

	/// Create an [`OklabCounts`] from color counts
	#[cfg(not(feature = "threads"))]
	fn from_counts(counts: HashMap<u32, u32>) -> Self {
		let color_counts = counts
			.into_iter()
			.map(|(key, count)| (Srgb::from_u32::<Packed>(key).into_format().into_color(), count))
			.collect::<Vec<_>>();

		OklabCounts { color_counts, lightness_weight: 1.0 }
	}

	/// Create an [`OklabCounts`] from a slice of [`Srgb`] colors
	#[cfg(not(feature = "threads"))]
	#[must_use]
	pub fn from_srgb(pixels: &[Srgb<u8>]) -> Self {
		// Converting from Srgb to Oklab is expensive, so let's group identical pixels.
		// This will also have the effect of speeding up k-means, since there will be less data points.

		// Packed Srgb -> count
		let mut counts: HashMap<u32, u32> = HashMap::new();
		for srgb in pixels {
			let key = srgb.into_u32::<Packed>();
			*counts.entry(key).or_insert(0) += 1;
		}

		Self::from_counts(counts)
	}

	/// Create an [`OklabCounts`] from a slice of [`Srgba`] colors
	///
	/// Colors with an alpha value less than `alpha_threshold` are excluded from the resulting [`OklabCounts`].
	#[cfg(not(feature = "threads"))]
	#[must_use]
	pub fn from_srgba(pixels: &[Srgba<u8>], alpha_threshold: u8) -> Self {
		// Converting from Srgb to Oklab is expensive, so let's group identical pixels.
		// This will also have the effect of speeding up k-means, since there will be less data points.

		// Packed Srgb -> count
		let mut counts: HashMap<u32, u32> = HashMap::new();
		for srgb in pixels {
			if srgb.alpha >= alpha_threshold {
				let key = srgb.with_alpha(0).into_u32::<Packed>();
				*counts.entry(key).or_insert(0) += 1;
			}
		}

		Self::from_counts(counts)
	}

	/// Create an [`OklabCounts`] from an `RgbImage`
	#[must_use]
	pub fn from_rgbimage(image: &RgbImage) -> Self {
		Self::from_srgb(palette::cast::from_component_slice(image.as_raw()))
	}

	/// Create an [`OklabCounts`] from an `RgbaImage`
	///
	/// Pixels with an alpha value less than `alpha_threshold` are excluded from the resulting [`OklabCounts`].
	#[must_use]
	pub fn from_rgbaimage(image: &RgbaImage, alpha_threshold: u8) -> Self {
		Self::from_srgba(palette::cast::from_component_slice(image.as_raw()), alpha_threshold)
	}

	/// Create an [`OklabCounts`] from an `DynamicImage`
	///
	/// Pixels with an alpha value less than `alpha_threshold` are excluded from the resulting [`OklabCounts`].
	/// Of course, if the image does not have an alpha channel, then `alpha_threshold` is ignored.
	#[must_use]
	pub fn from_image(image: &DynamicImage, alpha_threshold: u8) -> Self {
		use image::DynamicImage::*;
		match image {
			&ImageLuma8(_) | &ImageLuma16(_) | &ImageRgb8(_) | &ImageRgb16(_) | &ImageRgb32F(_) => {
				Self::from_rgbimage(&image.to_rgb8())
			},

			&ImageLumaA8(_) | &ImageLumaA16(_) | &ImageRgba8(_) | &ImageRgba16(_) | &ImageRgba32F(_) => {
				Self::from_rgbaimage(&image.to_rgba8(), alpha_threshold)
			},

			_ => Self::from_rgbaimage(&image.to_rgba8(), alpha_threshold),
		}
	}
}

/// Runs k-means on a [`OklabCounts`].
///
/// See the crate documentation for examples and information on each parameter.
#[must_use]
pub fn run(
	oklab_counts: &OklabCounts,
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
) -> KmeansResult {
	kmeans::run(oklab_counts, trials, k, convergence_threshold, max_iter, seed)
}

#[cfg(test)]
mod tests {
	use super::*;
	use approx::assert_relative_eq;
	use palette::WithAlpha;

	fn test_colors() -> Vec<Srgb<u8>> {
		let range = (0..u8::MAX).step_by(16);
		let mut colors = Vec::with_capacity(range.len().pow(3));

		for r in range.clone() {
			for g in range.clone() {
				for b in range.clone() {
					colors.push(Srgb::new(r, g, b));
				}
			}
		}

		colors
	}

	fn cmp_oklab_count((x, _): &(Oklab, u32), (y, _): &(Oklab, u32)) -> std::cmp::Ordering {
		use std::cmp::Ordering::*;
		match f32::total_cmp(&x.l, &y.l) {
			Equal => match f32::total_cmp(&x.a, &y.a) {
				Equal => f32::total_cmp(&x.b, &y.b),
				cmp => cmp,
			},
			cmp => cmp,
		}
	}

	fn assert_oklab_counts_eq(result: &OklabCounts, expected: &OklabCounts) {
		assert_eq!(expected.lightness_weight, result.lightness_weight);

		for (expected, color) in expected.color_counts.iter().zip(&result.color_counts) {
			assert_relative_eq!(expected.0, color.0);
			assert_eq!(expected.1, color.1);
		}
	}

	#[test]
	#[allow(clippy::float_cmp)]
	fn set_lightness_weight_restores_lightness() {
		let mut oklab = OklabCounts::from_srgb(&test_colors());

		let expected = oklab.clone();

		oklab.set_lightness_weight(0.325);
		assert_ne!(expected.lightness_weight, oklab.lightness_weight);

		oklab.set_lightness_weight(expected.lightness_weight);
		assert_oklab_counts_eq(&expected, &oklab);
	}

	#[test]
	fn transparent_results_match() {
		let rgb = test_colors();
		let matte = rgb.iter().map(|color| color.with_alpha(u8::MAX)).collect::<Vec<_>>();
		let transparent = rgb.iter().map(|color| color.with_alpha(0_u8)).collect::<Vec<_>>();

		let mut expected = OklabCounts::from_srgb(&rgb);
		let mut matte_result_a = OklabCounts::from_srgba(&matte, u8::MAX);
		let mut matte_result_b = OklabCounts::from_srgba(&matte, 0);
		let mut transparent_result = OklabCounts::from_srgba(&transparent, 0);

		// colors may be in different order due to differences in rayon's scheduling between `from_srgb` and `from_srgba`
		expected.color_counts.sort_unstable_by(cmp_oklab_count);
		matte_result_a.color_counts.sort_unstable_by(cmp_oklab_count);
		matte_result_b.color_counts.sort_unstable_by(cmp_oklab_count);
		transparent_result.color_counts.sort_unstable_by(cmp_oklab_count);

		assert_oklab_counts_eq(&expected, &matte_result_a);
		assert_oklab_counts_eq(&expected, &matte_result_b);
		assert_oklab_counts_eq(&expected, &transparent_result);
	}

	#[test]
	fn alpha_threshold() {
		let transparent = test_colors()
			.iter()
			.map(|color| color.with_alpha(0_u8))
			.collect::<Vec<_>>();

		let result_a = OklabCounts::from_srgba(&transparent, u8::MAX);
		let result_b = OklabCounts::from_srgba(&transparent, 1);

		assert_eq!(result_a.num_colors(), 0);
		assert_eq!(result_b.num_colors(), 0);
	}
}
