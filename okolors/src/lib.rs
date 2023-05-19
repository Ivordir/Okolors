//! Perform k-means clustering in the Oklab color space.
//!
//! # Overview
//!
//! Okolors takes an image in the `sRGB` color space or a slice of [`Srgb`] colors and returns `k` average [`Oklab`] colors.
//!
//! See the [parameters](#parameters) section for information and recommended values for each parameter.
//!
//! For visual examples and more information (e.g., features, performance, or the Okolors binary)
//! see the [README](https://github.com/Ivordir/Okolors#readme).
//!
//! # Examples
//!
//! ## Read an image file and get 5 average colors.
//!
//! ```no_run
//! # fn main() -> Result<(), image::ImageError> {
//! let img = image::open("some image")?;
//! let result = okolors::from_image(&img, 0.325, 1, 5, 0.05, 64, 0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Run k-means multiple times with different arguments.
//!
//! ```no_run
//! # fn main() -> Result<(), image::ImageError> {
//! let img = image::open("some image")?;
//! let oklab = okolors::OklabCounts::from_image(&img, 0.325);
//!
//! let avg5 = okolors::from_oklab_counts(&oklab, 1, 5, 0.05, 64, 0);
//! let avg8 = okolors::from_oklab_counts(&oklab, 1, 8, 0.05, 64, 0);
//!
//! let result = if avg5.variance < avg8.variance { avg5 } else { avg8 };
//! # Ok(())
//! # }
//! ```
//!
//! ## Run with different lightness weights.
//!
//! ```no_run
//! # fn main() -> Result<(), image::ImageError> {
//! let img = image::open("some image")?;
//! let mut oklab = okolors::OklabCounts::from_image(&img, 0.325);
//!
//! let resultA = okolors::from_oklab_counts(&oklab, 1, 5, 0.05, 64, 0);
//!
//! oklab.set_lightness_weight(1.0);
//! let resultB = okolors::from_oklab_counts(&oklab, 1, 5, 0.05, 64, 0);
//! # Ok(())
//! # }
//! ```
//!
//! # Parameters
//!
//! Here are explanations of the various parameters that are shared between
//! [`from_image`], [`from_rgbimage`], [`from_srgb`], and [`from_oklab_counts`].
//!
//! In short, the `trials`, `convergence_threshold`, and `max_iter` parameters
//! control the color accuracy at the expense of running time.
//! `k` indicates the number of colors to return,
//! and `lightness_weight` is a subjective parameter that affects how the colors are clustered.
//!
//! The [README](https://github.com/Ivordir/Okolors#readme) contains some visual examples of the effects of the parameters below.
//!
//! Note that if `trials = 0`, `k = 0`, or an empty slice of Srgb colors is provided,
//! then the [`KmeansResult`] will have no centroids and a variance of `0.0`.
//!
//! ## Lightness Weight
//!
//! This is used to scale down the lightness component of the Oklab colors when performing color difference.
//!
//! A value around `0.325` seems to provide similar results to the CIELAB color space.
//!
//! Lightness weights should be in the range `0.0..=1.0`.
//! A value of `1.0` indicates no scaling, and performs color difference in the Oklab color space using standard euclidean distance.
//! Otherwise, lower weights have the effect of merging similar colors together, possibly bringing out more distinct hues.
//! Note that for weights near `0.0`, if the image contains black and white, then they will be averaged into a shade of gray.
//! Also, the lightness weight affects the final variance, so it does not make sense to compare two results using their variance
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
//! After a certain point, the centroids change so little that there is no longer a visual, percievable difference.
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
use image::{DynamicImage, RgbImage};
use palette::{FromColor, Oklab, Srgb};

mod kmeans;
pub use kmeans::KmeansResult;

/// Deduplicated [`Oklab`] colors converted from [`Srgb`] colors
#[derive(Debug, Clone)]
pub struct OklabCounts {
	/// Oklab colors
	pub(crate) colors: Vec<Oklab>,
	/// The number of duplicate [`Srgb`] pixels for each [`Oklab`] color
	pub(crate) counts: Vec<u32>,
	/// The value used to scale down the lightness of each color
	pub(crate) lightness_weight: f32,
}

impl OklabCounts {
	/// Create an `OklabCounts` with empty Vecs and a lightness weight of `1.0`
	#[must_use]
	pub const fn new() -> Self {
		Self {
			colors: Vec::new(),
			counts: Vec::new(),
			lightness_weight: 1.0,
		}
	}

	/// Get the underlying Vec of [`Oklab`] colors
	#[must_use]
	pub fn colors(&self) -> &Vec<Oklab> {
		&self.colors
	}

	/// Get the number of duplicate [`Srgb`] pixels for each [`Oklab`] color
	#[must_use]
	pub fn counts(&self) -> &Vec<u32> {
		&self.counts
	}

	/// Returns an iterator over each `(Oklab, count: u32)` pair
	pub fn pairs(&self) -> impl Iterator<Item = (Oklab, u32)> + '_ {
		self.colors.iter().copied().zip(self.counts.iter().copied())
	}

	/// Get the number of unique colors which is less than or equal to `2.pow(24)`
	#[must_use]
	#[allow(clippy::cast_possible_truncation)]
	pub fn num_colors(&self) -> u32 {
		// Only 2^8^3 = 2^24 possible Srgb colors
		debug_assert!(self.colors().len() <= usize::pow(2, 24));
		self.colors.len() as u32
	}

	/// Get the current lightness weight
	#[must_use]
	pub fn lightness_weight(&self) -> f32 {
		self.lightness_weight
	}

	/// Change the lightness weight to provided value which should be in the range `0.0..=1.0`.
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

			for color in &mut self.colors {
				color.l = (color.l / old_weight) * new_weight;
			}
		}

		self.lightness_weight = weight;
	}

	/// Converts a slice of [`Srgb`] colors to [`Oklab`] colors, merging duplicate [`Srgb`] colors in the process.
	///
	/// `lightness_weight` is used to scale down each color's lightness when performing color difference
	/// and should be in the range `0.0..=1.0`.
	#[cfg(feature = "threads")]
	#[must_use]
	pub fn from_srgb(pixels: &[Srgb<u8>], lightness_weight: f32) -> Self {
		use rayon::prelude::*;

		// Converting from Srgb to Oklab is expensive, so let's group identical pixels.
		// This will also have the effect of speeding up k-means, since there will be less data points.

		/// The format used to convert an Srgb pixel into a u32 for hashing
		type Packed = palette::rgb::channels::Rgba;

		// We use hashbrown::HashMap instead of std::collections::HashMap, since:
		// - AHash is faster than SipHash (we do not need the DDoS protection)
		// - the standard HashMap uses thead-local random state which causes non-deterministic output with rayon,
		//   so we would have to sort the final colors/counts to restore determinism.
		let mut thread_counts = pixels
			.par_iter()
			// setting min_len reduces the number of intermediate HashMaps (needed to be merged at the end, etc.)
			.with_min_len(pixels.len() / rayon::current_num_threads())
			.fold(HashMap::new, |mut counts, srgb| {
				let key = srgb.into_u32::<Packed>();
				*counts.entry(key).or_insert(0) += 1_u32;
				counts
			})
			.collect::<Vec<_>>();

		// Merge counts from each thread
		let mut counts = thread_counts.pop().expect("one thread");
		for other_counts in thread_counts {
			for (key, add_count) in other_counts {
				*counts.entry(key).or_insert(0) += add_count;
			}
		}

		let (colors, counts) = counts
			.into_par_iter()
			.map(|(key, count)| (Oklab::from_color(Srgb::from_u32::<Packed>(key).into_format()), count))
			.unzip();

		let mut data = OklabCounts { colors, counts, lightness_weight: 1.0 };
		data.set_lightness_weight(lightness_weight);
		data
	}

	/// Converts a slice of [`Srgb`] colors to [`Oklab`] colors, merging duplicate [`Srgb`] colors in the process.
	///
	/// `lightness_weight` is used to scale down each color's lightness when performing color difference
	/// and should be in the range `0.0..=1.0`.
	#[cfg(not(feature = "threads"))]
	#[must_use]
	pub fn from_srgb(pixels: &[Srgb<u8>], lightness_weight: f32) -> Self {
		// Converting from Srgb to Oklab is expensive, so let's group identical pixels.
		// This will also have the effect of speeding up k-means, since there will be less data points.

		/// The format used to convert an Srgb pixel into a u32 for hashing
		type Packed = palette::rgb::channels::Rgba;

		// Packed Srgb -> count
		let mut counts: HashMap<u32, u32> = HashMap::new();
		for srgb in pixels {
			let key = srgb.into_u32::<Packed>();
			*counts.entry(key).or_insert(0) += 1;
		}

		let (colors, counts) = counts
			.into_iter()
			.map(|(key, count)| (Oklab::from_color(Srgb::from_u32::<Packed>(key).into_format()), count))
			.unzip();

		let mut data = OklabCounts { colors, counts, lightness_weight: 1.0 };
		data.set_lightness_weight(lightness_weight);
		data
	}

	/// Converts an [`RgbImage`]'s colors to [`Oklab`] colors, merging duplicate [`Srgb`] colors in the process.
	///
	/// `lightness_weight` is used to scale down each color's lightness when performing color difference
	/// and should be in the range `0.0..=1.0`.
	#[must_use]
	pub fn from_rgbimage(image: &RgbImage, lightness_weight: f32) -> Self {
		Self::from_srgb(palette::cast::from_component_slice(image.as_raw()), lightness_weight)
	}

	/// Converts an image's [`Srgb`] colors to [`Oklab`] colors, merging duplicate [`Srgb`] colors in the process.
	///
	/// `lightness_weight` is used to scale down each color's lightness when performing color difference
	/// and should be in the range `0.0..=1.0`.
	#[must_use]
	pub fn from_image(image: &DynamicImage, lightness_weight: f32) -> Self {
		Self::from_rgbimage(&image.to_rgb8(), lightness_weight)
	}
}

/// Runs k-means on the provided slice of [`Srgb`] colors.
///
/// See the crate documentation for examples and information on each parameter.
#[must_use]
pub fn from_srgb(
	pixels: &[Srgb<u8>],
	lightness_weight: f32,
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
) -> KmeansResult {
	from_oklab_counts(
		&OklabCounts::from_srgb(pixels, lightness_weight),
		trials,
		k,
		convergence_threshold,
		max_iter,
		seed,
	)
}

/// Runs k-means on the provided [`RgbImage`]. The image is assumed to be in the `sRGB` color space.
///
/// See the crate documentation for examples and information on each parameter.
#[must_use]
pub fn from_rgbimage(
	image: &RgbImage,
	lightness_weight: f32,
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
) -> KmeansResult {
	from_oklab_counts(
		&OklabCounts::from_rgbimage(image, lightness_weight),
		trials,
		k,
		convergence_threshold,
		max_iter,
		seed,
	)
}

/// Runs k-means on the provided image. The image is assumed to be in the `sRGB` color space.
///
/// See the crate documentation for examples and information on each parameter.
#[must_use]
pub fn from_image(
	image: &DynamicImage,
	lightness_weight: f32,
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
) -> KmeansResult {
	from_oklab_counts(
		&OklabCounts::from_image(image, lightness_weight),
		trials,
		k,
		convergence_threshold,
		max_iter,
		seed,
	)
}

/// Runs k-means on a [`OklabCounts`].
///
/// Converting from [`Srgb`] to [`Oklab`] is expensive,
/// so use this function if you need to run k-means multiple times on the same data but with different arguments.
/// This function allows you to reuse the [`OklabCounts`],
/// whereas [`from_image`], [`from_rgbimage`], and [`from_srgb`] must recompute [`OklabCounts`] every time.
///
/// See the crate documentation for examples and information on each parameter.
#[must_use]
pub fn from_oklab_counts(
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

	pub fn assert_oklab_eq(x: Oklab, y: Oklab, eps: f32) {
		assert!((x.l - y.l).abs() <= eps);
		assert!((x.a - y.a).abs() <= eps);
		assert!((x.b - y.b).abs() <= eps);
	}

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

	#[test]
	#[allow(clippy::float_cmp)]
	fn set_lightness_weight_restores_lightness() {
		let mut oklab = OklabCounts::new();

		for color in test_colors() {
			oklab.colors.push(Oklab::from_color(color.into_format()));
			oklab.counts.push(1);
		}

		let expected = oklab.clone();

		oklab.set_lightness_weight(0.325);
		oklab.set_lightness_weight(1.0);

		for (&color, &expected) in oklab.colors.iter().zip(&expected.colors) {
			assert_oklab_eq(color, expected, 1e-7);
		}

		assert_eq!(expected.counts, oklab.counts);
		assert_eq!(expected.lightness_weight, oklab.lightness_weight);
	}
}
