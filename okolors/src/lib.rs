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
//! let oklab = okolors::OklabCounts::try_from_image(&img, u8::MAX).expect("non-gigantic image");
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
//! let mut oklab = okolors::OklabCounts::try_from_image(&img, u8::MAX)
//!     .expect("non-gigantic image")
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
// temporary?
#![allow(clippy::multiple_crate_versions)]

use image::{buffer::ConvertBuffer, DynamicImage, RgbImage, RgbaImage};
use palette::{IntoColor, Oklab, Srgb, Srgba, WithAlpha};
#[cfg(feature = "threads")]
use rayon::prelude::*;
use std::ops::Range;

pub(crate) mod kmeans;
pub use kmeans::KmeansResult;

pub mod gpu;

/// An [`Oklab`] color and a count for the number of pixels that are this color.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OklabN {
	/// The lightness component for an [`Oklab`] color
	pub l: f32,
	/// The a component for an [`Oklab`] color
	pub a: f32,
	/// The b component for an [`Oklab`] color
	pub b: f32,
	/// The number of pixel that are this color
	pub n: u32,
}

impl OklabN {
	/// Create a new [`OklabN`] from an [`Oklab`] color and a `u32` count
	fn new(oklab: Oklab, n: u32) -> Self {
		let Oklab { l, a, b } = oklab;
		OklabN { l, a, b, n }
	}

	/// Get the underlying [`Oklab`] color
	fn to_lab(self) -> Oklab {
		let OklabN { l, a, b, .. } = self;
		Oklab { l, a, b }
	}

	/// Return the underlying [`Oklab`] color and `u32` count
	fn to_tuple(self) -> (Oklab, u32) {
		(self.to_lab(), self.n)
	}
}

/// Deduplicated [`Oklab`] colors converted from [`Srgb`] colors
#[derive(Debug, Clone)]
pub struct OklabCounts {
	/// [`Oklab`] colors and the corresponding number of duplicate [`Srgb`] pixels
	pub(crate) color_counts: Vec<OklabN>,
	/// The value used to scale down the lightness of each color
	pub(crate) lightness_weight: f32,
}

impl OklabCounts {
	/// Gets the underlying Vec of [`Oklab`] colors and each corresponding [`u32`] count that indicates the number of duplicate [`Srgb`] pixels for the color.
	/// Each [`Oklab`] color's lightness component is scaled down according to the current `lightness_weight`.
	#[must_use]
	pub fn weighted_pairs(&self) -> &Vec<OklabN> {
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

			for labn in &mut self.color_counts {
				labn.l = (labn.l / old_weight) * new_weight;
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

	/// A byte-sized Radix
	const RADIX: usize = u8::MAX as usize + 1;

	/// Computes the prefix sum of an array in place
	#[inline]
	fn prefix_sum(counts: &mut [u32; Self::RADIX + 1]) {
		for i in 1..Self::RADIX {
			counts[i] += counts[i - 1];
		}
	}

	/// Return a [`Range`] over the `i`-th chunk by doing necessary conversions/casts
	#[inline]
	fn get_chunk(chunks: &[u32; Self::RADIX + 1], i: usize) -> Range<usize> {
		(chunks[i] as usize)..(chunks[i + 1] as usize)
	}

	/*
		The following preprocessing step is arguably the most important section with regards to running time.
		This function will deduplicate the provided pixels using a partial radix sort
		and then finally convert the unique Srgb colors to the Oklab color space.

		Why do we deduplicate?
		1.
			The running time of each k-means iteration is O(n * k * d)
			where n is the number of data points, pixels in this case.
			For many images the number of unique colors is 16 to 60 times less than the number of pixels.
			So, this alone already results in a massive speedup.

		2.
			Converting from Srgb to Oklab is expensive.
			Each Srgb pixel first needs to be linearized, this takes a 6 floating point operations and 3 powf() calls.
			The linearized color is then converted to Oklab which takes another 36 flops and 3 cbrt() calls.
			By converting only after deduplicating, this also greatly reduces the running time.

		Why do we use a radix sort based approach opposed to, e.g., a HashMap approach?
		1.
			It's faster -- who would've guessed!
			It's hard to beat the speed of radix sort. The overhead of a HashMap is too large in this case.

		2.
			It gives a roughly 20% time reduction for the later k-means algorithm compared to the HashMap approach.
			My guess is that the sorting approach will group similar colors together,
			thereby decreasing the number of branch mispredictions in k-means.
			Collecting a Vec from a HashMap, on the other hand, may give pixels in any random order.

		3.
			Because of the random iteration order for HashMaps, the HashMap approach may provide different results
			for different numbers of threads. Ultimately, this would affect the final k-means results.
			The radix sort approach, on the other hand, will provide the same result regardless of the number of threads.

		One thing to note is that the radix approach may be slower for (very?) small inputs/images,
		but the goal is to reduce the time for large inputs where the difference can be most felt.
	*/

	/// Create an [`OklabCounts`] from a slice of [`Srgb`] colors
	///
	/// # Errors
	/// An `Error` is returned if the length of `pixels` is greater than `u32::MAX`.
	/// Otherwise, the result can be safely unwrapped.
	#[cfg(feature = "threads")]
	pub fn try_from_srgb(pixels: &[Srgb<u8>]) -> Result<Self, std::num::TryFromIntError> {
		if pixels.is_empty() {
			Ok(OklabCounts {
				color_counts: Vec::new(),
				lightness_weight: 1.0,
			})
		} else {
			let n = u32::try_from(pixels.len())?;

			let mut green_blue = vec![(0, 0); pixels.len()];

			let mut red_prefix = [0; Self::RADIX + 1];

			// Excuse the manual unrolling below...

			for rgb in pixels {
				red_prefix[usize::from(rgb.red)] += 1;
			}

			Self::prefix_sum(&mut red_prefix);

			for rgb in pixels {
				let r = usize::from(rgb.red);
				let i = red_prefix[r] - 1;
				green_blue[i as usize] = (rgb.green, rgb.blue);
				red_prefix[r] = i;
			}
			red_prefix[Self::RADIX] = n;

			let color_counts = (0..Self::RADIX)
				.into_par_iter()
				.flat_map(|r| {
					let chunk = Self::get_chunk(&red_prefix, r);

					let mut color_counts = Vec::new();

					if !chunk.is_empty() {
						#[allow(clippy::cast_possible_truncation)]
						let chunk_len = chunk.len() as u32;

						let mut green_prefix = [0; Self::RADIX + 1];
						let mut blue_counts = [0; Self::RADIX];
						let mut blue = vec![0; chunk.len()];

						for &(green, _) in &green_blue[chunk.clone()] {
							green_prefix[usize::from(green)] += 1;
						}

						Self::prefix_sum(&mut green_prefix);

						for &(g, b) in &green_blue[chunk] {
							let g = usize::from(g);
							let i = green_prefix[g] - 1;
							blue[i as usize] = b;
							green_prefix[g] = i;
						}
						green_prefix[Self::RADIX] = chunk_len;

						for g in 0..Self::RADIX {
							let chunk = Self::get_chunk(&green_prefix, g);

							if !chunk.is_empty() {
								for &b in &blue[chunk] {
									blue_counts[usize::from(b)] += 1;
								}

								for (b, &count) in blue_counts.iter().enumerate() {
									if count > 0 {
										#[allow(clippy::cast_possible_truncation)]
										let srgb = Srgb::new(r as u8, g as u8, b as u8);
										let oklab: Oklab = srgb.into_format().into_color();
										color_counts.push(OklabN::new(oklab, count));
									}
								}

								blue_counts = [0; Self::RADIX];
							}
						}
					}

					color_counts
				})
				.collect::<Vec<_>>();

			Ok(OklabCounts { color_counts, lightness_weight: 1.0 })
		}
	}

	/// Create an [`OklabCounts`] from a slice of [`Srgb`] colors
	///
	/// # Errors
	/// An `Error` is returned if the length of `pixels` is greater than `u32::MAX`.
	/// Otherwise, the result can be safely unwrapped.
	#[cfg(not(feature = "threads"))]
	pub fn try_from_srgb(pixels: &[Srgb<u8>]) -> Result<Self, std::num::TryFromIntError> {
		if pixels.is_empty() {
			Ok(OklabCounts {
				color_counts: Vec::new(),
				lightness_weight: 1.0,
			})
		} else {
			let n = u32::try_from(pixels.len())?;

			let mut color_counts = Vec::new();

			let mut green_blue = vec![(0, 0); pixels.len()];
			let mut blue = Vec::new();

			let mut red_prefix = [0; Self::RADIX + 1];
			let mut green_prefix = [0; Self::RADIX + 1];
			let mut blue_counts = [0; Self::RADIX];

			// Excuse the manual unrolling below...

			for rgb in pixels {
				red_prefix[usize::from(rgb.red)] += 1;
			}

			Self::prefix_sum(&mut red_prefix);

			for rgb in pixels {
				let r = usize::from(rgb.red);
				let i = red_prefix[r] - 1;
				green_blue[i as usize] = (rgb.green, rgb.blue);
				red_prefix[r] = i;
			}
			red_prefix[Self::RADIX] = n;

			for r in 0..Self::RADIX {
				let chunk = Self::get_chunk(&red_prefix, r);

				if !chunk.is_empty() {
					blue.resize(chunk.len(), 0);

					for &(green, _) in &green_blue[chunk.clone()] {
						green_prefix[usize::from(green)] += 1;
					}

					Self::prefix_sum(&mut green_prefix);

					for &(g, b) in &green_blue[chunk.clone()] {
						let g = usize::from(g);
						let i = green_prefix[g] - 1;
						blue[i as usize] = b;
						green_prefix[g] = i;
					}
					#[allow(clippy::cast_possible_truncation)]
					let chunk_len = chunk.len() as u32;
					green_prefix[Self::RADIX] = chunk_len;

					for g in 0..Self::RADIX {
						let chunk = Self::get_chunk(&green_prefix, g);

						if !chunk.is_empty() {
							for &b in &blue[chunk] {
								blue_counts[usize::from(b)] += 1;
							}

							for (b, &count) in blue_counts.iter().enumerate() {
								if count > 0 {
									#[allow(clippy::cast_possible_truncation)]
									let srgb = Srgb::new(r as u8, g as u8, b as u8);
									let oklab: Oklab = srgb.into_format().into_color();
									color_counts.push((oklab, count));
								}
							}

							blue_counts = [0; Self::RADIX];
						}
					}

					green_prefix = [0; Self::RADIX + 1];
				}
			}

			Ok(OklabCounts { color_counts, lightness_weight: 1.0 })
		}
	}

	/// Create an [`OklabCounts`] from a slice of [`Srgba`] colors
	///
	/// Colors with an alpha value less than `alpha_threshold` are excluded from the resulting [`OklabCounts`].
	///
	/// # Errors
	/// An `Error` is returned if the length of `pixels` is greater than `u32::MAX`.
	/// Otherwise, the result can be safely unwrapped.
	#[cfg(feature = "threads")]
	pub fn try_from_srgba(pixels: &[Srgba<u8>], alpha_threshold: u8) -> Result<Self, std::num::TryFromIntError> {
		Self::try_from_srgb(
			&pixels
				.par_iter()
				.filter_map(|c| {
					if c.alpha >= alpha_threshold {
						Some(c.without_alpha())
					} else {
						None
					}
				})
				.collect::<Vec<_>>(),
		)
	}

	/// Create an [`OklabCounts`] from a slice of [`Srgba`] colors
	///
	/// Colors with an alpha value less than `alpha_threshold` are excluded from the resulting [`OklabCounts`].
	///
	/// # Errors
	/// An `Error` is returned if the length of `pixels` is greater than `u32::MAX`.
	/// Otherwise, the result can be safely unwrapped.
	#[cfg(not(feature = "threads"))]
	pub fn try_from_srgba(pixels: &[Srgba<u8>], alpha_threshold: u8) -> Result<Self, std::num::TryFromIntError> {
		Self::try_from_srgb(
			&pixels
				.iter()
				.filter_map(|c| {
					if c.alpha >= alpha_threshold {
						Some(c.without_alpha())
					} else {
						None
					}
				})
				.collect::<Vec<_>>(),
		)
	}

	/// Create an [`OklabCounts`] from an `RgbImage`
	///
	/// # Errors
	/// An `Error` is returned if the number of pixels in `image` is greater than `u32::MAX`.
	/// Otherwise, the result can be safely unwrapped.
	pub fn try_from_rgbimage(image: &RgbImage) -> Result<Self, std::num::TryFromIntError> {
		Self::try_from_srgb(palette::cast::from_component_slice(image.as_raw()))
	}

	/// Create an [`OklabCounts`] from an `RgbaImage`
	///
	/// Pixels with an alpha value less than `alpha_threshold` are excluded from the resulting [`OklabCounts`].
	///
	/// # Errors
	/// An `Error` is returned if the number of pixels in `image` is greater than `u32::MAX`.
	/// Otherwise, the result can be safely unwrapped.
	pub fn try_from_rgbaimage(image: &RgbaImage, alpha_threshold: u8) -> Result<Self, std::num::TryFromIntError> {
		if alpha_threshold == 0 {
			Self::try_from_rgbimage(&image.convert())
		} else {
			Self::try_from_srgba(palette::cast::from_component_slice(image.as_raw()), alpha_threshold)
		}
	}

	/// Create an [`OklabCounts`] from an `DynamicImage`
	///
	/// Pixels with an alpha value less than `alpha_threshold` are excluded from the resulting [`OklabCounts`].
	/// Of course, if the image does not have an alpha channel, then `alpha_threshold` is ignored.
	///
	/// # Errors
	/// An `Error` is returned if the number of pixels in `image` is greater than `u32::MAX`.
	/// Otherwise, the result can be safely unwrapped.
	pub fn try_from_image(image: &DynamicImage, alpha_threshold: u8) -> Result<Self, std::num::TryFromIntError> {
		use image::DynamicImage::*;
		match image {
			&ImageLuma8(_) | &ImageLuma16(_) | &ImageRgb8(_) | &ImageRgb16(_) | &ImageRgb32F(_) => {
				Self::try_from_rgbimage(&image.to_rgb8())
			},

			&ImageLumaA8(_) | &ImageLumaA16(_) | &ImageRgba8(_) | &ImageRgba16(_) | &ImageRgba32F(_)
				if alpha_threshold == 0 =>
			{
				Self::try_from_rgbimage(&image.to_rgb8())
			},

			&ImageLumaA8(_) | &ImageLumaA16(_) | &ImageRgba8(_) | &ImageRgba16(_) | &ImageRgba32F(_) => {
				Self::try_from_rgbaimage(&image.to_rgba8(), alpha_threshold)
			},

			_ if alpha_threshold == 0 => Self::try_from_rgbimage(&image.to_rgb8()),

			_ => Self::try_from_rgbaimage(&image.to_rgba8(), alpha_threshold),
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

	fn assert_oklab_counts_eq(expected: &OklabCounts, result: &OklabCounts) {
		assert_eq!(expected.lightness_weight, result.lightness_weight);

		for (expected, color) in expected.color_counts.iter().zip(&result.color_counts) {
			assert_relative_eq!(expected.to_lab(), color.to_lab());
			assert_eq!(expected.n, color.n);
		}
	}

	#[test]
	#[allow(clippy::float_cmp)]
	fn set_lightness_weight_restores_lightness() {
		let mut oklab = OklabCounts::try_from_srgb(&test_colors()).expect("non-gigantic slice");

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

		let expected = OklabCounts::try_from_srgb(&rgb).expect("non-gigantic slice");
		let matte_result_a = OklabCounts::try_from_srgba(&matte, u8::MAX).expect("non-gigantic slice");
		let matte_result_b = OklabCounts::try_from_srgba(&matte, 0).expect("non-gigantic slice");
		let transparent_result = OklabCounts::try_from_srgba(&transparent, 0).expect("non-gigantic slice");

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

		let result_a = OklabCounts::try_from_srgba(&transparent, u8::MAX).expect("non-gigantic slice");
		let result_b = OklabCounts::try_from_srgba(&transparent, 1).expect("non-gigantic slice");

		assert_eq!(result_a.num_colors(), 0);
		assert_eq!(result_b.num_colors(), 0);
	}

	#[test]
	#[cfg(feature = "threads")]
	fn different_num_threads_match() {
		let rgb = test_colors();

		let expected = OklabCounts::try_from_srgb(&rgb).expect("non-gigantic slice");

		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(rayon::current_num_threads() / 2)
			.build()
			.expect("built thread pool");

		let result = pool.install(|| OklabCounts::try_from_srgb(&rgb).expect("non-gigantic slice"));

		assert_oklab_counts_eq(&expected, &result);
	}
}
