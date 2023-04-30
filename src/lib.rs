//! Perform k-means clustering in the Oklab color space.
//!
//! # Examples
//!
//! ## Read an image file and get 5 average colors.
//!
//! ```rust
//! let pixels = image::open(path).unwrap().into_rgb8();
//! let srgb = palette::cast::from_component_slice(pixels.as_raw());
//! let result = okolors::from_srgb(srgb, 1, 5, 0.05, 128, 0, false);
//! ```
//!
//! ## Run k-means multiple times with different settings.
//!
//! ```rust
//! let pixels = image::open(path).unwrap().into_rgb8();
//! let srgb = palette::cast::from_component_slice(pixels.as_raw());
//! let lab = okolors::srgb_to_oklab_counts(srgb);
//!
//! let avg5 = okolors::from_oklab_counts(lab, 1, 5, 0.05, 64, 0, false);
//! let avg8 = okolors::from_oklab_counts(lab, 1, 8, 0.05, 64, 0, false);
//!
//! let result = if avg5.variance < avg8.variance { avg5 } else { avg8 };
//! ```
//!
//! # Arguments
//!
//! Here are explanations of the various arguments that are shared between
//! [`okolors::from_srgb`] and [`okolors::from_oklab_counts`].
//!
//! The [README](https://github.com/Ivordir/Okolors) contains some visual examples of the effects of the arguments below.
//!
//! Note that if `trials` = 0, `k` = 0, or an empty slice of Srgb colors is provided,
//! then the [`KmeansResult`] will have no centroids and a variance of `0.0`.
//!
//! ## Trials
//!
//! This is the number of times to run k-means, taking the trial with the lowest variance.
//!
//! 1 to 4 trials is recommended, possibly up to 8 if you want more accuracy.
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
//! 4 to 16 is most likely the range you want.
//!
//! The ideal number of clusters is hard to know in advance, if there even is an "ideal" number.
//! Lower k values give faster runtime but also typically lower color accuracy.
//! Higher k values provide higher accuracy, but can potentially give more colors than you need/want.
//! The [`KmeansResult`] will provide the number of pixels in each color/centroid,
//! so this can be used to filter out colors that make less than a certain percentage of the image.
//!
//! ## Convergence Threshold
//!
//! After a certain point, the centroids (average colors) change so little that there is no longer a visual, percievable difference.
//! This is the cutoff value used to determine whether any change is significant.
//!
//! 0.01 to 0.1 is the recommended range, with lower values indicating higher accuracy.
//!
//! A value of `0.1` is very fast (often only a few iterations are needed for regular sized images),
//! and this should be enough to get a decent looking palette.
//!
//! A value of `0.01` is the lowest sensible value for maximum accuracy.
//! Convergence thresholds should probably be not too far lower than this,
//! as any iterations after this either do not or barely effect the final colors once converted to Srgb.
//! I.e., don't use `0.0` as the convergence threshold,
//! as that may require many more iterations and would just be wasting time.
//!
//! ## Max Iterations
//!
//! This is the maximum number of iterations allowed for each k-means trial.
//!
//! 16 to 64 iterations is a decent range to use.
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
//!
//! ## Ignore Lightness
//!
//! This controls whether or not the lightness component of the Oklab colors is factored in when performing color difference.
//!
//! If this is `true`, then colors will only be compared using their hue and saturation.
//! This has the effect of merging similar colors together, possibly bringing out more distinct hues.
//! One downside is that if the image contains black and white, then they will be averaged into a shade of gray.
//!
//! Otherwise, colors are compared using standard euclidean distance in the Oklab color space
//! where lightness has a compartively higher influence on color difference than the other components.

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

use palette::{FromColor, Oklab, Srgb};
use std::collections::HashMap;

mod kmeans;
pub use kmeans::KmeansResult;

/// Deduplicated Oklab colors converted from Srgb colors
pub struct OklabCounts {
	/// Oklab colors
	colors: Vec<Oklab>,
	/// The number of duplicate Srgb pixels for each Oklab color
	counts: Vec<u32>,
}

impl OklabCounts {
	/// Create an `OklabCounts` with empty Vecs
	const fn new() -> Self {
		Self { colors: Vec::new(), counts: Vec::new() }
	}
}

/// Runs k-means on the provided slice of Srgb colors.
///
/// See the crate documentation for examples and information on each argument.
#[must_use]
pub fn from_srgb(
	pixels: &[Srgb<u8>],
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
	ignore_lightness: bool,
) -> KmeansResult {
	from_oklab_counts(
		&srgb_to_oklab_counts(pixels),
		trials,
		k,
		convergence_threshold,
		max_iter,
		seed,
		ignore_lightness,
	)
}

/// Runs k-means on a `OklabCounts` from `okolors::srgb_to_oklab_counts`
///
/// Converting from Srgb to Oklab colors is expensive,
/// so use this function if you need to run k-means multiple times on the same data but with different arguments.
/// This function allows you to reuse the `OklabCounts` from `srgb_to_oklab_counts`,
/// whereas `okolors::from_srgb` must recompute `OklabCounts` every time.
///
/// See the crate documentation for examples and information on each argument.
#[must_use]
pub fn from_oklab_counts(
	oklab_counts: &OklabCounts,
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
	ignore_lightness: bool,
) -> KmeansResult {
	kmeans::run(
		oklab_counts,
		trials,
		k,
		convergence_threshold,
		max_iter,
		seed,
		ignore_lightness,
	)
}

/// Converts a slice of Srgb colors to Oklab colors, merging duplicate Srgb colors in the process.
#[must_use]
pub fn srgb_to_oklab_counts(pixels: &[Srgb<u8>]) -> OklabCounts {
	let mut data = OklabCounts::new();

	// Converting from Srgb to Oklab is expensive.
	// Memoizing the results almost halves the time needed.
	// This also groups identical pixels, speeding up k-means.

	// Packed Srgb -> data index
	let mut memo: HashMap<u32, u32> = HashMap::new();

	// Convert to an Oklab color, merging entries as necessary
	for srgb in pixels {
		let key = srgb.into_u32::<palette::rgb::channels::Rgba>();
		let index = *memo.entry(key).or_insert_with(|| {
			let color = Oklab::from_color(srgb.into_format());

			// data.len() < u32::MAX because there are only (2^8)^3 < u32::MAX possible sRGB colors
			#[allow(clippy::cast_possible_truncation)]
			let index = data.colors.len() as u32;

			data.colors.push(color);
			data.counts.push(0);
			index
		});

		data.counts[index as usize] += 1;
	}

	data
}
