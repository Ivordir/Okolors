//! Performs k-means clustering in the Oklab color space.

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
