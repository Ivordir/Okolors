//! # **Note**
//! **This module currently does not follow semantic versioning.**
//!
//! This module contains low level functions for use in the Okolors CLI application.

use crate::{Oklab, Srgb};
use palette::{
    encoding::{self, FromLinear},
    rgb::RgbStandard,
    FromColor, LinSrgb,
};
use quantette::{
    kmeans::{self, Centroids},
    wu::{self, FloatBinner},
    ColorSlice, ColorSpace, PaletteSize, UniqueColorCounts,
};

pub use quantette::QuantizeOutput;

/// [`Oklab`] colors from deduplicated [`Srgb`] colors.
pub type UniqueOklabCounts = UniqueColorCounts<Oklab, f32, 3>;

/// Create a [`UniqueOklabCounts`] from [`Srgb`] colors while applying the given lightness weight.
#[must_use]
pub fn unique_oklab_counts(
    colors: ColorSlice<Srgb<u8>>,
    lightness_weight: f32,
) -> UniqueOklabCounts {
    UniqueColorCounts::new(colors, |srgb| {
        let mut oklab = Oklab::from_color(srgb.into_linear());
        oklab.l *= lightness_weight;
        oklab
    })
}

/// Create a [`UniqueOklabCounts`] in parallel from [`Srgb`] colors
/// while applying the given lightness weight.
#[cfg(feature = "threads")]
#[must_use]
pub fn unique_oklab_counts_par(
    colors: ColorSlice<Srgb<u8>>,
    lightness_weight: f32,
) -> UniqueOklabCounts {
    UniqueColorCounts::new_par(colors, |srgb| {
        let mut oklab = Oklab::from_color(srgb.into_linear());
        oklab.l *= lightness_weight;
        oklab
    })
}

/// Create an [`Oklab`] binner for the given lightness weight.
fn binner(lightness_weight: f32) -> FloatBinner<f32, 32> {
    let mut ranges = ColorSpace::OKLAB_F32_COMPONENT_RANGES_FROM_SRGB;
    ranges[0].1 *= lightness_weight;
    FloatBinner::new(ranges)
}

/// Generate a color palette using Wu's color quantizer.
#[must_use]
pub fn wu_palette(
    unique: &UniqueOklabCounts,
    palette_size: PaletteSize,
    lightness_weight: f32,
) -> QuantizeOutput<Oklab> {
    wu::palette(unique, palette_size, &binner(lightness_weight))
}

/// Generate a color palette in parallel using Wu's color quantizer.
#[cfg(feature = "threads")]
#[must_use]
pub fn wu_palette_par(
    unique: &UniqueOklabCounts,
    palette_size: PaletteSize,
    lightness_weight: f32,
) -> QuantizeOutput<Oklab> {
    wu::palette_par(unique, palette_size, &binner(lightness_weight))
}

/// Returns the number of samples based off the sampling factor.
#[must_use]
pub fn num_samples(unique: &UniqueOklabCounts, sampling_factor: f32) -> u32 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (f64::from(sampling_factor) * f64::from(unique.num_colors())) as u32
    }
}

/// Generate a color palette using k-means clustering.
#[must_use]
pub fn kmeans_palette(
    unique: &UniqueOklabCounts,
    samples: u32,
    palette: Vec<Oklab>,
    seed: u64,
) -> QuantizeOutput<Oklab> {
    let centroids = Centroids::from_truncated(palette);
    kmeans::palette(unique, samples, centroids, seed)
}

/// Generate a color palette in parallel using k-means clustering.
#[cfg(feature = "threads")]
#[must_use]
pub fn kmeans_palette_par(
    unique: &UniqueOklabCounts,
    samples: u32,
    batch_size: u32,
    palette: Vec<Oklab>,
    seed: u64,
) -> QuantizeOutput<Oklab> {
    let centroids = Centroids::from_truncated(palette);
    kmeans::palette_par(unique, samples, batch_size, centroids, seed)
}

/// Unapply the lightness weight to get the final palette colors.
pub fn restore_lightness(palette: &mut [Oklab], lightness_weight: f32) {
    for color in palette {
        color.l /= lightness_weight;
    }
}

/// Sort the palette by ascending frequency
#[must_use]
pub fn sort_by_frequency<Color>(output: QuantizeOutput<Color>) -> Vec<Color> {
    let mut pairs = output
        .palette
        .into_iter()
        .zip(output.counts)
        .collect::<Vec<_>>();

    pairs.sort_by_key(|&(_, n)| n);

    pairs.into_iter().map(|(c, _)| c).collect()
}

/// Convert an [`Oklab`] palette to an [`Srgb`] palette.
#[must_use]
pub fn oklab_to_srgb<T>(oklab: Vec<Oklab>) -> Vec<Srgb<T>>
where
    <encoding::Srgb as RgbStandard>::TransferFn: FromLinear<f32, T>,
{
    oklab
        .into_iter()
        .map(|oklab| LinSrgb::from_color(oklab).into_encoding())
        .collect()
}
