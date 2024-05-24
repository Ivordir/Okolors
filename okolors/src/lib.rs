//! Create a color palette from an image using k-means clustering in the Oklab color space.
//!
//! This library is a simple wrapper around the [`quantette`] crate
//! but only exposes functionality for generating color palettes.
//! Additionally, this crate adds a few additional options not present in [`quantette`].
//!
//! # Features
//! This crate has two features that are enabled by default:
//! - `threads`: adds parallel versions of the palette functions (see below).
//! - `image`: enables integration with the [`image`] crate.
//!
//! # Examples
//!
//! To start, create an [`Okolors`] from a [`RgbImage`] (note that the `image` feature is needed):
//! ```no_run
//! # use okolors::Okolors;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let img = image::open("some image")?.into_rgb8();
//! let palette_builder = Okolors::try_from(&img)?;
//! # Ok(())
//! # }
//! ```
//!
//! Instead of an [`RgbImage`], a slice of [`Srgb<u8>`] colors can be used instead:
//! ```
//! # use okolors::Okolors;
//! # use quantette::AboveMaxLen;
//! # use palette::Srgb;
//! # fn main() -> Result<(), AboveMaxLen<u32>> {
//! let srgb = vec![Srgb::new(0, 0, 0)];
//! let palette_builder = Okolors::try_from(srgb.as_slice())?;
//! # Ok(())
//! # }
//! ```
//!
//! If the default options aren't to your liking, you can tweak them:
//! ```no_run
//! # use okolors::Okolors;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let img = image::open("some image")?.into_rgb8();
//! let palette_builder = Okolors::try_from(&img)?
//!     .palette_size(16)
//!     .lightness_weight(0.5)
//!     .sampling_factor(0.25);
//! # Ok(())
//! # }
//! ```
//!
//! To finally generate the palette, use:
//! - [`Okolors::srgb8_palette`] for a [`Srgb<u8>`] palette
//! - [`Okolors::srgb_palette`] for a [`Srgb`] palette (components are `f32` instead of `u8`)
//! - [`Okolors::oklab_palette`] for an [`Oklab`] palette
//!
//! For example:
//! ```
//! # use okolors::Okolors;
//! # use quantette::AboveMaxLen;
//! # use palette::Srgb;
//! # fn main() -> Result<(), AboveMaxLen<u32>> {
//! # let srgb = vec![Srgb::new(0, 0, 0)];
//! # let palette_builder = Okolors::new(srgb.as_slice().try_into()?);
//! let palette = palette_builder.srgb8_palette();
//! # Ok(())
//! # }
//! ```
//!
//! If the `threads` feature is enabled, you can enable parallelism with [`Okolors::parallel`].

#![deny(unsafe_code, unsafe_op_in_unsafe_fn)]
#![warn(
    clippy::pedantic,
    clippy::cargo,
    clippy::use_debug,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    clippy::expect_used,
    clippy::unneeded_field_pattern,
    clippy::unnecessary_self_imports,
    clippy::str_to_string,
    clippy::string_to_string,
    clippy::string_slice,
    missing_docs,
    clippy::missing_docs_in_private_items,
    rustdoc::all,
    clippy::float_cmp_const,
    clippy::lossy_float_literal
)]
#![allow(
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::unreadable_literal
)]

#[cfg(not(feature = "_internal"))]
mod internal;

#[cfg(feature = "_internal")]
pub mod internal;

use quantette::QuantizeOutput;

#[cfg(feature = "image")]
use image::RgbImage;
use palette::{Oklab, Srgb};

// Re-export third-party crates whose types are part of our public API
#[cfg(feature = "image")]
pub use image;
pub use palette;

// We have tight integration/control over `quantette`, let's re-export the types directly.
pub use quantette::{AboveMaxLen, ColorSlice, PaletteSize};

/// A builder struct to specify options for palette generation.
///
/// # Examples
/// ```no_run
/// # use okolors::Okolors;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let img = image::open("some image")?.into_rgb8();
/// let palette = Okolors::try_from(&img)?
///     .palette_size(16)
///     .lightness_weight(0.5)
///     .sampling_factor(0.25)
///     .seed(42)
///     .srgb8_palette();
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug, Clone)]
#[must_use]
pub struct Okolors<'a> {
    /// The colors to create a palette from.
    colors: ColorSlice<'a, Srgb<u8>>,
    /// The amount to scale down the lightness component by.
    lightness_weight: f32,
    /// The number of colors to have in the palette.
    palette_size: PaletteSize,
    /// The percentage of the unique colors to sample.
    sampling_factor: f32,
    /// Return the palette sorted by increasing frequency.
    sort_by_frequency: bool,
    /// The batch size for parallel k-means.
    #[cfg(feature = "threads")]
    batch_size: u32,
    /// Whether or not to use parallelism.
    #[cfg(feature = "threads")]
    parallel: bool,
    /// The seed value for the random number generator.
    seed: u64,
}

impl<'a> From<ColorSlice<'a, Srgb<u8>>> for Okolors<'a> {
    fn from(slice: ColorSlice<'a, Srgb<u8>>) -> Self {
        Self::new(slice)
    }
}

impl<'a> TryFrom<&'a [Srgb<u8>]> for Okolors<'a> {
    type Error = AboveMaxLen<u32>;

    fn try_from(slice: &'a [Srgb<u8>]) -> Result<Self, Self::Error> {
        Ok(Self::new(slice.try_into()?))
    }
}

#[cfg(feature = "image")]
impl<'a> TryFrom<&'a RgbImage> for Okolors<'a> {
    type Error = AboveMaxLen<u32>;

    fn try_from(image: &'a RgbImage) -> Result<Self, Self::Error> {
        Ok(Self::new(image.try_into()?))
    }
}

impl<'a> Okolors<'a> {
    /// Creates a new [`Okolors`] with default options.
    ///
    /// See [`ColorSlice`] for examples on how to create it.
    ///
    /// Alternatively, use `Okolors::try_from` or
    /// `try_into` on an [`RgbImage`] or slice of [`Srgb<u8>`].
    pub fn new(colors: ColorSlice<'a, Srgb<u8>>) -> Self {
        Self {
            colors,
            lightness_weight: 0.325,
            palette_size: 8.into(),
            sampling_factor: 0.5,
            sort_by_frequency: false,
            #[cfg(feature = "threads")]
            batch_size: 4096,
            #[cfg(feature = "threads")]
            parallel: false,
            seed: 0,
        }
    }

    /// Sets the lightness weight used to scale down the lightness component of the colors.
    ///
    /// The brightness of colors has more influence on the perceived difference between colors.
    /// So, the generated the palette may contain colors that differ mainly in brightness only.
    /// The lightness weight is used scale down the lightness component of the colors,
    /// potentially bringing out more distinct hues in the final color palette.
    /// One downside to this is that colors near white and black may be merged into a shade of gray.
    ///
    /// The provided `lightness_weight` should be in the range `0.0..=1.0`,
    /// and it is clamped to this range otherwise.
    ///
    /// The default lightness weight is `0.325`.
    pub fn lightness_weight(mut self, lightness_weight: f32) -> Self {
        self.lightness_weight = lightness_weight.clamp(f32::EPSILON, 1.0);
        self
    }

    /// Sets the palette size which determines the (maximum) number of colors to have in the palette.
    ///
    /// The default palette size is `8`.
    pub fn palette_size(mut self, palette_size: impl Into<PaletteSize>) -> Self {
        self.palette_size = palette_size.into();
        self
    }

    /// Sets the sampling factor which controls what percentage of the unique colors to sample.
    ///
    /// Higher sampling factors take longer but give more accurate results.
    /// Sampling factors can be above `1.0`, but this may not give noticeably better results.
    /// Negative, NAN, or zero sampling factors will skip k-means optimization.
    ///
    /// The default sampling factor is `0.5`, that is, to sample half of the colors.
    pub fn sampling_factor(mut self, sampling_factor: f32) -> Self {
        self.sampling_factor = sampling_factor;
        self
    }

    /// Sort the returned palette by ascending frequency.
    ///
    /// Frequency refers to the number of pixels in the image that are most similar to the palette color.
    /// I.e., the number of pixels assigned to the palette color.
    ///
    /// By default, the palette is not sorted.
    pub fn sort_by_frequency(mut self, sort: bool) -> Self {
        self.sort_by_frequency = sort;
        self
    }

    /// Sets the seed value for the random number generator.
    ///
    /// The default seed is `0`.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Computes the [`Oklab`] quatization output.
    fn oklab_quantize_result(&self) -> QuantizeOutput<Oklab> {
        let Self {
            lightness_weight,
            colors,
            palette_size,
            seed,
            sampling_factor,
            #[cfg(feature = "threads")]
            batch_size,
            #[cfg(feature = "threads")]
            parallel,
            ..
        } = *self;

        #[cfg(feature = "threads")]
        if parallel {
            let unique = internal::unique_oklab_counts_par(colors, lightness_weight);
            let result = internal::wu_palette_par(&unique, palette_size, lightness_weight);
            let samples = internal::num_samples(&unique, sampling_factor);

            return if samples < batch_size {
                result
            } else {
                internal::kmeans_palette_par(&unique, samples, batch_size, result.palette, seed)
            };
        }

        let unique = internal::unique_oklab_counts(colors, lightness_weight);
        let result = internal::wu_palette(&unique, palette_size, lightness_weight);
        let samples = internal::num_samples(&unique, sampling_factor);

        if samples == 0 {
            result
        } else {
            internal::kmeans_palette(&unique, samples, result.palette, seed)
        }
    }

    /// Computes the [`Oklab`] color palette.
    #[must_use]
    pub fn oklab_palette(self) -> Vec<Oklab> {
        let result = self.oklab_quantize_result();

        let mut palette = if self.sort_by_frequency {
            internal::sort_by_frequency(result)
        } else {
            result.palette
        };

        internal::restore_lightness(&mut palette, self.lightness_weight);

        palette
    }

    /// Computes the [`Srgb<u8>`] color palette.
    #[must_use]
    pub fn srgb8_palette(self) -> Vec<Srgb<u8>> {
        internal::oklab_to_srgb(self.oklab_palette())
    }

    /// Computes the [`Srgb`] color palette.
    #[must_use]
    pub fn srgb_palette(self) -> Vec<Srgb> {
        internal::oklab_to_srgb(self.oklab_palette())
    }
}

#[cfg(feature = "threads")]
impl<'a> Okolors<'a> {
    /// Sets the batch size which determines the number of samples to group together in k-means.
    ///
    /// Increasing the batch size reduces the running time but with dimishing returns.
    /// Smaller batch sizes are more accurate but slower to run.
    ///
    /// The default batch size is `4096`.
    pub fn batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets whether or not to use multiple threads to compute the palette.
    ///
    /// The number of threads can be configured using a `rayon` thread pool.
    ///
    /// By default, single-threaded execution is used.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use palette::{cast::ComponentsInto, Srgb};

    #[rustfmt::skip]
    fn test_colors() -> Vec<Srgb<u8>> {
        vec![
            92, 88, 169, 165, 149, 73, 71, 40, 98, 83, 27, 26, 60, 128, 246, 159, 239, 169, 96, 30, 166, 176, 222, 97, 90, 70, 180, 179, 50, 228, 181, 40, 254, 145, 9, 78, 245, 233, 56, 84, 53, 206, 200, 144, 18, 254, 153, 172, 223, 72, 106, 137, 14, 80, 239, 226, 123, 194, 101, 45, 76, 50, 123, 191, 174, 48, 111, 113, 179, 128, 130, 102, 126, 243, 217, 64, 200, 191, 229, 251, 214, 70, 3, 67, 144, 244, 134, 135, 56, 56, 32, 221, 192, 216, 13, 56, 44, 181, 97, 110, 206, 127, 119, 110, 175, 195, 190, 120, 38, 123, 177, 226, 54, 223, 196, 60, 106, 167, 18, 123, 227, 127, 16, 204, 120, 122, 75, 30, 230, 142, 31, 217, 182, 59, 187, 239, 108, 11, 85, 49, 145, 24, 23, 185, 43, 69, 179, 66, 140, 107, 226, 19, 91, 101, 220, 155, 253, 116, 238, 117, 110, 200, 0, 193, 15, 153, 4, 67, 15, 187, 210, 42, 179, 90, 84, 90, 172, 128, 20, 92, 6, 170, 137, 172, 90, 43, 22, 234, 31, 212, 91, 47, 185, 27, 142, 187, 223, 24, 113, 208, 177, 70, 85, 100, 120, 77, 38, 204, 203, 121, 212, 36, 92, 244, 70, 138, 212, 17, 75, 240, 39, 75, 186, 221, 115, 240, 170, 63, 74, 224, 227, 120, 211, 88, 232, 2, 147, 156, 32, 28, 127, 238, 114, 31, 29, 255, 173, 73, 182, 101, 9, 111, 42, 139, 70, 15, 233, 172, 133, 223, 175, 178, 90, 98, 195, 53, 125, 208, 3, 253, 237, 181, 133, 4, 199, 253, 221, 58, 124, 99, 126, 239, 253, 151, 224, 24, 73, 6, 86, 161, 76, 151, 255, 10, 23, 65, 32, 210, 248, 81, 38, 48, 9, 108, 199, 14, 138, 170, 35, 55, 108, 6, 88, 188, 118, 195, 26, 8, 130, 175, 179, 227, 196, 145, 207, 102, 19, 216, 220, 45, 202, 154, 21, 156, 170, 212, 94, 215, 116, 193, 90, 180, 171, 47, 21, 77, 23, 123, 44, 78, 143, 42, 2, 74, 255, 133, 118, 33, 155, 111, 202, 64, 45, 63, 63, 214, 96, 139, 135, 219, 245, 179, 192, 127, 60, 35, 241, 104, 248, 230, 223, 6, 62, 218, 80, 205, 120, 184, 35, 154, 32, 105, 114, 8, 41, 111, 62, 6, 25, 95, 92, 58, 93, 240, 248, 169, 200, 218, 52, 158, 14, 14, 55, 250, 127, 136, 121, 102, 27, 182, 227, 205, 181, 76, 120, 153, 207, 142, 125, 169, 141, 124, 248, 169, 158, 190, 9, 3, 167, 93, 100, 181, 73, 103, 242, 153, 223, 185, 64, 151, 120, 186, 169, 4, 95, 137, 88, 82, 119, 205, 53, 97, 74, 55, 163, 232, 225, 60, 124, 248, 135, 218, 11, 99, 85, 108, 37, 178, 126, 222, 178, 80, 48, 13, 176, 179, 111, 243, 161, 245, 57, 252, 241, 40, 10, 190, 17, 17, 249, 108, 253, 70, 237, 119, 123, 113, 103, 9, 45, 60, 51, 160, 48, 191, 237, 109, 32, 124, 31, 32, 145, 68, 99, 149, 15, 91, 186, 160, 185, 96, 217, 110, 125, 187, 124, 68, 118, 32, 105, 9, 88, 87, 75, 48, 142, 194, 190, 54, 40, 20, 194, 46, 29, 75, 253, 48, 216, 170, 184, 118, 169, 51, 41, 244, 153, 147, 59, 133, 237, 163, 69, 114, 5, 61, 88, 110, 154, 83, 77, 107, 102, 152, 56, 33, 46, 158, 66, 129, 180, 36, 189, 131, 94, 233, 145, 10, 17, 168, 110, 47, 85, 84, 124, 120, 116, 168, 215, 178, 188, 159, 198, 245, 139, 42, 255, 56, 252, 77, 191, 14, 217, 108, 180, 44, 112, 245, 122, 189, 242, 41, 78, 218, 155, 241, 116, 244, 82, 192, 128, 86, 138, 175, 245, 51, 210, 2, 123, 114, 95, 113, 201, 86, 112, 68, 23, 18, 31, 146, 47, 222, 247, 251, 120, 59, 243, 239, 201, 115, 166, 126, 189, 227, 238, 204, 127, 11, 5, 52, 196, 63, 46, 149, 184, 150, 122, 143, 215, 1, 115, 164, 99, 238, 166, 1, 21, 132, 88, 117, 104, 88, 216, 106, 132, 233, 31, 41, 160, 153, 89, 180, 66, 151, 201, 50, 164, 208, 15, 160, 43, 43, 205, 204, 148, 178, 102, 188, 72, 46, 251, 40, 137, 184, 252, 241, 224, 101, 17, 77, 157, 83, 96, 93, 211, 83, 209, 73, 112, 195, 74, 91, 54, 41, 168, 129, 87, 81, 149, 63, 173, 37, 36, 112, 184, 36, 28, 8, 129, 153, 124, 27, 134, 50, 134, 24, 205, 118, 68, 60, 59, 214, 39, 78, 18, 243, 225, 206, 19, 125, 90, 210, 225, 249, 254, 210, 125, 34, 224, 203, 42, 126, 3, 126, 107, 10, 121, 113, 207, 234, 248, 44, 31, 158, 223, 128, 47, 147, 61, 0, 63, 44, 84, 252, 39, 69, 75, 190, 129, 116, 40, 198, 230, 137, 53, 130, 106, 68, 194, 233, 58, 197, 160, 130, 205, 169, 243, 118, 175, 252, 67, 251, 81, 20, 108, 22, 247, 161, 8, 50, 37, 224, 251, 154, 197, 172, 93, 113, 46, 206, 148, 47, 119, 102, 140, 82, 128, 115, 144, 250, 77, 110, 152, 182, 160, 44, 131, 202, 202, 130, 6, 187, 152, 19, 208, 179, 185, 78, 9, 124, 55, 17, 76, 70, 38, 94, 105, 152, 240, 85, 248, 246, 66, 39, 226, 25, 112, 7, 112, 123, 107, 179, 237, 82, 212, 236, 167, 236, 231, 103, 123, 133, 224, 169, 146, 235, 91, 49, 231, 69, 160, 6, 254, 137, 35, 207, 148, 223, 206, 250, 143, 189, 47, 132, 122, 163, 94, 39, 176, 108, 66, 163, 206, 247, 201, 87, 229, 10, 124, 164, 235, 6, 76, 188, 117, 127, 67, 217, 124, 24, 184, 114, 206, 145, 197, 108, 219, 3, 225, 89, 154, 25, 112, 11, 56, 213, 39, 21, 231, 5, 180, 33, 52, 155, 249, 221, 100, 39, 79, 150, 55, 196, 150, 1, 135, 185, 227, 26, 218, 153, 80, 17, 167, 90, 227, 172, 229, 129, 113, 31, 200, 121, 23, 223, 155, 73, 242, 66, 79, 220, 29, 78, 179, 185, 170, 111, 87, 245, 196, 208, 94, 108, 25, 31, 92, 212, 68, 252, 164, 6, 63, 214, 206, 226, 6, 172, 175, 170, 141, 236, 152, 139, 48, 178, 195, 137, 95, 47, 204, 47, 38, 78, 203, 24, 240, 73, 83, 191, 145, 29, 20, 161, 27, 142, 205, 65, 205, 169, 138, 148, 45, 124, 59, 114, 108, 6, 70, 100, 229, 220, 191, 68, 105, 3, 159, 13, 13, 195, 103, 203, 41, 242, 137, 153, 202, 121, 135, 252, 167, 154, 57, 234, 75, 163, 219, 195, 72, 100, 5, 67, 223, 207, 195, 0, 189, 48, 63, 101, 91, 80, 37, 151, 18, 5, 41, 109, 77, 222, 164, 85, 53, 189, 164, 66, 217, 195, 183, 41, 220, 251, 1, 76, 133, 72, 114, 24, 110, 73, 23, 105, 220, 61, 248, 148, 52, 130, 134, 84, 252, 181, 80, 180, 152, 166, 116, 248, 23, 16, 51, 227, 195, 249, 178, 163, 178, 70, 226, 79, 113, 6, 61, 122, 56, 99, 40, 22, 37, 58, 58, 150, 13, 63, 35, 59, 115, 139, 195, 222, 162, 160, 186, 57, 118, 5, 104, 79, 235, 174, 84, 123, 79, 221, 25, 149, 110, 116, 16, 215, 43, 153, 87, 86, 20, 174, 42, 238, 248, 66, 23, 25, 31, 112, 17, 83, 14, 112, 24, 37, 126, 66, 64, 6, 47, 207, 32, 184, 1, 237, 52, 79, 135, 204, 219, 180, 208, 18, 106, 87, 70, 50, 20, 168, 205, 89, 245, 38, 207, 136, 192, 179, 142, 82, 248, 42, 104, 39, 155, 214, 253, 34, 69, 14, 200, 1, 244, 210, 59, 193, 90, 161, 137, 48, 46, 172, 160, 75, 175, 194, 212, 76, 215, 236, 98, 144, 255, 206, 143, 141, 53, 238, 64, 6, 22, 164, 17, 244, 88, 73, 47, 197, 130, 24, 13, 234, 207, 117, 122, 244, 175, 250, 238, 244, 149, 206, 63, 112, 202, 102, 201, 224, 100, 79, 195, 126, 120, 237, 55, 67, 90, 246, 239, 248, 222, 137, 216, 194, 107, 213, 8, 182, 98, 69, 216, 200, 218, 252, 135, 201, 253, 194, 17, 70, 89, 250, 185, 50, 72, 127, 224, 225, 112, 67, 82, 20, 196, 114, 118, 85, 86, 119, 71, 31, 220, 197, 92, 5, 246, 208, 110, 141, 182, 174, 119, 56, 74, 238, 234, 216, 7, 11, 251, 158, 0, 43, 192, 4, 227, 62, 135, 161, 164, 221, 82, 99, 89, 216, 194, 79, 236, 14, 127, 255, 191, 142, 143, 229, 23, 206, 20, 61, 73, 165, 47, 235, 205, 47, 215, 247, 106, 239, 172, 116, 150, 140, 191, 53, 146, 68, 114, 173, 62, 204, 232, 115, 130, 35, 154, 22, 248, 180, 85, 178, 151, 236, 28, 46, 23, 212, 219, 82, 214, 239, 34, 102, 18, 58, 153, 241, 28, 133, 58, 226, 176, 255, 177, 208, 21, 118, 200, 119, 38, 14, 164, 74, 174, 112, 32, 191
        ].components_into()
    }

    #[test]
    fn zero_lightness_weight() {
        let colors = test_colors();
        let palette = Okolors::try_from(colors.as_slice())
            .unwrap()
            .lightness_weight(0.0)
            .oklab_palette();

        assert!(!palette.into_iter().any(|oklab| oklab.l.is_nan()));
    }

    #[test]
    fn not_enough_colors() {
        let colors = test_colors();
        let k = 100;
        let palette = Okolors::try_from(&colors[..k])
            .unwrap()
            .palette_size(PaletteSize::MAX)
            .oklab_palette();

        assert!(palette.len() <= k);

        #[cfg(feature = "threads")]
        {
            let palette = Okolors::try_from(&colors[..k])
                .unwrap()
                .palette_size(PaletteSize::MAX)
                .batch_size(64)
                .parallel(true)
                .oklab_palette();

            assert!(palette.len() <= k);
        }
    }

    #[test]
    fn no_samples() {
        let colors = test_colors();
        let palette = Okolors::try_from(colors.as_slice())
            .unwrap()
            .sampling_factor(0.0)
            .oklab_palette();

        assert_eq!(palette.len(), 8);

        #[cfg(feature = "threads")]
        {
            let palette = Okolors::try_from(colors.as_slice())
                .unwrap()
                .sampling_factor(0.0)
                .batch_size(64)
                .parallel(true)
                .oklab_palette();

            assert_eq!(palette.len(), 8);
        }
    }

    #[test]
    #[cfg(feature = "threads")]
    fn zero_batch_size() {
        let colors = test_colors();
        let palette = Okolors::try_from(colors.as_slice())
            .unwrap()
            .batch_size(0)
            .parallel(true)
            .oklab_palette();

        assert_eq!(palette.len(), 8);
    }
}
