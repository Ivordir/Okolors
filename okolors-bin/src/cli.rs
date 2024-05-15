//! Specifies the CLI and handles arg parsing

use std::{ops::RangeInclusive, path::PathBuf};

use clap::{Parser, ValueEnum};
use palette::Okhsl;
use quantette::{AboveMaxLen, PaletteSize};

/// Supported output formats for the final colors
#[derive(Copy, Clone, ValueEnum)]
pub enum Format {
    /// sRGB hexcode
    Hex,
    /// sRGB (r,g,b) triple
    Rgb,
    /// Whitespace with true color background
    Swatch,
}

/// Sort orders for the final colors
#[derive(Copy, Clone, ValueEnum)]
pub enum Sort {
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
pub enum Colorize {
    /// Foreground
    Fg,
    /// Background
    Bg,
}

/// Generate a color palette from an image by performing k-means clustering in the Oklab color space.
#[derive(Parser)]
#[command(name = "okolors", version)]
pub struct Options {
    /// The path to the input image
    pub image: PathBuf,

    /// The output format to print the colors in
    #[arg(short = 'o', long, default_value = "hex")]
    pub format: Format,

    /// Color the foreground or background for each printed color
    #[arg(short, long)]
    pub colorize: Option<Colorize>,

    /// The order to print the colors in
    ///
    /// The h, s, and l options below refer to Okhsl component values and not the HSL color space.
    #[arg(short, long, default_value = "n")]
    pub sort: Sort,

    /// Reverse the printed order of the colors
    #[arg(short, long)]
    pub reverse: bool,

    /// A comma separated list of additional lightness levels that each color should be printed in
    ///
    /// Lightness refers to Okhsl lightness with values in the range [0.0, 100.0].
    /// A separate line is used for printing the colors at each lightness level.
    #[arg(short, long, value_delimiter = ',', value_parser = parse_lightness)]
    pub lightness_levels: Vec<f32>,

    /// Do not print each color with its average lightness
    ///
    /// This is useful if you only care about colors resulting from the --lightness-levels option.
    #[arg(long)]
    pub no_avg_lightness: bool,

    /// The value used to scale down the influence of the lightness component on color difference
    ///
    /// The brightness of colors has more influence on the perceived difference between colors.
    /// So, the generated the palette may contain colors that differ mainly in brightness only.
    /// The lightness weight is used scale down the lightness component of the colors,
    /// potentially bringing out more distinct hues in the final color palette.
    /// One downside to this is that colors near white and black may be merged into a shade of gray.
    /// Provided values should be in the range [0.0, 1.0].
    #[arg(short = 'w', long, default_value_t = 0.325, value_parser = parse_lightness_weight)]
    pub lightness_weight: f32,

    /// The (maximum) number of colors to put in the palette
    ///
    /// The provided value should be in the range [0, 256].
    #[arg(short, default_value_t = 8.into(), value_parser = parse_palette_size)]
    pub k: PaletteSize,

    /// The number of samples to make, expressed as a percentage of
    /// the number of unique colors in the image
    ///
    /// Higher sampling factors take longer but give more accurate results.
    /// The sampling factor can be above `1.0`, but this may not give noticeably better results.
    /// The provided sampling factor should be non-negative.
    #[arg(short = 'f', long, default_value_t = 0.5, value_parser = parse_sampling_factor)]
    pub sampling_factor: f32,

    /// The maximum image size, in number of pixels, before a thumbnail is created
    ///
    /// Unfortunately, this option may reduce the color accuracy,
    /// as multiple pixels in the original image are interpolated to form a pixel in the thumbnail.
    /// This option is intended for reducing the time needed for large images,
    /// but it can also be used to provide fast, inaccurate results for any image.
    #[arg(short = 'p', long, default_value_t = u32::MAX)]
    pub max_pixels: u32,

    /// The number of samples to batch together in k-means
    ///
    /// Increasing the batch size reduces the running time but with dimishing returns.
    /// Smaller batch sizes are more accurate but slower to run.
    #[cfg(feature = "threads")]
    #[arg(long, default_value_t = 4096)]
    pub batch_size: u32,

    /// The seed value used for the random number generator
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// The number of threads to use
    ///
    /// A value of 0 indicates to automatically choose the number of threads.
    #[cfg(feature = "threads")]
    #[arg(short, long, default_value_t = 0)]
    pub threads: u8,

    /// Print additional information and the elapsed time for various steps
    #[arg(long)]
    pub verbose: bool,
}

/// Parse the palette size and ensure it is <= `MAX_COLORS`
fn parse_palette_size(s: &str) -> Result<PaletteSize, String> {
    let value: u16 = s.parse().map_err(|e| format!("{e}"))?;
    value
        .try_into()
        .map_err(|AboveMaxLen(max)| format!("not in the range [0, {max}]"))
}

/// Parse a float value and ensure it in the provided range
fn parse_float_in_range(s: &str, range: RangeInclusive<f32>) -> Result<f32, String> {
    let value = s.parse().map_err(|e| format!("{e}"))?;
    if range.contains(&value) {
        Ok(value)
    } else {
        Err(format!(
            "not in the range [{}, {}]",
            range.start(),
            range.end()
        ))
    }
}

/// Parse the sampling factor and ensure it is >= `0.0`
fn parse_sampling_factor(s: &str) -> Result<f32, String> {
    parse_float_in_range(s, 0.0..=f32::INFINITY)
}

/// The factor used to scale the lightness values for a better interface,
/// as Okhsl lightness values are in the range `0.0..=1.0`
pub const LIGHTNESS_SCALE: f32 = 100.0;

/// Parse a lightness value and ensure it is in `0.0..=100.0`
fn parse_lightness(s: &str) -> Result<f32, String> {
    let min = LIGHTNESS_SCALE * Okhsl::<f32>::min_lightness();
    let max = LIGHTNESS_SCALE * Okhsl::<f32>::max_lightness();
    parse_float_in_range(s, min..=max)
}

/// Parse the lightness weight and ensure it is in `0.0..=1.0`
fn parse_lightness_weight(s: &str) -> Result<f32, String> {
    parse_float_in_range(s, 0.0..=1.0).map(|v| f32::max(v, f32::EPSILON))
}
