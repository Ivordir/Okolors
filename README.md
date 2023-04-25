# Ok Palette

Still a WIP, but...

Ok Palette takes an image and produces a color palette consisting of the image's average colors.
It does this by converting the image's pixels to the [Oklab](https://bottosson.github.io/posts/oklab/) color space
and then performing k-means clustering.
By using a proper color space for color difference and a more accurate clustering algorithm,
this helps to ensure that the generated palette is truly representative of the input image.

One of the main intended use cases for Ok Palette is to generate colors for a theme based off a wallpaper.
In line with this goal, Ok Palette also supports printing the final average colors in multiple Okhsl lightness levels.
For example, you can specify a low lightness level for background colors
and a high lightness for foreground text in order to achieve a certain contrast ratio.
The [Okhsl](https://bottosson.github.io/posts/colorpicker/) color space is ideal for this,
because as the lightness is changed, Okhsl preserves the hue and saturation of the color
(better than other color spaces like HSL). You can see some of the [examples]() below.

## Notes

Ok Palette uses the [`image`](https://github.com/image-rs/image) crate to load images
and so should work with any image format supported by it.
Although, currently only png, jpeg, and avif have been (somewhat) tested.
Also, due an [issue](https://github.com/image-rs/image/issues/1647) with avif support in the image crate,
avif support is currently handled using `aom`.
So, loading avif images requires having `aom` on your system
(e.g., [Arch package](https://archlinux.org/packages/extra/x86_64/aom/)).

## Performance: ~~blazingly~~ smolderingly fast

Despite using k-means which is more accurate but slower than something like median cut quantization,
Ok Palette still seems to be decently fast. For example, for a 1920x1080 jpeg image on my hardware,
Ok Palette takes about 100-200ms to complete using the default options.
(15-50ms of this time is simply loading the image from disk and decoding it.)
The plan is to reduce this time further by implementing multi-threading
and/or possibly SIMD once [portable-simd](https://github.com/rust-lang/rust/issues/86656) becomes stable.

(Benchmarks will hopefully come soon as well!)

# Examples

TODO!

# References

- [kmeans-colors](https://github.com/okaneco/kmeans-colors/) served as the basis for Ok Palette.
  If you want to perform other k-means related operations on images or prefer the CIELAB colorspace, then check it out!
- The work by [Dr. Greg Hamerly and others](https://cs.baylor.edu/~hamerly/software/kmeans)
  was a very helpful resource for the k-means implementation in Ok Palette.
- The awesome [palette](https://github.com/Ogeon/palette) library is used for all color conversions.
