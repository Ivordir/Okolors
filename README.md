# Okolors

Still a WIP, but...

Okolors takes an image and produces a color palette consisting of the image's average colors.
It does this by converting the image's pixels to the [Oklab](https://bottosson.github.io/posts/oklab/) color space
and then performing [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering).
By using a proper color space for color difference and a more accurate clustering algorithm,
this helps to ensure that the generated palette is truly representative of the input image.

One of the main intended use cases for Okolors is to generate colors for a theme based off a wallpaper.
In line with this goal, Okolors also supports printing the final average colors in multiple Okhsl lightness levels.
For example, you can specify a low lightness level for background colors
and a high lightness for foreground text in order to achieve a certain contrast ratio.
The [Okhsl](https://bottosson.github.io/posts/colorpicker/) color space is ideal for this,
because as the lightness is changed, Okhsl preserves the hue and saturation of the color
(better than other color spaces like HSL). You can see some of the [examples](#examples) below.

## Notes

Okolors uses the [`image`](https://github.com/image-rs/image) crate to load images
and so should work with any image format supported by it.
Although, currently only png, jpeg, and avif have been (somewhat) tested.
Also, due an [issue](https://github.com/image-rs/image/issues/1647) with avif support in the image crate,
avif support is currently handled using `aom`.
So, loading avif images requires having `aom` on your system
(e.g., [Arch package](https://archlinux.org/packages/extra/x86_64/aom/)).

# Examples

Let's use the following photo for the examples below.

![Jewel Changi Airport Waterfall](docs/Jewel%20Changi.jpg)

Running Okolors for this image with the default options gives the following sRGB hex values.

```bash
> okolors 'img/Jewel Changi.jpg'
020707 0A3F48 E8E8E9 352620 90ADBD CC876C 896D94 7B4638
```

If your terminal supports true color,
then you can use `-o swatch` to see blocks of the output colors.

```bash
> okolors 'img/Jewel Changi.jpg' -o swatch
```

![](docs/swatch1.svg)

We can increase the color accuracy by increasing the number of trials, `-n`, and lowering the convergence threshold, `-e`.

```bash
> okolors 'img/Jewel Changi.jpg' -n 4 -e 0.01 -o swatch
```

![](docs/swatch2.svg)

Let's get these colors in additional lightness levels using `-l`.

```bash
> okolors 'img/Jewel Changi.jpg' -l 10,30,50,70 -n 4 -e 0.01 -o swatch
```

![](docs/swatch3.svg)

If we're providing our own lightness levels, maybe we want to cluster the colors by hue and saturation only.
Let's set the lightness weight to `0` using `-w`.

```bash
> okolors 'img/Jewel Changi.jpg' -w 0 -l 10,30,50,70 -n 4 -e 0.01 -o swatch
```

![](docs/swatch4.svg)

That ended up bringing out an additional pinkish color but also merged white and black into a dark gray.
So, use this at your own discretion!

It seems that two of the colors are quite similiar. Let's reduce the number of colors, `k`, by 1.

```bash
> okolors 'img/Jewel Changi.jpg' -k 7 -w 0 -l 10,30,50,70 -n 4 -e 0.01 -o swatch
```

![](docs/swatch5.svg)

# Performance: ~~blazingly~~ smolderingly fast

Despite using k-means which is more accurate but slower than something like median cut quantization,
Okolors still seems to be decently fast. For example, for a 1920x1080 jpeg image on my hardware,
Okolors takes about 100-200ms to complete using the default options.
(15-50ms of this time is simply loading the image from disk and decoding it.)
The plan is to reduce this time further by implementing multi-threading
and/or possibly SIMD once [portable-simd](https://github.com/rust-lang/rust/issues/86656) becomes stable.

(Benchmarks will hopefully come soon as well!)

# References

- [kmeans-colors](https://github.com/okaneco/kmeans-colors/) served as the basis for Okolors.
  If you want to perform other k-means related operations on images or prefer the CIELAB colorspace, then check it out!
- The work by [Dr. Greg Hamerly and others](https://cs.baylor.edu/~hamerly/software/kmeans)
  was a very helpful resource for the k-means implementation in Okolors.
- The awesome [palette](https://github.com/Ogeon/palette) library is used for all color conversions.
