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

Okolors supports jpeg, png, gif, and qoi images by default.

WebP support is not enabled by default, as it seems that the `image` crate still has lingering bugs for WebP in certain cases:
[1](https://github.com/image-rs/image/issues/1873),
[2](https://github.com/image-rs/image/issues/1872),
[3](https://github.com/image-rs/image/issues/1712),
[4](https://github.com/image-rs/image/issues/1647).
Panics and bugs resulting from this should be directed upstream.

Similarly, due an [issue](https://github.com/image-rs/image/issues/1647) with AVIF support in the `image` crate,
the avif feature for Okolors is not enabled by default and instead uses the `libavif-image` crate.
Compiling with this feature requires cmake and nasm on the system.

Other image formats not enabled by default include bmp and tiff.

# Examples

Let's use the following photo for the examples below.

![Jewel Changi Airport Waterfall](docs/Jewel%20Changi.jpg)

Running Okolors for this image with the default options gives the following sRGB hex values.

```bash
> okolors 'img/Jewel Changi.jpg'
020606 08373E E2E4E7 4F8091 4E2F26 CA835A 926F97 BB7B7B
```

If your terminal supports true color,
then you can use `-o swatch` to see blocks of the output colors.

```bash
> okolors 'img/Jewel Changi.jpg' -o swatch
```

![](docs/swatch1.svg)

We can increase the color accuracy by increasing the number of trials, `-n`, and lowering the convergence threshold, `-e`.

```bash
> okolors 'img/Jewel Changi.jpg' -n 3 -e 0.01 -o swatch
```

![](docs/swatch2.svg)

Let's get these colors in additional lightness levels using `-l`.

```bash
> okolors 'img/Jewel Changi.jpg' -l 10,30,50,70 -n 3 -e 0.01 -o swatch
```

![](docs/swatch3.svg)

If we're providing our own lightness levels, maybe we want to cluster the colors by hue and saturation only.
Let's set the lightness weight to `0` using `-w`.

```bash
> okolors 'img/Jewel Changi.jpg' -w 0 -l 10,30,50,70 -n 3 -e 0.01 -o swatch
```

![](docs/swatch4.svg)

That ended up bringing out an additional pinkish color but also merged white and black into a dark gray.
So, use this at your own discretion!

It seems that two of the colors are quite similar. Let's reduce the number of colors, `k`, by 1.

```bash
> okolors 'img/Jewel Changi.jpg' -k 7 -w 0 -l 10,30,50,70 -n 3 -e 0.01 -o swatch
```

![](docs/swatch5.svg)

# Performance

Despite using k-means which is more accurate but slower than something like median cut quantization,
Okolors still seems to be pretty fast. Excluding the time to read and decode the image from disk,
here is Okolors's running time for each test image as reported by [criteron](https://github.com/bheisler/criterion.rs). The parameters used were `k = 8`, `convergence_threshold = 0.05`, `trials = 1`, and `max_iter = 1024`. The benchmarks were run on a 4-core CPU, so YMMV on different hardware (especially with more or less cores).

| Image                    | Dimensions | Time (ms) |
|:------------------------ |:----------:| ---------:|
| Akihabara.jpg            | 5663x3669  |       376 |
| Bryggen.jpg              | 5508x3098  |        89 |
| Cesky Krumlov.jpg        | 4608x3456  |       177 |
| Hokkaido.jpg             | 6000x4000  |       186 |
| Jewel Changi.jpg         | 6000x4000  |       128 |
| Lake Atitlan.jpg         | 5112x3408  |       374 |
| Lake Mendota.jpg         | 3839x5758  |       124 |
| Louvre.jpg               | 6056x4000  |       172 |
| Sydney Sunset.jpg        | 2880x1508  |        22 |
| Termas Geometricas.jpg   | 5472x3648  |       171 |
| Yellow Crane Tower.jpg   | 3785x2839  |       121 |
| Yosemite Tunnel View.jpg | 5580x3720  |       176 |

The above times could be reduced by properly leveraging SIMD,
possibly once [portable-simd](https://github.com/rust-lang/rust/issues/86656) becomes stable.

# References

- [kmeans-colors](https://github.com/okaneco/kmeans-colors/) served as the basis for Okolors.
  If you want to perform other k-means related operations on images or prefer the CIELAB colorspace, then check it out!
- The work by [Dr. Greg Hamerly and others](https://cs.baylor.edu/~hamerly/software/kmeans)
  was a very helpful resource for the k-means implementation in Okolors.
- The awesome [palette](https://github.com/Ogeon/palette) library is used for all color conversions.
