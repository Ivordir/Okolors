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

Let's use the following photo for the examples below.

![Jewel Changi Aiport Waterfall](doc/Jewel%20Changi.jpg)

Running Ok Palette for this image with the default options gives the following sRGB hex values:

```bash
> ok-palette 'img/Jewel Changi.jpg'
010303 091E22 104852 E7E7E9 BEA8A9 527A8F A76965 583830
```

If your terminal supports true color,
then you can use `-o swatch` to see blocks of the output colors.

```bash
> ok-palette 'img/Jewel Changi.jpg' -o swatch
```

<svg width="auto" height="3em" viewBox="0 0 8 1">
  <rect id="swatch" width="1" height="1"/>
  <use href="#swatch" x="0" fill="rgb(1,3,3)"/>
  <use href="#swatch" x="1" fill="rgb(9,30,34)"/>
  <use href="#swatch" x="2" fill="rgb(16,72,82)"/>
  <use href="#swatch" x="3" fill="rgb(231,231,233)"/>
  <use href="#swatch" x="4" fill="rgb(190,168,169)"/>
  <use href="#swatch" x="5" fill="rgb(82,122,143)"/>
  <use href="#swatch" x="6" fill="rgb(167,105,101)"/>
  <use href="#swatch" x="7" fill="rgb(88,56,48)"/>
</svg>

We can increase the color accuracy by increasing the number of trials, `-t`, and lowering the convergence threshold, `-e`.

```bash
> ok-palette 'img/Jewel Changi.jpg' -t 4 -e 0.01 -o swatch
```

<svg width="auto" height="3em" viewBox="0 0 8 1">
  <rect id="swatch" width="1" height="1"/>
  <use href="#swatch" x="0" fill="rgb(1,2,2)"/>
  <use href="#swatch" x="1" fill="rgb(5,17,18)"/>
  <use href="#swatch" x="2" fill="rgb(16,44,48)"/>
  <use href="#swatch" x="3" fill="rgb(31,76,85)"/>
  <use href="#swatch" x="4" fill="rgb(231,232,233)"/>
  <use href="#swatch" x="5" fill="rgb(105,109,122)"/>
  <use href="#swatch" x="6" fill="rgb(202,134,116)"/>
  <use href="#swatch" x="7" fill="rgb(158,176,196)"/>
</svg>

Oh no, that's too accurate!
The image is made up of mostly black, white, and blue-green,
so the other colors are hardly coming through anymore.

We can try increasing k, but now we have a lot of colors, and some are very similar.

```bash
> ok-palette 'img/Jewel Changi.jpg' -k 12 -t 4 -e 0.01 -o swatch
```

<svg width="auto" height="3em" viewBox="0 0 12 1">
  <rect id="swatch" width="1" height="1"/>
  <use href="#swatch" x="0" fill="rgb(1,1,1)"/>
  <use href="#swatch" x="1" fill="rgb(3,10,11)"/>
  <use href="#swatch" x="2" fill="rgb(7,30,33)"/>
  <use href="#swatch" x="3" fill="rgb(9,56,63)"/>
  <use href="#swatch" x="4" fill="rgb(233,233,234)"/>
  <use href="#swatch" x="5" fill="rgb(22,86,98)"/>
  <use href="#swatch" x="6" fill="rgb(94,128,150)"/>
  <use href="#swatch" x="7" fill="rgb(172,189,206)"/>
  <use href="#swatch" x="8" fill="rgb(67,41,33)"/>
  <use href="#swatch" x="9" fill="rgb(175,110,106)"/>
  <use href="#swatch" x="10" fill="rgb(117,78,80)"/>
  <use href="#swatch" x="10" fill="rgb(216,151,128)"/>
</svg>

Alternatively, let's ignore color lightness to merge similar colors together, thereby bringing out other colors in the process:

```bash
> ok-palette 'img/Jewel Changi.jpg' --ignore-lightness -t 4 -e 0.01 -o swatch
```

<svg width="auto" height="3em" viewBox="0 0 8 1">
  <rect id="swatch" width="1" height="1"/>
  <use href="#swatch" x="0" fill="rgb(41,40,39)"/>
  <use href="#swatch" x="1" fill="rgb(8,18,19)"/>
  <use href="#swatch" x="2" fill="rgb(11,42,48)"/>
  <use href="#swatch" x="3" fill="rgb(15,81,92)"/>
  <use href="#swatch" x="4" fill="rgb(86,54,40)"/>
  <use href="#swatch" x="5" fill="rgb(187,116,78)"/>
  <use href="#swatch" x="6" fill="rgb(124,113,152)"/>
  <use href="#swatch" x="7" fill="rgb(164,103,121)"/>
</svg>


That's better! Unfortunately, now white and black have been merged into a dark gray. We can, however, specify additional lightness levels to print the colors in with `-l`.

```bash
> ok-palette 'img/Jewel Changi.jpg' -l 10,30,50,70 --ignore-lightness -t 4 -e 0.01 -o swatch
```

<svg width="auto" height="15em" viewBox="0 0 8 5">
  <rect id="swatch" width="1" height="1"/>
  <use href="#swatch" x="0" fill="rgb(41,40,39)"/>
  <use href="#swatch" x="1" fill="rgb(8,18,19)"/>
  <use href="#swatch" x="2" fill="rgb(11,42,48)"/>
  <use href="#swatch" x="3" fill="rgb(15,81,92)"/>
  <use href="#swatch" x="4" fill="rgb(86,54,40)"/>
  <use href="#swatch" x="5" fill="rgb(187,116,78)"/>
  <use href="#swatch" x="6" fill="rgb(124,113,152)"/>
  <use href="#swatch" x="7" fill="rgb(164,103,121)"/>
  <use href="#swatch" x="0" y="1" fill="rgb(23,22,21)"/>
  <use href="#swatch" x="1" y="1" fill="rgb(12,25,26)"/>
  <use href="#swatch" x="2" y="1" fill="rgb(5,26,30)"/>
  <use href="#swatch" x="3" y="1" fill="rgb(2,26,31)"/>
  <use href="#swatch" x="4" y="1" fill="rgb(33,18,11)"/>
  <use href="#swatch" x="5" y="1" fill="rgb(36,17,7)"/>
  <use href="#swatch" x="6" y="1" fill="rgb(23,20,33)"/>
  <use href="#swatch" x="7" y="1" fill="rgb(34,16,21)"/>
  <use href="#swatch" x="0" y="2" fill="rgb(71,69,68)"/>
  <use href="#swatch" x="1" y="2" fill="rgb(47,76,78)"/>
  <use href="#swatch" x="2" y="2" fill="rgb(27,78,87)"/>
  <use href="#swatch" x="3" y="2" fill="rgb(14,79,89)"/>
  <use href="#swatch" x="4" y="2" fill="rgb(95,60,45)"/>
  <use href="#swatch" x="5" y="2" fill="rgb(100,57,34)"/>
  <use href="#swatch" x="6" y="2" fill="rgb(72,64,93)"/>
  <use href="#swatch" x="7" y="2" fill="rgb(97,55,68)"/>
  <use href="#swatch" x="0" y="3" fill="rgb(121,119,117)"/>
  <use href="#swatch" x="1" y="3" fill="rgb(85,128,131)"/>
  <use href="#swatch" x="2" y="3" fill="rgb(52,131,145)"/>
  <use href="#swatch" x="3" y="3" fill="rgb(31,133,150)"/>
  <use href="#swatch" x="4" y="3" fill="rgb(156,105,83)"/>
  <use href="#swatch" x="5" y="3" fill="rgb(166,100,65)"/>
  <use href="#swatch" x="6" y="3" fill="rgb(123,112,151)"/>
  <use href="#swatch" x="7" y="3" fill="rgb(158,99,116)"/>
  <use href="#swatch" x="0" y="4" fill="rgb(173,171,169)"/>
  <use href="#swatch" x="1" y="4" fill="rgb(133,182,185)"/>
  <use href="#swatch" x="2" y="4" fill="rgb(89,188,206)"/>
  <use href="#swatch" x="3" y="4" fill="rgb(54,191,213)"/>
  <use href="#swatch" x="4" y="4" fill="rgb(209,158,137)"/>
  <use href="#swatch" x="5" y="4" fill="rgb(222,153,117)"/>
  <use href="#swatch" x="6" y="4" fill="rgb(174,165,199)"/>
  <use href="#swatch" x="7" y="4" fill="rgb(208,154,169)"/>
</svg>

# References

- [kmeans-colors](https://github.com/okaneco/kmeans-colors/) served as the basis for Ok Palette.
  If you want to perform other k-means related operations on images or prefer the CIELAB colorspace, then check it out!
- The work by [Dr. Greg Hamerly and others](https://cs.baylor.edu/~hamerly/software/kmeans)
  was a very helpful resource for the k-means implementation in Ok Palette.
- The awesome [palette](https://github.com/Ogeon/palette) library is used for all color conversions.
