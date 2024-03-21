# Changelog

# v0.7.0
- External crates that have types present in `okolors`'s public API are now reexported (`palette` and `image`).
- Types from `quantette` that are in the public API are also now reexported.
- Bumped `image` version to `0.25.0`.

## Breaking changes
Removed the `{color}_palette_par` functions in favor of the `Okolors::parallel` function.

# v0.6.0

- Added a `Okolors::sort_by_frequency` function which will sort the colors in the returned palette by ascending frequency (the number of pixels corresponding to the palette color).
- `Okolors` now implements `Debug` and `Clone`.
- `Okolors` builder now takes and returns `Self` instead of `&mut Self`.

# v0.5.1

Changelog starts here.
