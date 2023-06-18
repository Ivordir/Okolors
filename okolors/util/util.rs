#![allow(dead_code)]

use okolors::OklabCounts;
use std::path::{Path, PathBuf};

pub const CQ100_DIR: &str = "img/CQ100/img";
pub const UNSPLASH_DIR: &str = "img/unsplash/img";

pub fn load_image_dir(dir: impl AsRef<Path>) -> Vec<(String, image::DynamicImage)> {
	let mut paths = std::fs::read_dir(dir)
		.expect("read img directory")
		.collect::<Result<Vec<_>, _>>()
		.expect("read each file")
		.iter()
		.map(std::fs::DirEntry::path)
		.collect::<Vec<_>>();

	paths.sort();

	load_images(&paths)
}

pub fn load_images(images: &[PathBuf]) -> Vec<(String, image::DynamicImage)> {
	images
		.iter()
		.map(|path| image::open(path).map(|image| (path.file_name().unwrap().to_owned().into_string().unwrap(), image)))
		.collect::<Result<Vec<_>, _>>()
		.expect("loaded each image")
}

pub fn to_oklab_counts(images: Vec<(String, image::DynamicImage)>) -> Vec<(String, OklabCounts)> {
	images
		.into_iter()
		.map(|(path, image)| {
			(
				path,
				OklabCounts::try_from_image(&image, u8::MAX)
					.expect("non-gigantic image")
					.with_lightness_weight(0.325),
			)
		})
		.collect::<Vec<_>>()
}
