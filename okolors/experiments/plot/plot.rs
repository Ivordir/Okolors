use std::path::PathBuf;

use clap::Parser;
use okolors::OklabCounts;
use palette::{FromColor, Oklab, Srgb};

#[derive(Parser)]
struct Options {
	image: PathBuf,

	#[arg(short = 'w', long, default_value_t = 1.0)]
	lightness_weight: f32,

	#[arg(short, long, default_value_t = 3)]
	trials: u32,

	#[arg(short = 'e', long, default_value_t = 0.01)]
	convergence_threshold: f32,

	#[arg(short, long, default_value_t = 8)]
	k: u8,

	#[arg(short, long)]
	proportional: bool,

	#[arg(long, default_value_t = 16)]
	min_count: u32,

	#[arg(long, default_value_t = 0.1)]
	point_size: f32,

	#[arg(long, default_value_t = 10.0)]
	centroid_size: f32,
}

fn main() {
	let options = Options::parse();

	let img = image::open(options.image).expect("opened image");

	let oklab = OklabCounts::from_image(&img, u8::MAX).with_lightness_weight(options.lightness_weight);

	let result = if options.centroid_size == 0.0 {
		okolors::KmeansResult::empty()
	} else {
		okolors::run(
			&oklab,
			options.trials,
			options.k,
			options.convergence_threshold,
			1024,
			0,
		)
	};

	let min_size = options.min_count.ilog2();

	println!("#Colors");
	println!("a b l n color");
	for (oklab, count) in oklab.pairs() {
		if count >= options.min_count {
			let count = if options.proportional {
				count.ilog2() + 1 - min_size
			} else {
				1
			};
			let size = options.point_size * count as f32;

			let srgb: Srgb<u8> = Srgb::from_color(Oklab {
				l: oklab.l / options.lightness_weight,
				..oklab
			})
			.into_format();

			println!("{} {} {} {} 0x{:X}", oklab.a, oklab.b, oklab.l, size, srgb);
		}
	}

	println!();
	println!();

	println!("#Centroids");
	println!("a b l n color");
	for &centroid in result.centroids.iter() {
		let srgb: Srgb<u8> = Srgb::from_color(centroid).into_format();
		println!(
			"{} {} {} {} 0x{:X}",
			centroid.a,
			centroid.b,
			centroid.l * options.lightness_weight,
			options.centroid_size,
			srgb
		);
	}
}
