use clap::Parser;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;

#[path = "../util/util.rs"]
mod util;

#[derive(Parser)]
struct Options {
	#[arg(long, default_value_t = 32)]
	seeds: u32,

	#[arg(short, long, default_value = "8,16,32", value_delimiter = ',')]
	k: Vec<u8>,

	#[arg(short = 'n', long, default_value_t = 1)]
	trials: u32,

	#[arg(short = 'e', long, default_value_t = 0.01)]
	convergence: f32,

	#[arg(short = 'i', long, default_value_t = 1024)]
	max_iter: u32,

	images: Vec<PathBuf>,
}

fn main() {
	let options = Options::parse();

	let images = util::to_oklab_counts(if options.images.is_empty() {
		util::load_image_dir(util::CQ100_DIR)
	} else {
		util::load_images(&options.images)
	});

	let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
	let seeds = (0..options.seeds).map(|_| rng.gen()).collect::<Vec<_>>();

	// use char count as supplement for grapheme count
	let max_name_len = images.iter().map(|(name, _)| name.chars().count()).max().unwrap_or(0);

	const NUM_COL_WIDTH: usize = 8;

	println!(
		"{:width$} {}",
		"image",
		options
			.k
			.iter()
			.map(|k| format!("{k:<0$}", NUM_COL_WIDTH))
			.collect::<Vec<_>>()
			.join(" "),
		width = max_name_len,
	);

	// MSE is very low for oklab colors since all components are usually < 1.0
	const MSE_SCALE: f64 = 10e3;

	for (path, oklab) in images {
		let avg_mse_by_k = options
			.k
			.iter()
			.map(|&k| {
				let mse = seeds.iter().map(|&seed| {
					MSE_SCALE
						* f64::from(
							okolors::run(&oklab, options.trials, k, options.convergence, options.max_iter, seed).mse,
						)
				});

				format!(
					"{:>width$.2}",
					mse.sum::<f64>() / f64::from(options.seeds),
					width = NUM_COL_WIDTH
				)
			})
			.collect::<Vec<_>>();

		println!("{:width$} {}", path, avg_mse_by_k.join(" "), width = max_name_len);
	}
}
