use std::fmt::Display;

use itertools::iproduct;
use okolors::{KmeansResult, OklabCounts};
use palette::Oklab;

fn load_images() -> Vec<image::DynamicImage> {
	std::fs::read_dir("img")
		.expect("read img directory")
		.collect::<Result<Vec<_>, _>>()
		.expect("read each file")
		.iter()
		.filter_map(|file| {
			let path = file.path();
			if path.extension().map_or(false, |ext| ext == "jpg") {
				Some(image::open(&path))
			} else {
				None
			}
		})
		.collect::<Result<Vec<_>, _>>()
		.expect("loaded each image")
}

fn color_difference(a: &Oklab, b: &Oklab) -> f32 {
	let dl = a.l - b.l;
	let da = a.a - b.a;
	let db = a.b - b.b;
	dl * dl + da * da + db * db
}

fn centroids_difference(prev: &KmeansResult, result: &KmeansResult) -> f32 {
	assert_eq!(prev.centroids.len(), result.centroids.len());
	// Some colors/centroids may be present in the prev KmeansResult but not in the current result (and vice versa).
	// Instead of using, e.g., the Hungarian algorithm for minimum weight matching
	// we use an O(n^3) approximation? algorithm where we continuously match each pair of closest centroids.

	let mut prev_centroids = prev.centroids.clone();
	let mut curr_centroids = result.centroids.clone();
	let mut total_dist = 0.0;

	while !prev_centroids.is_empty() {
		let mut min_i = 0;
		let mut min_j = 0;
		let mut min_dist = f32::INFINITY;

		for (i, prev) in prev_centroids.iter().enumerate() {
			for (j, curr) in curr_centroids.iter().enumerate() {
				let dist = color_difference(prev, curr);
				if dist < min_dist {
					min_dist = dist;
					min_i = i;
					min_j = j;
				}
			}
		}

		prev_centroids.swap_remove(min_i);
		curr_centroids.swap_remove(min_j);
		total_dist += min_dist;
	}

	total_dist
}

fn get_results<P: Copy, T: Copy>(
	param_combos: &(impl Iterator<Item = P> + Clone),
	test_param: T,
	run_kmeans: impl Fn(P, T) -> KmeansResult,
) -> Vec<KmeansResult> {
	param_combos
		.clone()
		.map(|params| run_kmeans(params, test_param))
		.collect::<Vec<_>>()
}

fn print_results<P: Copy, T: Display + Copy>(
	testing_param_name: &str,
	testing_params: &[T],
	param_combos: impl Iterator<Item = P> + Clone,
	run_kmeans: impl Fn(P, T) -> KmeansResult,
) {
	println!("{testing_param_name}: max_iter avg_centroid_diff avg_variance_perc_diff");

	let mut last_result = get_results(&param_combos, testing_params[0], &run_kmeans);
	println!(
		"{}: {}",
		testing_params[0],
		last_result.iter().map(|result| result.iterations).max().unwrap(),
	);

	let n_combos = last_result.len() as f64;

	for &test_param in testing_params[1..].iter() {
		let result = get_results(&param_combos, test_param, &run_kmeans);

		let total_centroid_diff = last_result
			.iter()
			.zip(&result)
			.map(|(prev, curr)| f64::from(centroids_difference(prev, curr)))
			.sum::<f64>();

		let avg_centroid_diff = total_centroid_diff / n_combos;

		let total_variance_perc_diff = last_result
			.iter()
			.zip(&result)
			.map(|(prev, curr)| (prev.variance - curr.variance) / prev.variance)
			.sum::<f64>();

		let avg_variance_perc_diff = total_variance_perc_diff / n_combos;

		let max_iter = result.iter().map(|result| result.iterations).max().unwrap();

		println!("{test_param}: {max_iter} {avg_centroid_diff} {avg_variance_perc_diff}");

		last_result = result;
	}
}

use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, ValueEnum)]
enum Parameter {
	Trials,
	Convergence,
}

#[derive(Parser)]
struct Options {
	parameter: Parameter,
}

fn main() {
	let options = Options::parse();

	let images = load_images()
		.iter()
		.flat_map(|image| {
			[(480, 270), (1920, 1080)].into_iter().map(|(width, height)| {
				OklabCounts::try_from_image(&image.thumbnail(width, height), u8::MAX).expect("non-gigantic image")
			})
		})
		.collect::<Vec<_>>();

	let trials = [1, 4, 8];
	let k = [4, 8, 32];
	let convergence = [0.1, 0.05, 0.01];
	let seed = [0, 42, 123456789];

	match options.parameter {
		Parameter::Trials => {
			// The avg_centroid_diff and avg_variance_perc_diff seem to decrease exponentially as trials increase (R^2 ~= 0.93).
			// 1-3 or 1-5 trials give the the most "bang for your buck", as trials past this give ever more dimishing returns.
			print_results(
				"trials",
				&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
				iproduct!(&images, k, convergence, seed),
				|(counts, k, convergence, seed), trial| okolors::run(counts, trial, k, convergence, 1024, seed),
			)
		},
		Parameter::Convergence => {
			// This option seems to depend on k, so maybe Okolors should treat this as an average?
			// But for k around 4-10, convergence values lower than 0.01 do not make sense as the avg_centroid_diff
			// is much lower than than the "just noticeable difference".
			print_results(
				"convergence",
				&[0.1, 0.05, 0.01, 0.005, 0.001],
				iproduct!(&images, trials, [4_u8, 6, 8, 10], seed),
				|(counts, trial, k, seed), convergence| okolors::run(counts, trial, k, convergence, 1024, seed),
			)
		},
	}
}
