use criterion::{
	black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
	SamplingMode,
};
use image::GenericImageView;
use okolors::OklabCounts;
use std::{path::PathBuf, time::Duration};

#[path = "../util/util.rs"]
mod util;

fn move_up(path: &str) -> PathBuf {
	["..", path].iter().collect()
}

fn create_group<'a>(c: &'a mut Criterion, name: &'a str) -> BenchmarkGroup<'a, WallTime> {
	let mut group = c.benchmark_group(name);
	group
		.sample_size(30)
		.noise_threshold(0.05)
		.sampling_mode(SamplingMode::Flat)
		.warm_up_time(Duration::from_millis(500));
	group
}

fn preprocessing(c: &mut Criterion) {
	let mut group = create_group(c, "preprocessing");

	for (path, image) in util::load_image_dir(&move_up(util::UNSPLASH_DIR)) {
		for (width, height) in [(480, 270), (1920, 1080), image.dimensions()] {
			let image = image.thumbnail(width, height);
			group.bench_with_input(
				BenchmarkId::new(&path, format!("{width}x{height}")),
				&image,
				|b, image| {
					b.iter(|| {
						OklabCounts::try_from_image(image, black_box(u8::MAX))
							.expect("non-gigantic image")
							.with_lightness_weight(black_box(0.325))
					});
				},
			);
		}
	}
}

fn kmeans(c: &mut Criterion) {
	let mut group = create_group(c, "kmeans");

	let counts = util::to_oklab_counts(util::load_image_dir(move_up(util::UNSPLASH_DIR)));

	fn bench(
		name: &str,
		group: &mut BenchmarkGroup<WallTime>,
		counts: &[(String, OklabCounts)],
		k: u8,
		convergence: f32,
	) {
		for (path, counts) in counts {
			group.bench_with_input(BenchmarkId::new(name, path), &counts, |b, counts| {
				b.iter(|| {
					okolors::run(
						counts,
						black_box(1),
						black_box(k),
						black_box(convergence),
						black_box(1024),
						black_box(0),
					)
				});
			});
		}
	}

	group.measurement_time(Duration::from_secs(2));
	bench("default", &mut group, &counts, 8, 0.05);
	bench("low k", &mut group, &counts, 4, 0.05);
	bench("high convergence", &mut group, &counts, 8, 0.1);

	group.measurement_time(Duration::from_secs(4));
	bench("high k", &mut group, &counts, 32, 0.05);
	bench("low convergence", &mut group, &counts, 8, 0.01);
}

fn all_steps(name: &str, image_dir: &str, c: &mut Criterion) {
	let mut group = create_group(c, name);
	group.measurement_time(Duration::from_secs(8));

	for (path, image) in util::load_image_dir(&move_up(image_dir)) {
		group.bench_with_input(BenchmarkId::from_parameter(path), &image, |b, image| {
			b.iter(|| {
				okolors::run(
					&okolors::OklabCounts::try_from_image(image, black_box(u8::MAX))
						.expect("non-gigantic image")
						.with_lightness_weight(black_box(0.325)),
					black_box(1),
					black_box(8),
					black_box(0.05),
					black_box(1024),
					black_box(0),
				)
			});
		});
	}
}

// fn cq100(c: &mut Criterion) {
// 	all_steps("cq100", util::CQ100_DIR, c)
// }

fn unsplash(c: &mut Criterion) {
	all_steps("unsplash", util::UNSPLASH_DIR, c)
}

criterion_group!(benches, preprocessing, kmeans, unsplash);
criterion_main!(benches);
