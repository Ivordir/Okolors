use criterion::{
	black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
	SamplingMode,
};
use okolors::OklabCounts;
use std::time::Duration;

fn load_images() -> Vec<(String, image::DynamicImage)> {
	std::fs::read_dir("../img")
		.expect("read img directory")
		.collect::<Result<Vec<_>, _>>()
		.expect("read each file")
		.iter()
		.filter_map(|file| {
			let path = file.path();
			if path.extension().map_or(false, |ext| ext == "jpg") {
				Some(image::open(&path).map(|image| {
					(
						path.file_name().unwrap().to_owned().into_string().unwrap(),
						image.thumbnail(1920, 1080),
					)
				}))
			} else {
				None
			}
		})
		.collect::<Result<Vec<_>, _>>()
		.expect("loaded each image")
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

	for (path, image) in load_images() {
		let image = image.to_rgb8();
		group.bench_with_input(BenchmarkId::from_parameter(path), &image, |b, image| {
			b.iter(|| OklabCounts::from_rgbimage(image, 1.0));
		});
	}
}

fn kmeans(c: &mut Criterion) {
	let mut group = create_group(c, "kmeans");

	let images = load_images();
	let counts = images
		.iter()
		.map(|(path, image)| (path, OklabCounts::from_image(image, 1.0)))
		.collect::<Vec<_>>();

	fn bench(
		name: &str,
		group: &mut BenchmarkGroup<WallTime>,
		counts: &[(&String, OklabCounts)],
		k: u8,
		convergence: f32,
	) {
		for (path, counts) in counts {
			group.bench_with_input(BenchmarkId::new(name, path), &counts, |b, counts| {
				b.iter(|| {
					okolors::from_oklab_counts(
						counts,
						black_box(1),
						black_box(k),
						black_box(convergence),
						black_box(64),
						black_box(0),
					)
				});
			});
		}
	}

	group.measurement_time(Duration::from_secs(3));
	bench("default", &mut group, &counts, 8, 0.05);

	group.measurement_time(Duration::from_secs(2));
	bench("low k", &mut group, &counts, 4, 0.05);
	bench("high convergence", &mut group, &counts, 8, 0.1);

	group.measurement_time(Duration::from_secs(8));
	bench("high k", &mut group, &counts, 32, 0.05);
	bench("low convergence", &mut group, &counts, 8, 0.01);
}

criterion_group!(benches, preprocessing, kmeans);
criterion_main!(benches);
