use criterion::{
	black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
	SamplingMode,
};
use image::GenericImageView;
use okolors::OklabCounts;
use std::time::Duration;
use wgpu::{Device, Queue};

fn load_images() -> Vec<(String, image::DynamicImage)> {
	std::fs::read_dir("../img")
		.expect("read img directory")
		.collect::<Result<Vec<_>, _>>()
		.expect("read each file")
		.iter()
		.filter_map(|file| {
			let path = file.path();
			if path.extension().map_or(false, |ext| ext == "jpg") {
				Some(
					image::open(&path)
						.map(|image| (path.file_name().unwrap().to_owned().into_string().unwrap(), image)),
				)
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

	let counts = load_images()
		.into_iter()
		.map(|(path, image)| {
			(
				path,
				OklabCounts::try_from_image(&image, u8::MAX)
					.expect("non-gigantic image")
					.with_lightness_weight(black_box(0.325)),
			)
		})
		.collect::<Vec<_>>();

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

/// Attempts to get a gpu [`Device`] and its associated [`Queue`]
async fn get_device() -> Option<(Device, Queue)> {
	let instance = wgpu::Instance::default();

	let adapter = instance
		.request_adapter(&wgpu::RequestAdapterOptions::default())
		.await?;

	adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				label: None,
				features: wgpu::Features::empty(),
				limits: wgpu::Limits::downlevel_defaults(),
			},
			None,
		)
		.await
		.ok()
}

fn kmeans_gpu(c: &mut Criterion) {
	let (device, queue) = pollster::block_on(get_device()).unwrap();

	let mut group = create_group(c, "kmeans-gpu");

	let counts = load_images()
		.into_iter()
		.map(|(path, image)| {
			(
				path,
				okolors::OklabCounts::try_from_image(&image, u8::MAX)
					.expect("non-gigantic image")
					.with_lightness_weight(black_box(0.325)),
			)
		})
		.collect::<Vec<_>>();

	fn bench(
		name: &str,
		group: &mut BenchmarkGroup<WallTime>,
		counts: &[(String, okolors::OklabCounts)],
		k: u8,
		convergence: f32,
		device: &Device,
		queue: &Queue,
	) {
		for (path, counts) in counts {
			group.bench_with_input(BenchmarkId::new(name, path), &counts, |b, counts| {
				b.iter(|| {
					okolors::gpu::run(
						counts,
						black_box(1),
						black_box(k),
						black_box(convergence),
						black_box(1024),
						black_box(0),
						device,
						queue,
					)
				});
			});
		}
	}

	group.measurement_time(Duration::from_secs(2));
	bench("default", &mut group, &counts, 8, 0.05, &device, &queue);
	bench("low k", &mut group, &counts, 4, 0.05, &device, &queue);
	bench("high convergence", &mut group, &counts, 8, 0.1, &device, &queue);

	group.measurement_time(Duration::from_secs(4));
	bench("high k", &mut group, &counts, 32, 0.05, &device, &queue);
	bench("low convergence", &mut group, &counts, 8, 0.01, &device, &queue);
}

fn all_steps(c: &mut Criterion) {
	let mut group = create_group(c, "all_steps");
	group.measurement_time(Duration::from_secs(8));

	let images = load_images();
	for (path, image) in &images {
		group.bench_with_input(BenchmarkId::from_parameter(path), image, |b, image| {
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

criterion_group!(benches, preprocessing, kmeans, all_steps, kmeans_gpu);
criterion_main!(benches);
