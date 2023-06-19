//! Provides the implementation for k-means on the GPU
//!

// temporary
#![allow(clippy::too_many_arguments)]

use crate::{
	kmeans::{self, ChromaHueDistance, ColorDifference, EuclideanDistance},
	KmeansResult, OklabCounts, OklabN,
};
use bytemuck::Zeroable;
use palette::Oklab;
use rand::{Rng, SeedableRng};
use std::{borrow::Cow, mem::size_of};
use wgpu::{util::DeviceExt, BindGroup, Buffer, ComputePipeline, Device, Queue};

/// Number of innovations per workgroup for (most) shaders
const WORKGROUP_SIZE: u32 = 64;

/// ceil div
fn div_ceil(dividend: u32, divisor: u32) -> u32 {
	(dividend + divisor - 1) / divisor
}

/// Global constants provided to the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
	/// The number of cluters/centroids
	k: u32,
	/// The number of data points
	n: u32,
}

/// Global state held by the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct State {
	/// The current centroid being reduced
	curr_k: u32,
	/// The number of elements to process in the last workgroup
	n_remainder: u32,
	/// The index of the next random centroid to use if a centroid ends up having a count of 0
	curr_random: u32,
}

/// Bookkeeping for each k-means data point
struct PointData {
	/// Weight of each data point used to randomly select starting centroids in k-means++
	weight: Vec<f32>,
}

impl PointData {
	/// Create a [`PointData`] with the given number data points
	fn new(n: u32) -> Self {
		let n = n as usize;
		Self { weight: vec![f32::INFINITY; n] }
	}

	/// Reset data for the next k-means trial
	fn reset(&mut self) {
		self.weight.fill(f32::INFINITY);
	}
}

/// Data for each center/centroid
struct CenterData {
	/// The current centroid point
	centroid: Vec<OklabN>,
	/// The centroid buffer used to copy over results from gpu
	buffer: Vec<OklabN>,
	/// Vector sum for all data points in this center
	sum: Vec<Oklab<f64>>,
}

impl CenterData {
	/// Create a [`CenterData`] with the given number of centers
	fn new(k: u8) -> Self {
		let k = usize::from(k);
		Self {
			centroid: Vec::new(),
			buffer: vec![OklabN::zeroed(); k],
			sum: vec![Oklab { l: 0.0, a: 0.0, b: 0.0 }; k],
		}
	}

	/// Copies new centroids over from gpu and swaps `centroid` and `buffer`,
	/// returning the sum of the distances moved by each centroid.
	fn copy_buffer_slice<D: ColorDifference>(&mut self, slice: &[OklabN]) -> f32 {
		self.buffer.copy_from_slice(slice);
		std::mem::swap(&mut self.buffer, &mut self.centroid);
		self.centroid
			.iter()
			.zip(&self.buffer)
			.map(|(new, old)| D::squared_distance(old.to_lab(), new.to_lab()).sqrt())
			.sum::<f32>()
	}

	/// Reset data for the next k-means trial
	fn reset(&mut self) {
		self.centroid.clear();
		self.sum.fill(Oklab { l: 0.0, a: 0.0, b: 0.0 });
	}
}

/// Data and state for the GPU (buffers, pipelines, etc.)
#[allow(clippy::missing_docs_in_private_items)]
struct GPUState<'a> {
	k: u8,
	n: u32,

	device: &'a Device,
	queue: &'a Queue,

	update_assignments: ComputePipeline,
	update_centroids_first_pass: ComputePipeline,
	update_centroids: ComputePipeline,

	update_assignments_group: BindGroup,
	update_centroids_first_pass_group: BindGroup,
	update_centroids_group: BindGroup,

	centroids: Buffer,
	result: Buffer,
}

impl<'a> GPUState<'a> {
	/// Initialize a new [`GPUState`]
	#[allow(clippy::too_many_lines)]
	fn new(
		device: &'a Device,
		queue: &'a Queue,
		k: u8,
		n: u32,
		oklab: &[OklabN],
		centroids: &[OklabN],
		rng: &mut impl Rng,
	) -> Self {
		let n_size = n as usize;

		let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: None,
			source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/kmeans.wgsl"))),
		});

		let update_assignments = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: None,
			layout: None,
			module: &cs_module,
			entry_point: "lloyd_update_assignments",
		});

		let update_centroids_first_pass = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: None,
			layout: None,
			module: &cs_module,
			entry_point: "lloyd_update_centroids_first_pass",
		});

		let update_centroids = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: None,
			layout: None,
			module: &cs_module,
			entry_point: "lloyd_update_centroids",
		});

		let globals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("globals"),
			contents: bytemuck::cast_slice(&[Globals { k: u32::from(k), n }]),
			usage: wgpu::BufferUsages::UNIFORM,
		});

		let oklab = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("oklab"),
			contents: bytemuck::cast_slice(oklab),
			usage: wgpu::BufferUsages::STORAGE,
		});

		let random_centroids = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("oklab"),
			contents: bytemuck::cast_slice(&(0..k).map(|_| kmeans::random_centroid(rng)).collect::<Vec<_>>()),
			usage: wgpu::BufferUsages::UNIFORM,
		});

		let centroids = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("centroids"),
			contents: bytemuck::cast_slice(centroids),
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
		});

		let sums = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("sum"),
			size: (n_size * size_of::<OklabN>()) as u64,
			usage: wgpu::BufferUsages::STORAGE,
			mapped_at_creation: false,
		});

		let assignment = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("assignment"),
			size: (n_size * size_of::<u32>()) as u64,
			usage: wgpu::BufferUsages::STORAGE,
			mapped_at_creation: false,
		});

		let state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("state"),
			contents: bytemuck::cast_slice(&[State {
				curr_k: 0,
				n_remainder: n % WORKGROUP_SIZE,
				curr_random: 0,
			}]),
			usage: wgpu::BufferUsages::STORAGE,
		});

		let result = device.create_buffer(&wgpu::BufferDescriptor {
			label: None,
			size: (usize::from(k) * size_of::<OklabN>()) as u64,
			usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let update_assignments_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &update_assignments.get_bind_group_layout(0),
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: globals.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: oklab.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 3,
					resource: centroids.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 5,
					resource: assignment.as_entire_binding(),
				},
			],
		});

		let update_centroids_first_pass_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &update_centroids_first_pass.get_bind_group_layout(0),
			entries: &[
				wgpu::BindGroupEntry {
					binding: 1,
					resource: oklab.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 4,
					resource: sums.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 5,
					resource: assignment.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 6,
					resource: state.as_entire_binding(),
				},
			],
		});

		let update_centroids_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &update_centroids.get_bind_group_layout(0),
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: globals.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: random_centroids.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 3,
					resource: centroids.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 4,
					resource: sums.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 6,
					resource: state.as_entire_binding(),
				},
			],
		});

		Self {
			k,
			n,

			device,
			queue,

			update_assignments,
			update_centroids_first_pass,
			update_centroids,

			update_assignments_group,
			update_centroids_first_pass_group,
			update_centroids_group,

			centroids,
			result,
		}
	}
}

/// Holds all the state used by k-means
struct KmeansState {
	/// Data for each center
	centers: CenterData,
	/// Data for each point
	points: PointData,
}

impl KmeansState {
	/// Initialize a new [`KmeansState`] with `k` centers and `n` data points
	fn new(k: u8, n: u32) -> Self {
		Self {
			centers: CenterData::new(k),
			points: PointData::new(n),
		}
	}
}

/// Run a trial of sort k-means
fn kmeans<D: ColorDifference>(
	oklab: &OklabCounts,
	KmeansState { centers, points }: &mut KmeansState,
	k: u8,
	max_iter: u32,
	convergence: f32,
	seed: u64,
	device: &Device,
	queue: &Queue,
) -> KmeansResult {
	let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
	kmeans::kmeans_plus_plus::<D>(k, &mut rng, oklab, &mut centers.centroid, &mut points.weight);

	// TODO: move this to `run_trial` so that it can be reused each trial
	let mut gpu = GPUState::new(
		device,
		queue,
		k,
		oklab.num_colors(),
		&oklab.color_counts,
		&centers.centroid,
		&mut rng,
	);

	let mut iterations = 0;
	let mut total_delta = f32::INFINITY;
	while iterations < max_iter && total_delta > convergence {
		total_delta = pollster::block_on(run_gpu::<D>(&mut gpu, centers));
		iterations += 1;
	}

	// let variance = oklab
	// 	.data
	// 	.iter()
	// 	.zip(&points.assignment)
	// 	.map(|(color, &center)| {
	// 		f64::from(color.n)
	// 			* f64::from(D::squared_distance(
	// 				color.to_lab(),
	// 				centers.centroid[usize::from(center)].to_lab(),
	// 			))
	// 	})
	// 	.sum();

	// Compute on gpu
	let variance = 10000.0;

	let (mut centroids, counts): (Vec<_>, Vec<_>) = centers
		.centroid
		.iter()
		.filter_map(|centroid| {
			if centroid.n == 0 {
				None
			} else {
				Some((centroid.to_lab(), centroid.n))
			}
		})
		.unzip();

	#[allow(clippy::float_cmp)]
	if oklab.lightness_weight != 0.0 && oklab.lightness_weight != 1.0 {
		for centroid in &mut centroids {
			centroid.l /= oklab.lightness_weight;
		}
	}

	centers.reset();
	points.reset();

	KmeansResult { variance, centroids, counts, iterations }
}

/// Run multiple trials of k-means, taking the trial with the lowest variance
fn run_trials<D: ColorDifference>(
	oklab: &OklabCounts,
	trials: u32,
	k: u8,
	max_iter: u32,
	convergence: f32,
	seed: u64,
	device: &Device,
	queue: &Queue,
) -> KmeansResult {
	let mut state = KmeansState::new(k, oklab.num_colors());

	(0..trials)
		.map(|i| {
			kmeans::<D>(
				oklab,
				&mut state,
				k,
				max_iter,
				convergence,
				seed ^ u64::from(i),
				device,
				queue,
			)
		})
		.min_by(|x, y| f64::total_cmp(&x.variance, &y.variance))
		.unwrap_or(KmeansResult::empty())
}

/// Run multiple trials of k-means, taking the trial with the lowest variance
///
/// An empty result with no centroids is returned if `oklab` is empty, `trials` = 0, or `k` = 0.
#[must_use]
pub fn run(
	oklab: &OklabCounts,
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
	device: &Device,
	queue: &Queue,
) -> KmeansResult {
	#[allow(clippy::float_cmp)]
	if k == 0 || oklab.color_counts.is_empty() {
		KmeansResult::empty()
	} else if oklab.lightness_weight == 0.0 {
		run_trials::<ChromaHueDistance>(oklab, trials, k, max_iter, convergence_threshold, seed, device, queue)
	} else {
		run_trials::<EuclideanDistance>(oklab, trials, k, max_iter, convergence_threshold, seed, device, queue)
	}
}

/// Dispatch an iteration of k-means to the GPU
async fn run_gpu<'a, D: ColorDifference>(gpu: &mut GPUState<'a>, centers: &mut CenterData) -> f32 {
	let mut encoder = gpu
		.device
		.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
	{
		let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

		cpass.set_pipeline(&gpu.update_assignments);
		cpass.set_bind_group(0, &gpu.update_assignments_group, &[]);
		cpass.insert_debug_marker("update assignments");

		let mut num_workgroups = gpu.n;
		loop {
			num_workgroups = div_ceil(num_workgroups, WORKGROUP_SIZE);
			cpass.dispatch_workgroups(num_workgroups, 1, 1);
			if num_workgroups <= 1 {
				break;
			}
		}

		for _ in 0..gpu.k {
			let mut num_workgroups = div_ceil(gpu.n, WORKGROUP_SIZE);

			cpass.set_pipeline(&gpu.update_centroids_first_pass);
			cpass.set_bind_group(0, &gpu.update_centroids_first_pass_group, &[]);
			cpass.insert_debug_marker("update centroids first pass");
			cpass.dispatch_workgroups(num_workgroups, 1, 1);

			cpass.set_pipeline(&gpu.update_centroids);
			cpass.set_bind_group(0, &gpu.update_centroids_group, &[]);
			cpass.insert_debug_marker("update centroids");
			while num_workgroups > 1 {
				num_workgroups = div_ceil(num_workgroups, WORKGROUP_SIZE);
				cpass.dispatch_workgroups(num_workgroups, 1, 1);
			}
		}
	}

	encoder.copy_buffer_to_buffer(&gpu.centroids, 0, &gpu.result, 0, gpu.result.size());

	gpu.queue.submit(Some(encoder.finish()));
	let result_slice = gpu.result.slice(..);
	let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
	result_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).expect("openshot channel"));

	gpu.device.poll(wgpu::Maintain::Wait);

	if let Some(Ok(())) = receiver.receive().await {
		let data = result_slice.get_mapped_range();
		let delta = centers.copy_buffer_slice::<D>(bytemuck::cast_slice(&data));

		drop(data);
		gpu.result.unmap();

		delta
	} else {
		panic!("failed to run compute on gpu!")
	}
}
