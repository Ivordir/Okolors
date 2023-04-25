//! Provides the implementation for (sort) k-means

use crate::PixelDataVec;
use palette::Oklab;
use rand::{Rng, SeedableRng};
use soa_derive::{soa_zip, StructOfArray};

/// Color difference/distance in a uniform color space
trait ColorDifference {
	/// Squared color difference
	fn squared_distance(x: Oklab, y: Oklab) -> f32;
}

/// Regular Euclidean Distance
struct EuclideanDistance;

impl ColorDifference for EuclideanDistance {
	fn squared_distance(x: Oklab, y: Oklab) -> f32 {
		let dl = x.l - y.l;
		let da = x.a - y.a;
		let db = x.b - y.b;
		dl * dl + da * da + db * db
	}
}

/// Euclidean Distance ignoring the lightness component
struct ChromaHueDistance;

impl ColorDifference for ChromaHueDistance {
	fn squared_distance(x: Oklab, y: Oklab) -> f32 {
		let da = x.a - y.a;
		let db = x.b - y.b;
		da * da + db * db
	}
}

/// Bookkeeping for each k-means data point
#[derive(StructOfArray)]
struct PointData {
	/// Center assignment for this data point
	assignment: u8,
	/// Weight of each data point used to randomly select starting centroids in k-means++
	weight: f32,
}

impl PointDataVec {
	/// Initialize Vecs to the given number data points
	fn init(n: u32) -> Self {
		let n = n as usize;
		Self {
			assignment: vec![0; n],
			weight: vec![f32::MAX; n],
		}
	}

	/// Reset data for the next k-means trial
	fn reset(&mut self) {
		// assignments are corrected every iteration
		self.weight.fill(f32::MAX);
	}
}

/// Data for each center/centroid
#[derive(StructOfArray)]
struct CenterData {
	/// The centroid point
	centroid: Oklab,
	/// Distance that the centroid moved last iteration
	delta: f32,
	/// Vector sum for all data points in this center
	sum: Oklab<f64>,
	/// Number of points in this center
	count: u32,
}

impl CenterDataVec {
	/// Initialize Vecs to the given number of centers
	fn init(k: u8) -> Self {
		let k = usize::from(k);
		Self {
			centroid: Vec::new(),
			delta: vec![0.0; k],
			sum: vec![Oklab { l: 0.0, a: 0.0, b: 0.0 }; k],
			count: vec![0; k],
		}
	}

	/// Reset data for the next k-means trial
	fn reset(&mut self) {
		self.centroid.clear();
		// deltas are recomputed every iteration
		self.sum.fill(Oklab { l: 0.0, a: 0.0, b: 0.0 });
		self.count.fill(0);
	}
}

/// Holds all the state used by k-means
struct KmeansState {
	/// Data for each center
	centers: CenterDataVec,
	/// One fourth of the squared distance between each pairs of centers
	distances: Vec<(u8, f32)>,
	/// Data for each point
	points: PointDataVec,
}

impl KmeansState {
	/// Initialize a new `KmeansState` with `k` centers and `n` data points
	fn new(k: u8, n: u32) -> Self {
		Self {
			centers: CenterDataVec::init(k),
			distances: vec![(0, 0.0); usize::from(k) * usize::from(k)],
			points: PointDataVec::init(n),
		}
	}
}

/// Result from running k-means
pub struct KmeansResult {
	/// Variance achieved by these centroids
	pub variance: f64,
	/// Final centroids
	pub centroids: Vec<Oklab>,
	/// Number of points in each centroid
	pub counts: Vec<u32>,
	/// Number of elapsed iterations
	pub iterations: u32,
}

/// Choose the starting centroids using the k-means++ algorithm
fn kmeans_plus_plus<D: ColorDifference>(
	k: u8,
	mut rng: &mut impl Rng,
	data: &[Oklab],
	centroids: &mut Vec<Oklab>,
	weights: &mut [f32],
) {
	use rand::{
		distributions::{WeightedError::*, WeightedIndex},
		prelude::Distribution,
	};

	// Pick any random first centroid
	centroids.push(data[rng.gen_range(0..data.len())]);

	// Pick each next centroid with a weighted probability based off the squared distance to its closest centroid
	for i in 1..usize::from(k) {
		let centroid = centroids[i - 1];
		for (weight, &color) in weights.iter_mut().zip(data) {
			*weight = f32::min(*weight, D::squared_distance(color, centroid));
		}

		match WeightedIndex::new(weights.iter().copied()) {
			Ok(sampler) => centroids.push(data[sampler.sample(&mut rng)]),
			Err(AllWeightsZero) => return, // all points exactly match a centroid
			Err(InvalidWeight | NoItem | TooMany) => {
				unreachable!("distances are >= 0 and data.len() is in 1..=u32::MAX")
			},
		}
	}
}

/// For each pair of centers, update their distances and sort each center's row by increasing distance
// i and j are < centroids.len() <= u8::MAX
#[allow(clippy::cast_possible_truncation)]
fn update_distances<D: ColorDifference>(centroids: &[Oklab], distances: &mut [(u8, f32)]) {
	let k = centroids.len();
	for i in 0..k {
		let ci = centroids[i];
		distances[i * k + i] = (i as u8, 0.0);
		for j in (i + 1)..k {
			let cj = centroids[j];
			let dist = D::squared_distance(ci, cj) / 4.0;
			distances[j * k + i] = (i as u8, dist);
			distances[i * k + j] = (j as u8, dist);
		}
	}

	for i in 0..k {
		distances[(i * k)..((i + 1) * k)].sort_by(|(_, x), (_, y)| f32::total_cmp(x, y));
	}
}

/// For each data point, update its assigned center
fn update_assignments<D: ColorDifference>(
	data: &PixelDataVec,
	centers: &mut CenterDataVec,
	distances: &[(u8, f32)],
	points: &mut PointDataVec,
) {
	let k = centers.len();
	for (&color, &n, center) in soa_zip!(data, [color, count], &mut points.assignment) {
		let ci = usize::from(*center);
		let dist = D::squared_distance(color, centers.centroid[ci]);

		// Find the closest center
		let mut min_dist = dist;
		let mut min_center = *center;
		for &(other_center, half_dist) in &distances[(ci * k + 1)..((ci + 1) * k)] {
			if dist < half_dist {
				break;
			}

			let other_dist = D::squared_distance(color, centers.centroid[usize::from(other_center)]);
			if other_dist < min_dist {
				min_dist = other_dist;
				min_center = other_center;
			}
		}

		// Move this point to its new center
		if min_center != *center {
			let nf = f64::from(n);
			let l = nf * f64::from(color.l);
			let a = nf * f64::from(color.a);
			let b = nf * f64::from(color.b);

			let old_sum = &mut centers.sum[ci];
			old_sum.l -= l;
			old_sum.a -= a;
			old_sum.b -= b;
			centers.count[ci] -= n;

			let cj = usize::from(min_center);

			let new_sum = &mut centers.sum[cj];
			new_sum.l += l;
			new_sum.a += a;
			new_sum.b += b;
			centers.count[cj] += n;

			*center = min_center;
		}
	}
}

/// For each center, update its centroid using the vector sums and compute deltas
fn update_centroids<D: ColorDifference>(rng: &mut impl Rng, centers: &mut CenterDataVec) {
	for (centroid, delta, &n, sum) in soa_zip!(centers, [mut centroid, mut delta, count, sum]) {
		let new_centroid = if n == 0 {
			// Float literals below are the min and max values achievable when converting from Srgb colors
			Oklab {
				l: rng.gen_range(Oklab::min_l()..=Oklab::max_l()),
				a: rng.gen_range(-0.2338874..=0.2762164),
				b: rng.gen_range(-0.31152815..=0.19856972),
			}
		} else {
			let n = f64::from(n);
			// Sums may need greater precision, but the average can fall back down to a reduced precision
			#[allow(clippy::cast_possible_truncation)]
			Oklab {
				l: (sum.l / n) as f32,
				a: (sum.a / n) as f32,
				b: (sum.b / n) as f32,
			}
		};

		*delta = D::squared_distance(*centroid, new_centroid).sqrt();
		*centroid = new_centroid;
	}
}

/// Run a trial of sort k-means
fn kmeans<D: ColorDifference>(
	data: &PixelDataVec,
	KmeansState { centers, distances, points }: &mut KmeansState,
	k: u8,
	max_iter: u32,
	convergence: f32,
	seed: u64,
) -> KmeansResult {
	let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
	kmeans_plus_plus::<D>(k, &mut rng, &data.color, &mut centers.centroid, &mut points.weight);

	debug_assert!(u8::try_from(centers.len()).is_ok());

	// Compute initial vector sums and counts
	for (color, &n, &center) in soa_zip!(data, [color, count], &points.assignment) {
		let i = usize::from(center);
		let nf = f64::from(n);
		let sum = &mut centers.sum[i];
		sum.l += nf * f64::from(color.l);
		sum.a += nf * f64::from(color.a);
		sum.b += nf * f64::from(color.b);
		centers.count[i] += n;
	}

	let mut iterations = 0;
	loop {
		update_distances::<D>(&centers.centroid, distances);
		update_assignments::<D>(data, centers, distances, points);
		update_centroids::<D>(&mut rng, centers);

		iterations += 1;

		if iterations >= max_iter || centers.delta.iter().sum::<f32>() <= convergence {
			break;
		}
	}

	let variance = soa_zip!(data, [color, count], &points.assignment)
		.map(|(&color, &n, &center)| {
			f64::from(n) * f64::from(D::squared_distance(color, centers.centroid[usize::from(center)]))
		})
		.sum();

	let centroids = soa_zip!(&centers, [centroid, count])
		.filter_map(|(&color, &count)| if count == 0 { None } else { Some(color) })
		.collect::<Vec<_>>();

	let counts = centers.count.iter().copied().filter(|&n| n > 0).collect::<Vec<_>>();

	centers.reset();
	points.reset();

	KmeansResult { variance, centroids, counts, iterations }
}

/// Run multiple trials of k-means, taking the trial with the lowest variance
pub fn run_trials(
	data: &PixelDataVec,
	trials: u32,
	k: u8,
	max_iter: u32,
	convergence: f32,
	ignore_lightness: bool,
	seed: u64,
) -> KmeansResult {
	// data.len() <= u32::MAX because of `get_thumbnail` and `process_pixels`
	#[allow(clippy::cast_possible_truncation)]
	let mut state = KmeansState::new(k, data.len() as u32);

	if ignore_lightness {
		(0..trials)
			.map(|i| kmeans::<ChromaHueDistance>(data, &mut state, k, max_iter, convergence, seed + u64::from(i)))
			.min_by(|x, y| f64::total_cmp(&x.variance, &y.variance))
	} else {
		(0..trials)
			.map(|i| kmeans::<EuclideanDistance>(data, &mut state, k, max_iter, convergence, seed + u64::from(i)))
			.min_by(|x, y| f64::total_cmp(&x.variance, &y.variance))
	}
	.expect("at least one trial")
}
