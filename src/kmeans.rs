//! Provides the implementation for (sort) k-means

use crate::OklabCounts;
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
struct CenterData {
	/// The centroid point
	centroid: Vec<Oklab>,
	/// Vector sum for all data points in this center
	sum: Vec<Oklab<f64>>,
	/// Number of points in this center
	count: Vec<u32>,
}

impl CenterData {
	/// Initialize Vecs to the given number of centers
	fn new(k: u8) -> Self {
		let k = usize::from(k);
		Self {
			centroid: Vec::new(),
			sum: vec![Oklab { l: 0.0, a: 0.0, b: 0.0 }; k],
			count: vec![0; k],
		}
	}

	/// Reset data for the next k-means trial
	fn reset(&mut self) {
		self.centroid.clear();
		self.sum.fill(Oklab { l: 0.0, a: 0.0, b: 0.0 });
		self.count.fill(0);
	}
}

/// Holds all the state used by k-means
struct KmeansState {
	/// Data for each center
	centers: CenterData,
	/// One fourth of the squared distance between each pairs of centers
	distances: Vec<(u8, f32)>,
	/// Data for each point
	points: PointDataVec,
}

impl KmeansState {
	/// Initialize a new `KmeansState` with `k` centers and `n` data points
	fn new(k: u8, n: u32) -> Self {
		Self {
			centers: CenterData::new(k),
			distances: vec![(0, 0.0); usize::from(k) * usize::from(k)],
			points: PointDataVec::init(n),
		}
	}
}

/// Result from running k-means
pub struct KmeansResult {
	/// Variance achieved by these centroids
	///
	/// A lower variance indicates a higher accuracy.
	pub variance: f64,
	/// Final centroid colors
	pub centroids: Vec<Oklab>,
	/// Number of pixels in each centroid
	pub counts: Vec<u32>,
	/// Number of elapsed iterations
	pub iterations: u32,
}

impl KmeansResult {
	/// Create an empty result, representing that no k-means trials were able to be run
	const fn empty() -> Self {
		Self {
			variance: 0.0,
			centroids: Vec::new(),
			counts: Vec::new(),
			iterations: 0,
		}
	}
}

/// Choose the starting centroids using the k-means++ algorithm
fn kmeans_plus_plus<D: ColorDifference>(
	k: u8,
	mut rng: &mut impl Rng,
	colors: &[Oklab],
	centroids: &mut Vec<Oklab>,
	weights: &mut [f32],
) {
	use rand::{
		distributions::{WeightedError::*, WeightedIndex},
		prelude::Distribution,
	};

	// Pick any random first centroid
	centroids.push(colors[rng.gen_range(0..colors.len())]);

	// Pick each next centroid with a weighted probability based off the squared distance to its closest centroid
	for i in 1..usize::from(k) {
		let centroid = centroids[i - 1];
		for (weight, &color) in weights.iter_mut().zip(colors) {
			*weight = f32::min(*weight, D::squared_distance(color, centroid));
		}

		match WeightedIndex::new(weights.iter().copied()) {
			Ok(sampler) => centroids.push(colors[sampler.sample(&mut rng)]),
			Err(AllWeightsZero) => return, // all points exactly match a centroid
			Err(InvalidWeight | NoItem | TooMany) => {
				unreachable!("distances are >= 0 and data.len() is in 1..=u32::MAX")
			},
		}
	}
}

/// Initializes the center sums and counts based off the initial centroids
fn compute_initial_sums(oklab: &OklabCounts, centers: &mut CenterData, assignment: &[u8]) {
	for (color, &n, &center) in soa_zip!(oklab, [colors, counts], assignment) {
		let i = usize::from(center);
		let nf = f64::from(n);
		let sum = &mut centers.sum[i];
		sum.l += nf * f64::from(color.l);
		sum.a += nf * f64::from(color.a);
		sum.b += nf * f64::from(color.b);
		centers.count[i] += n;
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

	for row in distances.chunks_exact_mut(k) {
		row.sort_by(|(_, x), (_, y)| f32::total_cmp(x, y));
	}
}

/// For each data point, update its assigned center
fn update_assignments<D: ColorDifference>(
	oklab: &OklabCounts,
	centers: &mut CenterData,
	distances: &[(u8, f32)],
	points: &mut PointDataVec,
) {
	let k = centers.centroid.len();
	for (&color, &n, center) in soa_zip!(oklab, [colors, counts], &mut points.assignment) {
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
fn update_centroids<D: ColorDifference>(rng: &mut impl Rng, centers: &mut CenterData) -> f32 {
	let mut total_delta = 0.0;
	for (centroid, &n, sum) in soa_zip!(centers, [mut centroid, count, sum]) {
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

		total_delta += D::squared_distance(*centroid, new_centroid).sqrt();
		*centroid = new_centroid;
	}

	total_delta
}

/// Run a trial of sort k-means
fn kmeans<D: ColorDifference>(
	oklab: &OklabCounts,
	KmeansState { centers, distances, points }: &mut KmeansState,
	k: u8,
	max_iter: u32,
	convergence: f32,
	seed: u64,
) -> KmeansResult {
	let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
	kmeans_plus_plus::<D>(k, &mut rng, &oklab.colors, &mut centers.centroid, &mut points.weight);
	compute_initial_sums(oklab, centers, &points.assignment);

	let mut iterations = 0;
	loop {
		update_distances::<D>(&centers.centroid, distances);
		update_assignments::<D>(oklab, centers, distances, points);
		let total_delta = update_centroids::<D>(&mut rng, centers);

		iterations += 1;

		if iterations >= max_iter || total_delta <= convergence {
			break;
		}
	}

	let variance = soa_zip!(oklab, [colors, counts], &points.assignment)
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
fn run_trials<D: ColorDifference>(
	oklab: &OklabCounts,
	trials: u32,
	k: u8,
	max_iter: u32,
	convergence: f32,
	seed: u64,
) -> KmeansResult {
	let n = u32::try_from(oklab.colors.len()).expect("number of colors is within u32::MAX");
	let mut state = KmeansState::new(k, n);

	(0..trials)
		.map(|i| kmeans::<D>(oklab, &mut state, k, max_iter, convergence, seed ^ u64::from(i)))
		.min_by(|x, y| f64::total_cmp(&x.variance, &y.variance))
		.unwrap_or(KmeansResult::empty())
}

/// Run multiple trials of k-means, taking the trial with the lowest variance
///
/// An empty result with no centroids is returned if `oklab` is empty, `trials` = 0, or `k` = 0.
///
/// # Panics
/// Panics if the length of `oklab` is greater than `u32::MAX`
pub fn run(
	oklab: &OklabCounts,
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
	ignore_lightness: bool,
) -> KmeansResult {
	if k == 0 || oklab.colors.is_empty() {
		KmeansResult::empty()
	} else if ignore_lightness {
		run_trials::<ChromaHueDistance>(oklab, trials, k, max_iter, convergence_threshold, seed)
	} else {
		run_trials::<EuclideanDistance>(oklab, trials, k, max_iter, convergence_threshold, seed)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	// TODO: use relative error instead of absolute for floating point comparison

	fn test_colors() -> Vec<Oklab> {
		vec![
			Oklab { l: 0.5, a: 0.25, b: 0.25 },
			Oklab { l: 0.25, a: 0.1, b: 0.3 },
			Oklab { l: 0.5, a: 0.0, b: 0.0 },
			Oklab { l: 0.6, a: 0.2, b: 0.1 },
			Oklab { l: 0.1, a: 0.2, b: 0.3 },
			Oklab { l: 0.3, a: 0.3, b: 0.3 },
		]
	}

	fn test_counts() -> Vec<u32> {
		vec![4, 3, 2, 3, 1, 2]
	}

	fn kmeans_plus_plus_num_centroids(k: u8, n: u32) {
		let mut state = KmeansState::new(k, n);

		kmeans_plus_plus::<EuclideanDistance>(
			k,
			&mut rand_chacha::ChaCha8Rng::seed_from_u64(0),
			&test_colors()[..(n as usize)],
			&mut state.centers.centroid,
			&mut state.points.weight,
		);

		assert_eq!(state.centers.centroid.len(), usize::min(usize::from(k), n as usize));
	}

	#[test]
	fn kmeans_plus_plus_k_greater_than_n() {
		kmeans_plus_plus_num_centroids(6, 2);
	}

	#[test]
	fn kmeans_plus_plus_k_equals_n() {
		kmeans_plus_plus_num_centroids(4, 4);
	}

	#[test]
	fn kmeans_plus_plus_k_less_than_n() {
		kmeans_plus_plus_num_centroids(2, 6);
	}

	#[test]
	fn update_distances_sorts_each_row() {
		let data = test_colors();
		let len = data.len();
		let mut distances = vec![(0, 0.0); len * len];

		update_distances::<EuclideanDistance>(&data, &mut distances);

		#[allow(clippy::cast_possible_truncation)]
		for (i, row) in distances.chunks_exact(len).enumerate() {
			assert!(row[0] == (i as u8, 0.0));
			for j in 0..(len - 1) {
				assert!(row[j].1 <= row[j + 1].1);
			}
		}
	}

	fn initialize(k: u8) -> (OklabCounts, KmeansState, impl Rng) {
		let data = OklabCounts {
			colors: test_colors(),
			counts: test_counts(),
		};
		#[allow(clippy::cast_possible_truncation)]
		let mut state = KmeansState::new(k, data.colors.len() as u32);
		let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

		kmeans_plus_plus::<EuclideanDistance>(
			k,
			&mut rng,
			&data.colors,
			&mut state.centers.centroid,
			&mut state.points.weight,
		);

		compute_initial_sums(&data, &mut state.centers, &state.points.assignment);

		(data, state, rng)
	}

	fn center_sum(sums: &[Oklab<f64>]) -> Oklab<f64> {
		let mut center_sum = Oklab { l: 0.0, a: 0.0, b: 0.0 };
		for sum in sums {
			center_sum.l += sum.l;
			center_sum.a += sum.a;
			center_sum.b += sum.b;
		}
		center_sum
	}

	fn assert_oklab_eq(x: Oklab<f64>, y: Oklab<f64>) {
		assert!((x.l - y.l).abs() <= 1e-16);
		assert!((x.a - y.a).abs() <= 1e-16);
		assert!((x.b - y.b).abs() <= 1e-16);
	}

	#[test]
	fn compute_initial_sums_preserves_sum() {
		let (data, state, _) = initialize(4);

		let mut expected_sum = Oklab { l: 0.0, a: 0.0, b: 0.0 };
		let mut expected_count = 0;
		for (&color, &count) in soa_zip!(&data, [colors, counts]) {
			expected_count += count;
			let n = f64::from(count);
			expected_sum.l += n * f64::from(color.l);
			expected_sum.a += n * f64::from(color.a);
			expected_sum.b += n * f64::from(color.b);
		}

		let sum = center_sum(&state.centers.sum);

		assert_eq!(expected_count, state.centers.count.iter().sum());
		assert_oklab_eq(sum, expected_sum);
	}

	#[test]
	fn update_assignments_preverves_sum() {
		let (data, mut state, _) = initialize(4);

		let expected_sum = center_sum(&state.centers.sum);
		let expected_count = state.centers.count.iter().sum::<u32>();

		update_assignments::<EuclideanDistance>(&data, &mut state.centers, &state.distances, &mut state.points);

		let sum = center_sum(&state.centers.sum);

		assert_eq!(expected_count, state.centers.count.iter().sum());
		assert_oklab_eq(sum, expected_sum);
	}

	#[test]
	fn update_assignments_sum_reflects_assignment() {
		let (data, mut state, _) = initialize(4);

		update_assignments::<EuclideanDistance>(&data, &mut state.centers, &state.distances, &mut state.points);

		for (&color, &count, &center) in soa_zip!(&data, [colors, counts], &state.points.assignment) {
			let center = usize::from(center);
			let n = f64::from(count);
			let sum = &mut state.centers.sum[center];
			sum.l -= n * f64::from(color.l);
			sum.a -= n * f64::from(color.a);
			sum.b -= n * f64::from(color.b);
			state.centers.count[center] -= count;
		}

		for &sum in &state.centers.sum {
			assert_oklab_eq(sum, Oklab { l: 0.0, a: 0.0, b: 0.0 });
		}

		for &count in &state.centers.count {
			assert_eq!(count, 0);
		}
	}

	#[test]
	fn update_centroids_total_delta() {
		let (data, mut state, mut rng) = initialize(4);

		let old_centroids = state.centers.centroid.clone();

		update_assignments::<EuclideanDistance>(&data, &mut state.centers, &state.distances, &mut state.points);

		let total_delta = update_centroids::<EuclideanDistance>(&mut rng, &mut state.centers);

		let expected = old_centroids
			.iter()
			.zip(&state.centers.centroid)
			.map(|(&old, &new)| EuclideanDistance::squared_distance(old, new).sqrt())
			.sum::<f32>();

		assert!((total_delta - expected).abs() <= 1e-16);
	}
}
