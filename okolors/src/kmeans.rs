//! Provides the implementation for (sort) k-means

use crate::OklabCounts;
use palette::Oklab;
use rand::{Rng, SeedableRng};

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
struct PointData {
	/// Center assignment for this data point
	assignment: Vec<u8>,
	/// Weight of each data point used to randomly select starting centroids in k-means++
	weight: Vec<f32>,
}

impl PointData {
	/// Create a [`PointData`] with the given number data points
	fn new(n: u32) -> Self {
		let n = n as usize;
		Self {
			assignment: vec![0; n],
			weight: vec![f32::INFINITY; n],
		}
	}

	/// Reset data for the next k-means trial
	fn reset(&mut self) {
		// assignments are corrected every iteration
		self.weight.fill(f32::INFINITY);
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
	/// Create a [`CenterData`] with the given number of centers
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
	points: PointData,
}

impl KmeansState {
	/// Initialize a new [`KmeansState`] with `k` centers and `n` data points
	fn new(k: u8, n: u32) -> Self {
		Self {
			centers: CenterData::new(k),
			distances: vec![(0, 0.0); usize::from(k) * usize::from(k)],
			points: PointData::new(n),
		}
	}
}

/// Result from running k-means
#[derive(Debug, Clone)]
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
	rng: &mut impl Rng,
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

		match WeightedIndex::new(&*weights) {
			Ok(sampler) => centroids.push(colors[sampler.sample(rng)]),
			Err(AllWeightsZero) => return, // all points exactly match a centroid
			Err(InvalidWeight | NoItem | TooMany) => {
				unreachable!("distances are >= 0 and colors.len() is in 1..=2.pow(24)")
			},
		}
	}
}

/// Initializes the center sums and counts based off the initial centroids
fn compute_initial_sums(oklab: &OklabCounts, centers: &mut CenterData, assignment: &[u8]) {
	for ((color, n), &center) in oklab.pairs().zip(assignment) {
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
#[cfg(not(feature = "threads"))]
fn update_assignments<D: ColorDifference>(
	oklab: &OklabCounts,
	centers: &mut CenterData,
	distances: &[(u8, f32)],
	points: &mut PointData,
) {
	let k = centers.centroid.len();
	for ((color, n), center) in oklab.pairs().zip(&mut points.assignment) {
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

/// For each data point, update its assigned center
#[cfg(feature = "threads")]
fn update_assignments<D: ColorDifference>(
	oklab: &OklabCounts,
	centers: &mut CenterData,
	distances: &[(u8, f32)],
	points: &mut PointData,
) {
	use rayon::prelude::*;

	let k = centers.centroid.len();
	let num_points = oklab.colors().len();
	let deltas = points
		.assignment
		.par_iter_mut()
		.with_min_len(num_points / rayon::current_num_threads())
		.zip(&oklab.colors)
		.zip(&oklab.counts)
		.fold_with(
			(vec![Oklab { l: 0.0, a: 0.0, b: 0.0 }; k], vec![0; k]),
			|(mut sums, mut counts), ((center, &color), &n)| {
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
					let n = i64::from(n);
					let l = nf * f64::from(color.l);
					let a = nf * f64::from(color.a);
					let b = nf * f64::from(color.b);

					let old_sum = &mut sums[ci];
					old_sum.l -= l;
					old_sum.a -= a;
					old_sum.b -= b;
					counts[ci] -= n;

					let cj = usize::from(min_center);

					let new_sum = &mut sums[cj];
					new_sum.l += l;
					new_sum.a += a;
					new_sum.b += b;
					counts[cj] += n;

					*center = min_center;
				}

				(sums, counts)
			},
		)
		.collect::<Vec<_>>();

	for (delta_sums, delta_counts) in deltas {
		for (sum, delta_sum) in centers.sum.iter_mut().zip(&delta_sums) {
			sum.l += delta_sum.l;
			sum.a += delta_sum.a;
			sum.b += delta_sum.b;
		}
		#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
		for (count, &delta_count) in centers.count.iter_mut().zip(&delta_counts) {
			let new_count = i64::from(*count) + delta_count;
			// Each center count is the sum of the counts of its points,
			// so moving all points out of this center cannot give a negative value.
			// Similarly, since the sum of the counts of all points is <= u32::MAX,
			// then moving all points into this center cannot give a value > u32::MAX.
			debug_assert!(u32::try_from(new_count).is_ok());
			*count = new_count as u32;
		}
	}
}

/// For each center, update its centroid using the vector sums and compute deltas
fn update_centroids<D: ColorDifference>(rng: &mut impl Rng, centers: &mut CenterData) -> f32 {
	let mut total_delta = 0.0;
	for ((centroid, &n), sum) in centers.centroid.iter_mut().zip(&centers.count).zip(&centers.sum) {
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
	let mut total_delta = f32::INFINITY;
	while iterations < max_iter && total_delta > convergence {
		update_distances::<D>(&centers.centroid, distances);
		update_assignments::<D>(oklab, centers, distances, points);
		total_delta = update_centroids::<D>(&mut rng, centers);
		iterations += 1;
	}

	let variance = oklab
		.pairs()
		.zip(&points.assignment)
		.map(|((color, n), &center)| {
			f64::from(n) * f64::from(D::squared_distance(color, centers.centroid[usize::from(center)]))
		})
		.sum();

	let mut centroids = centers
		.centroid
		.iter()
		.zip(&centers.count)
		.filter_map(|(&color, &count)| if count == 0 { None } else { Some(color) })
		.collect::<Vec<_>>();

	#[allow(clippy::float_cmp)]
	if oklab.lightness_weight != 0.0 && oklab.lightness_weight != 1.0 {
		for color in &mut centroids {
			color.l /= oklab.lightness_weight;
		}
	}

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
	let mut state = KmeansState::new(k, oklab.num_colors());

	(0..trials)
		.map(|i| kmeans::<D>(oklab, &mut state, k, max_iter, convergence, seed ^ u64::from(i)))
		.min_by(|x, y| f64::total_cmp(&x.variance, &y.variance))
		.unwrap_or(KmeansResult::empty())
}

/// Run multiple trials of k-means, taking the trial with the lowest variance
///
/// An empty result with no centroids is returned if `oklab` is empty, `trials` = 0, or `k` = 0.
pub fn run(
	oklab: &OklabCounts,
	trials: u32,
	k: u8,
	convergence_threshold: f32,
	max_iter: u32,
	seed: u64,
) -> KmeansResult {
	#[allow(clippy::float_cmp)]
	if k == 0 || oklab.colors.is_empty() {
		KmeansResult::empty()
	} else if oklab.lightness_weight == 0.0 {
		run_trials::<ChromaHueDistance>(oklab, trials, k, max_iter, convergence_threshold, seed)
	} else {
		run_trials::<EuclideanDistance>(oklab, trials, k, max_iter, convergence_threshold, seed)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use approx::assert_relative_eq;

	fn test_colors() -> Vec<Oklab> {
		vec![
			Oklab {
				l: 0.13746236,
				a: 0.0035205781,
				b: -0.0074754357,
			},
			Oklab {
				l: 0.31124818,
				a: -0.0063036084,
				b: 0.045781255,
			},
			Oklab {
				l: 0.1562507,
				a: -0.063299775,
				b: -0.032072306,
			},
			Oklab {
				l: 0.2438251,
				a: -0.0007998347,
				b: 0.0027060509,
			},
			Oklab {
				l: 0.1443736,
				a: 0.0025939345,
				b: -0.004264146,
			},
			Oklab {
				l: 0.076568395,
				a: 0.016597964,
				b: 0.03622815,
			},
			Oklab {
				l: 0.28073087,
				a: 0.026428253,
				b: 0.116048574,
			},
			Oklab {
				l: 0.24430989,
				a: 0.108118445,
				b: 0.036724925,
			},
			Oklab {
				l: 0.288357,
				a: -0.008182496,
				b: 0.112403214,
			},
			Oklab {
				l: 0.29064906,
				a: -0.03578973,
				b: 0.11639464,
			},
			Oklab {
				l: 0.24213916,
				a: 0.0062482953,
				b: -0.09989107,
			},
			Oklab {
				l: 0.28579916,
				a: 0.00027871132,
				b: 0.002924323,
			},
		]
	}

	fn test_data() -> OklabCounts {
		OklabCounts {
			colors: test_colors(),
			counts: vec![12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
			lightness_weight: 1.0,
		}
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
		let centroids = test_colors();
		let len = centroids.len();
		let mut distances = vec![(0, 0.0); len * len];

		update_distances::<EuclideanDistance>(&centroids, &mut distances);

		#[allow(clippy::cast_possible_truncation)]
		for (i, row) in distances.chunks_exact(len).enumerate() {
			assert!(row[0] == (i as u8, 0.0));
			for j in 0..(len - 1) {
				assert!(row[j].1 <= row[j + 1].1);
			}
		}
	}

	fn initialize(k: u8) -> (OklabCounts, KmeansState, impl Rng) {
		let data = test_data();
		#[allow(clippy::cast_possible_truncation)]
		let mut state = KmeansState::new(k, data.num_colors());
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

	#[test]
	fn compute_initial_sums_preserves_sum() {
		let (data, state, _) = initialize(4);

		let mut expected_sum = Oklab { l: 0.0, a: 0.0, b: 0.0 };
		let mut expected_count = 0;
		for (color, count) in data.pairs() {
			expected_count += count;
			let n = f64::from(count);
			expected_sum.l += n * f64::from(color.l);
			expected_sum.a += n * f64::from(color.a);
			expected_sum.b += n * f64::from(color.b);
		}

		assert_eq!(expected_count, state.centers.count.iter().sum());
		assert_relative_eq!(expected_sum, center_sum(&state.centers.sum));
	}

	#[test]
	fn update_assignments_preverves_sum() {
		let (data, mut state, _) = initialize(4);

		let expected_sum = center_sum(&state.centers.sum);
		let expected_count = state.centers.count.iter().sum::<u32>();

		update_assignments::<EuclideanDistance>(&data, &mut state.centers, &state.distances, &mut state.points);

		assert_eq!(expected_count, state.centers.count.iter().sum());
		assert_relative_eq!(expected_sum, center_sum(&state.centers.sum));
	}

	#[test]
	fn update_assignments_sum_reflects_assignment() {
		let (data, mut state, _) = initialize(4);

		update_assignments::<EuclideanDistance>(&data, &mut state.centers, &state.distances, &mut state.points);

		for ((color, count), &center) in data.pairs().zip(&state.points.assignment) {
			let center = usize::from(center);
			let n = f64::from(count);
			let sum = &mut state.centers.sum[center];
			sum.l -= n * f64::from(color.l);
			sum.a -= n * f64::from(color.a);
			sum.b -= n * f64::from(color.b);
			state.centers.count[center] -= count;
		}

		for &sum in &state.centers.sum {
			assert_relative_eq!(sum, Oklab { l: 0.0, a: 0.0, b: 0.0 });
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

		assert!((total_delta - expected).abs() <= 1e-8);
	}

	fn assert_result_eq<D: ColorDifference>(data: &OklabCounts, k: u8, expected: &KmeansResult) {
		#[allow(clippy::cast_possible_truncation)]
		let mut state = KmeansState::new(k, data.num_colors());
		let result = kmeans::<D>(data, &mut state, k, 64, 0.01, 0);

		assert!((result.variance - expected.variance).abs() <= 1e-8);
		for (&result, &expected) in result.centroids.iter().zip(&expected.centroids) {
			assert_relative_eq!(result, expected);
		}
		assert_eq!(result.counts, expected.counts);
		assert!(result.iterations <= expected.iterations);
	}

	#[test]
	fn euclidean_distance_expected_results() {
		let k = 4;
		let data = test_data();

		let expected = KmeansResult {
			variance: 0.22378644230775535,
			centroids: vec![
				Oklab {
					l: 0.2967716,
					a: -0.0020236254,
					b: 0.08006425,
				},
				Oklab {
					l: 0.24430989,
					a: 0.108118445,
					b: 0.036724925,
				},
				Oklab {
					l: 0.15429236,
					a: -0.010022002,
					b: -0.0036215205,
				},
				Oklab {
					l: 0.24213916,
					a: 0.0062482953,
					b: -0.09989107,
				},
			],
			counts: vec![25, 5, 46, 2],
			iterations: 2,
		};

		assert_result_eq::<EuclideanDistance>(&data, k, &expected);
	}

	#[test]
	fn chroma_hue_distance_expected_results() {
		let k = 4;
		let data = test_data();

		let expected = KmeansResult {
			variance: 0.0875395935800043,
			centroids: vec![
				Oklab {
					l: 0.28536618,
					a: 0.0014207959,
					b: 0.11500679,
				},
				Oklab {
					l: 0.24213916,
					a: 0.0062482953,
					b: -0.09989107,
				},
				Oklab {
					l: 0.1863272,
					a: -0.009139191,
					b: 0.0058608307,
				},
				Oklab {
					l: 0.24430989,
					a: 0.108118445,
					b: 0.036724925,
				},
			],
			counts: vec![13, 2, 58, 5],
			iterations: 2,
		};

		assert_result_eq::<ChromaHueDistance>(&data, k, &expected);
	}

	#[test]
	fn lightness_weight_expected_results() {
		let k = 4;
		let weight = 0.325;
		let mut data = test_data();
		data.lightness_weight = weight;

		for color in &mut data.colors {
			color.l *= weight;
		}

		let expected = KmeansResult {
			variance: 0.10947524901712313,
			centroids: vec![
				Oklab {
					l: 0.29722878,
					a: -0.002119556,
					b: 0.08327842,
				},
				Oklab {
					l: 0.24213916,
					a: 0.0062482953,
					b: -0.09989107,
				},
				Oklab {
					l: 0.15709038,
					a: -0.009802838,
					b: -0.0034822472,
				},
				Oklab {
					l: 0.24430989,
					a: 0.108118445,
					b: 0.036724925,
				},
			],
			counts: vec![24, 2, 47, 5],
			iterations: 2,
		};

		assert_result_eq::<EuclideanDistance>(&data, k, &expected);
	}

	#[test]
	fn lower_convergence_gives_lower_variance() {
		let k = 2;
		let data = test_data();

		let higher = run(&data, 1, k, 0.1, 64, 0);
		let lower = run(&data, 1, k, 0.01, 64, 0);

		assert!(higher.variance > lower.variance);
	}

	#[test]
	fn max_iter_reached() {
		let data = test_data();

		let many_iter = 64;
		let converged = run(&data, 1, 4, 0.01, many_iter, 0);
		assert!(converged.iterations < many_iter);

		let max_iter = converged.iterations / 2;
		assert!(max_iter > 0);
		let result = run(&data, 1, 4, 0.1, max_iter, 0);
		assert_eq!(result.iterations, max_iter);
	}
}
