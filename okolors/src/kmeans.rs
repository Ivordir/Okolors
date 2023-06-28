//! Provides the implementation for (sort) k-means

use crate::OklabCounts;
use palette::Oklab;
use rand::{Rng, SeedableRng};
#[cfg(feature = "threads")]
use rayon::prelude::*;
use std::ops::{AddAssign, SubAssign};

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

/// Holds all the state used by k-means
#[derive(Debug, Clone)]
struct KmeansState {
	/// Each centroid point
	centroids: Vec<Oklab>,
	/// Vector sum for all data points in each center
	sums: Vec<Oklab<f64>>,
	/// Number of points in each center
	counts: Vec<u32>,
	/// One fourth of the squared distance between each pairs of centers
	distances: Vec<(u8, f32)>,
	/// Center assignment for each data point
	assignments: Vec<u8>,
	/// Weight of each data point used to randomly select starting centroids in k-means++
	weights: Vec<f32>,
}

impl KmeansState {
	/// Initialize a new [`KmeansState`] with `k` centers and `n` data points
	fn new(k: u8, n: u32) -> Self {
		let k = usize::from(k);
		let n = n as usize;
		Self {
			centroids: Vec::new(),
			sums: vec![Oklab { l: 0.0, a: 0.0, b: 0.0 }; k],
			counts: vec![0; k],
			distances: vec![(0, 0.0); k * k],
			assignments: vec![0; n],
			weights: vec![f32::INFINITY; n],
		}
	}

	/// Reset data for the next k-means trial
	fn reset(&mut self) {
		self.centroids.clear();
		self.sums.fill(Oklab { l: 0.0, a: 0.0, b: 0.0 });
		self.counts.fill(0);
		// distances and assignments are corrected every iteration
		self.weights.fill(f32::INFINITY);
	}
}

/// Result from running k-means
#[derive(Debug, Clone)]
pub struct KmeansResult {
	/// Mean squared error (MSE) achieved by these centroids
	///
	/// This is the average squared distance/error each color is away from its centroid,
	/// so a lower MSE indicates a more accurate result.
	pub mse: f64,
	/// Final centroid colors
	pub centroids: Vec<Oklab>,
	/// Number of pixels in each centroid
	pub counts: Vec<u32>,
	/// Number of elapsed iterations
	pub iterations: u32,
}

impl KmeansResult {
	/// Create an empty result, representing that no k-means trials were able to be run
	#[must_use]
	pub const fn empty() -> Self {
		Self {
			mse: 0.0,
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
	oklab: &OklabCounts,
	KmeansState { centroids, weights, .. }: &mut KmeansState,
) {
	use rand::{
		distributions::{WeightedError::*, WeightedIndex},
		prelude::Distribution,
	};

	// Pick any random first centroid
	centroids.push(oklab.color_counts[rng.gen_range(0..oklab.color_counts.len())].0);

	// Pick each next centroid with a weighted probability based off the squared distance to its closest centroid
	for i in 1..usize::from(k) {
		let centroid = centroids[i - 1];
		for (weight, &(color, _)) in weights.iter_mut().zip(&oklab.color_counts) {
			*weight = f32::min(*weight, D::squared_distance(color, centroid));
		}

		match WeightedIndex::new(&*weights) {
			Ok(sampler) => centroids.push(oklab.color_counts[sampler.sample(rng)].0),
			Err(AllWeightsZero) => return, // all points exactly match a centroid
			Err(InvalidWeight | NoItem | TooMany) => {
				unreachable!("distances are >= 0 and colors.len() is in 1..=2.pow(24)")
			},
		}
	}
}

/// For each pair of centers, update their distances and sort each center's row by increasing distance
// i and j are < centroids.len() <= u8::MAX
#[allow(clippy::cast_possible_truncation)]
fn update_distances<D: ColorDifference>(KmeansState { centroids, distances, .. }: &mut KmeansState) {
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

/// Move a data point to its first assigned center
fn first_move_point<Count: From<u32> + AddAssign>(
	sums: &mut [Oklab<f64>],
	counts: &mut [Count],
	center: &mut u8,
	new_center: u8,
	color: Oklab,
	n: u32,
) {
	let cj = usize::from(new_center);
	let nf = f64::from(n);
	let sum = &mut sums[cj];
	sum.l += nf * f64::from(color.l);
	sum.a += nf * f64::from(color.a);
	sum.b += nf * f64::from(color.b);
	counts[cj] += Count::from(n);
	*center = new_center;
}

/// Move a data point from one center to another
fn move_point<Count: From<u32> + Copy + AddAssign + SubAssign>(
	sums: &mut [Oklab<f64>],
	counts: &mut [Count],
	center: &mut u8,
	new_center: u8,
	color: Oklab,
	n: u32,
) {
	if *center != new_center {
		let nf = f64::from(n);
		let n = Count::from(n);
		let l = nf * f64::from(color.l);
		let a = nf * f64::from(color.a);
		let b = nf * f64::from(color.b);

		let ci = usize::from(*center);
		let old_sum = &mut sums[ci];
		old_sum.l -= l;
		old_sum.a -= a;
		old_sum.b -= b;
		counts[ci] -= n;

		let cj = usize::from(new_center);
		let new_sum = &mut sums[cj];
		new_sum.l += l;
		new_sum.a += a;
		new_sum.b += b;
		counts[cj] += n;

		*center = new_center;
	}
}

/// For each data point, update its assigned center
#[cfg(feature = "threads")]
fn update_assignments<D: ColorDifference>(
	oklab: &OklabCounts,
	KmeansState {
		centroids,
		sums,
		counts,
		distances,
		assignments,
		..
	}: &mut KmeansState,
	change_assignment: impl Fn(&mut [Oklab<f64>], &mut [i64], &mut u8, u8, Oklab, u32) + Send + Sync,
) -> f64 {
	let k = centroids.len();
	let num_points = oklab.color_counts.len();
	let deltas = assignments
		.par_iter_mut()
		.with_min_len(num_points / rayon::current_num_threads())
		.zip(&oklab.color_counts)
		.fold_with(
			(vec![Oklab { l: 0.0, a: 0.0, b: 0.0 }; k], vec![0; k], 0.0),
			|(mut sums, mut counts, mut variance), (center, &(color, n))| {
				let ci = usize::from(*center);
				let dist = D::squared_distance(color, centroids[ci]);

				// Find the closest center
				let mut min_dist = dist;
				let mut min_center = *center;
				for &(other_center, half_dist) in &distances[(ci * k + 1)..((ci + 1) * k)] {
					if dist < half_dist {
						break;
					}

					let other_dist = D::squared_distance(color, centroids[usize::from(other_center)]);
					if other_dist < min_dist {
						min_dist = other_dist;
						min_center = other_center;
					}
				}

				change_assignment(&mut sums, &mut counts, center, min_center, color, n);

				variance += f64::from(n) * f64::from(dist);

				(sums, counts, variance)
			},
		)
		.collect::<Vec<_>>();

	let mut mse = 0.0;
	for (delta_sums, delta_counts, variance) in deltas {
		for (sum, delta_sum) in sums.iter_mut().zip(&delta_sums) {
			sum.l += delta_sum.l;
			sum.a += delta_sum.a;
			sum.b += delta_sum.b;
		}
		#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
		for (count, &delta_count) in counts.iter_mut().zip(&delta_counts) {
			let new_count = i64::from(*count) + delta_count;
			// Each center count is the sum of the counts of its points,
			// so moving all points out of this center cannot give a negative value.
			// Similarly, since the sum of the counts of all points is <= u32::MAX,
			// then moving all points into this center cannot give a value > u32::MAX.
			debug_assert!(u32::try_from(new_count).is_ok(), "{new_count}");
			*count = new_count as u32;
		}
		mse += variance;
	}

	mse / f64::from(oklab.num_colors())
}

/// For each data point, update its assigned center
#[cfg(not(feature = "threads"))]
fn update_assignments<D: ColorDifference>(
	oklab: &OklabCounts,
	KmeansState {
		centroids,
		sums,
		counts,
		distances,
		assignments,
		..
	}: &mut KmeansState,
	change_assignment: impl Fn(&mut [Oklab<f64>], &mut [u32], &mut u8, u8, Oklab, u32),
) -> f64 {
	let k = centroids.len();

	let mut mse = 0.0;
	for (&(color, n), center) in oklab.color_counts.iter().zip(assignments) {
		let ci = usize::from(*center);
		let dist = D::squared_distance(color, centroids[ci]);

		// Find the closest center
		let mut min_dist = dist;
		let mut min_center = *center;
		for &(other_center, half_dist) in &distances[(ci * k + 1)..((ci + 1) * k)] {
			if dist < half_dist {
				break;
			}

			let other_dist = D::squared_distance(color, centroids[usize::from(other_center)]);
			if other_dist < min_dist {
				min_dist = other_dist;
				min_center = other_center;
			}
		}

		change_assignment(sums, counts, center, min_center, color, n);

		mse += f64::from(n) * f64::from(dist);
	}
	mse / f64::from(oklab.num_colors())
}

/// For each center, update its centroid using the vector sums and compute deltas
fn update_centroids<D: ColorDifference>(
	rng: &mut impl Rng,
	KmeansState { centroids, sums, counts, .. }: &mut KmeansState,
) -> f32 {
	let mut total_delta = 0.0;
	for ((centroid, &n), sum) in centroids.iter_mut().zip(&*counts).zip(&*sums) {
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
	state: &mut KmeansState,
	k: u8,
	max_iter: u32,
	convergence: f32,
	seed: u64,
) -> KmeansResult {
	let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(seed);
	kmeans_plus_plus::<D>(k, &mut rng, oklab, state);

	let mut iterations = 0;
	let mut mse = 0.0;
	if max_iter > 0 {
		update_distances::<D>(state);
		mse = update_assignments::<D>(oklab, state, first_move_point);
		let mut total_delta = update_centroids::<D>(&mut rng, state);
		iterations += 1;

		while iterations < max_iter && total_delta > convergence {
			update_distances::<D>(state);
			mse = update_assignments::<D>(oklab, state, move_point);
			total_delta = update_centroids::<D>(&mut rng, state);
			iterations += 1;
		}
	}

	let (mut centroids, counts): (Vec<_>, Vec<_>) = state
		.centroids
		.iter()
		.zip(&state.counts)
		.filter_map(|(&color, &count)| if count == 0 { None } else { Some((color, count)) })
		.unzip();

	#[allow(clippy::float_cmp)]
	if oklab.lightness_weight != 0.0 && oklab.lightness_weight != 1.0 {
		for color in &mut centroids {
			color.l /= oklab.lightness_weight;
		}
	}

	state.reset();

	KmeansResult { mse, centroids, counts, iterations }
}

/// Run multiple trials of k-means, taking the trial with the lowest MSE
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
		.min_by(|x, y| f64::total_cmp(&x.mse, &y.mse))
		.unwrap_or(KmeansResult::empty())
}

/// Run multiple trials of k-means, taking the trial with the lowest MSE
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
	if k == 0 || oklab.color_counts.is_empty() {
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
		let counts = vec![12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
		OklabCounts {
			color_counts: test_colors().into_iter().zip(counts).collect(),
			lightness_weight: 1.0,
		}
	}

	fn kmeans_plus_plus_num_centroids(k: u8, n: u32) {
		let mut state = KmeansState::new(k, n);
		let mut oklab_counts = test_data();
		oklab_counts.color_counts.truncate(n as usize);

		kmeans_plus_plus::<EuclideanDistance>(
			k,
			&mut rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0),
			&oklab_counts,
			&mut state,
		);

		assert_eq!(state.centroids.len(), usize::min(usize::from(k), n as usize));
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
		#[allow(clippy::cast_possible_truncation)]
		let mut state = KmeansState::new(len as u8, len as u32);
		state.centroids = centroids;

		update_distances::<EuclideanDistance>(&mut state);

		#[allow(clippy::cast_possible_truncation)]
		for (i, row) in state.distances.chunks_exact(len).enumerate() {
			assert!(row[0] == (i as u8, 0.0));
			for j in 0..(len - 1) {
				assert!(row[j].1 <= row[j + 1].1);
			}
		}
	}

	fn initialize(k: u8) -> (OklabCounts, KmeansState, impl Rng) {
		let data = test_data();
		let mut state = KmeansState::new(k, data.num_colors());
		let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0);

		kmeans_plus_plus::<EuclideanDistance>(k, &mut rng, &data, &mut state);
		update_distances::<EuclideanDistance>(&mut state);
		update_assignments::<EuclideanDistance>(&data, &mut state, first_move_point);
		update_centroids::<EuclideanDistance>(&mut rng, &mut state);

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
		for &(color, count) in &data.color_counts {
			expected_count += count;
			let n = f64::from(count);
			expected_sum.l += n * f64::from(color.l);
			expected_sum.a += n * f64::from(color.a);
			expected_sum.b += n * f64::from(color.b);
		}

		assert_eq!(expected_count, state.counts.iter().sum());
		assert_relative_eq!(expected_sum, center_sum(&state.sums));
	}

	#[test]
	fn update_assignments_preverves_sum() {
		let (data, mut state, _) = initialize(4);
		update_distances::<EuclideanDistance>(&mut state);

		let expected_sum = center_sum(&state.sums);
		let expected_count = state.counts.iter().sum::<u32>();

		update_assignments::<EuclideanDistance>(&data, &mut state, move_point);

		assert_eq!(expected_count, state.counts.iter().sum());
		assert_relative_eq!(expected_sum, center_sum(&state.sums));
	}

	#[test]
	fn update_assignments_sum_reflects_assignment() {
		let (data, mut state, _) = initialize(4);
		update_distances::<EuclideanDistance>(&mut state);

		update_assignments::<EuclideanDistance>(&data, &mut state, move_point);

		for (&(color, count), &center) in data.color_counts.iter().zip(&state.assignments) {
			let center = usize::from(center);
			let n = f64::from(count);
			let sum = &mut state.sums[center];
			sum.l -= n * f64::from(color.l);
			sum.a -= n * f64::from(color.a);
			sum.b -= n * f64::from(color.b);
			state.counts[center] -= count;
		}

		for &sum in &state.sums {
			assert_relative_eq!(sum, Oklab { l: 0.0, a: 0.0, b: 0.0 });
		}

		for &count in &state.counts {
			assert_eq!(count, 0);
		}
	}

	#[test]
	fn update_centroids_total_delta() {
		let (data, mut state, mut rng) = initialize(4);

		let old_centroids = state.centroids.clone();

		update_distances::<EuclideanDistance>(&mut state);
		update_assignments::<EuclideanDistance>(&data, &mut state, move_point);

		let total_delta = update_centroids::<EuclideanDistance>(&mut rng, &mut state);

		let expected = old_centroids
			.iter()
			.zip(&state.centroids)
			.map(|(&old, &new)| EuclideanDistance::squared_distance(old, new).sqrt())
			.sum::<f32>();

		assert_relative_eq!(total_delta, expected);
	}

	fn assert_result_eq<D: ColorDifference>(data: &OklabCounts, k: u8, expected: &KmeansResult) {
		#[allow(clippy::cast_possible_truncation)]
		let mut state = KmeansState::new(k, data.num_colors());
		let result = kmeans::<D>(data, &mut state, k, 64, 0.01, 0);

		assert_relative_eq!(result.mse, expected.mse);
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
			mse: 0.012136615691512512,
			centroids: vec![
				Oklab {
					l: 0.13251413,
					a: -0.012265232,
					b: -0.0051606596,
				},
				Oklab {
					l: 0.24704194,
					a: 0.00046473244,
					b: -0.01437528,
				},
				Oklab {
					l: 0.29722875,
					a: -0.002119556,
					b: 0.08327842,
				},
				Oklab {
					l: 0.24430989,
					a: 0.108118445,
					b: 0.036724925,
				},
			],
			counts: vec![37, 12, 24, 5],
			iterations: 3,
		};

		assert_result_eq::<EuclideanDistance>(&data, k, &expected);
	}

	#[test]
	fn chroma_hue_distance_expected_results() {
		let k = 4;
		let data = test_data();

		let expected = KmeansResult {
			mse: 0.004202910981499978,
			centroids: vec![
				Oklab {
					l: 0.19259314,
					a: 0.0021442638,
					b: 0.013763567,
				},
				Oklab {
					l: 0.28536618,
					a: 0.0014207959,
					b: 0.11500679,
				},
				Oklab {
					l: 0.24430989,
					a: 0.108118445,
					b: 0.036724925,
				},
				Oklab {
					l: 0.17056544,
					a: -0.05170843,
					b: -0.043375432,
				},
			],
			counts: vec![48, 13, 5, 12],
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

		for color_count in &mut data.color_counts {
			color_count.0.l *= weight;
		}

		let expected = KmeansResult {
			mse: 0.008327938582321318,
			centroids: vec![
				Oklab {
					l: 0.16056176,
					a: -0.00914769,
					b: -0.007417301,
				},
				Oklab {
					l: 0.28536618,
					a: 0.0014207959,
					b: 0.11500679,
				},
				Oklab {
					l: 0.24430989,
					a: 0.108118445,
					b: 0.036724925,
				},
				Oklab {
					l: 0.31124818,
					a: -0.0063036084,
					b: 0.045781255,
				},
			],
			counts: vec![49, 13, 5, 11],
			iterations: 4,
		};

		assert_result_eq::<EuclideanDistance>(&data, k, &expected);
	}

	#[test]
	fn lower_convergence_gives_lower_mse() {
		let k = 2;
		let data = test_data();

		let higher = run(&data, 1, k, 0.1, 64, 0);
		let lower = run(&data, 1, k, 0.01, 64, 0);

		assert!(higher.mse > lower.mse);
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
