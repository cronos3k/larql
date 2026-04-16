//! K-means clustering with BLAS-accelerated distance computation.

use ndarray::Array2;

/// Run k-means clustering on normalized direction vectors.
/// Returns (centres, assignments, distances).
pub fn kmeans(
    data: &Array2<f32>,
    k: usize,
    max_iterations: usize,
) -> (Array2<f32>, Vec<usize>, Vec<f32>) {
    let n = data.shape()[0];
    let dim = data.shape()[1];

    if n == 0 || k == 0 {
        return (Array2::zeros((0, dim)), vec![], vec![]);
    }

    let k = k.min(n);
    let mut centres = kmeans_pp_init(data, k);
    let mut assignments = vec![0usize; n];
    let mut distances = vec![0.0f32; n];

    for _iter in 0..max_iterations {
        // BLAS: similarities = data @ centres.T → (n, k)
        let cpu = larql_compute::default_backend();
        use larql_compute::ComputeBackend;
        let sims = cpu.matmul_transb(data.view(), centres.view());

        let mut changed = false;
        for i in 0..n {
            let row = sims.row(i);
            let mut best_c = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for c in 0..k {
                if row[c] > best_sim {
                    best_sim = row[c];
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                changed = true;
            }
            assignments[i] = best_c;
            distances[i] = 1.0 - best_sim;
        }

        if !changed {
            break;
        }

        // Recompute and normalize centres
        let mut new_centres = Array2::<f32>::zeros((k, dim));
        let mut counts = vec![0usize; k];

        for (i, &c) in assignments.iter().enumerate() {
            counts[c] += 1;
            let row = data.row(i);
            for j in 0..dim {
                new_centres[[c, j]] += row[j];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                for j in 0..dim {
                    new_centres[[c, j]] /= cnt;
                }
                let norm: f32 = larql_compute::norm(&new_centres.row(c));
                if norm > 1e-8 {
                    for j in 0..dim {
                        new_centres[[c, j]] /= norm;
                    }
                }
            }
        }

        centres = new_centres;
    }

    (centres, assignments, distances)
}

/// K-means++ initialisation with BLAS distance computation.
fn kmeans_pp_init(data: &Array2<f32>, k: usize) -> Array2<f32> {
    let n = data.shape()[0];
    let dim = data.shape()[1];
    let mut centres = Array2::<f32>::zeros((k, dim));

    // First centre: point with largest norm
    let mut best_norm = 0.0f32;
    let mut first = 0;
    for i in 0..n {
        let norm: f32 = larql_compute::dot(&data.row(i), &data.row(i));
        if norm > best_norm {
            best_norm = norm;
            first = i;
        }
    }
    centres.row_mut(0).assign(&data.row(first));

    let mut min_dists = vec![f32::MAX; n];

    for c in 1..k {
        let prev = centres.row(c - 1);
        let dim = prev.len();
        let prev_2d = prev.view().into_shape_with_order((dim, 1)).unwrap();
        let cpu = larql_compute::default_backend();
        use larql_compute::ComputeBackend;
        let sims_2d = cpu.matmul(data.view(), prev_2d.view()); // [n, 1]
        let sims = ndarray::Array1::from_vec(sims_2d.into_raw_vec_and_offset().0);
        for i in 0..n {
            let dist = 1.0 - sims[i];
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
        }

        let mut best_i = 0;
        let mut best_d = f32::NEG_INFINITY;
        for (i, &dist) in min_dists.iter().enumerate().take(n) {
            if dist > best_d {
                best_d = dist;
                best_i = i;
            }
        }

        centres.row_mut(c).assign(&data.row(best_i));
    }

    centres
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_basic() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 0.0, 0.9, 0.1, 0.8, 0.2,
                0.0, 1.0, 0.1, 0.9, 0.2, 0.8,
            ],
        )
        .unwrap();

        let (centres, assignments, _) = kmeans(&data, 2, 100);
        assert_eq!(centres.shape(), &[2, 2]);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }

    #[test]
    fn kmeans_single_cluster() {
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 0.0, 0.9, 0.1, 0.95, 0.05],
        )
        .unwrap();

        let (centres, assignments, _) = kmeans(&data, 1, 50);
        assert_eq!(centres.shape(), &[1, 2]);
        assert!(assignments.iter().all(|&a| a == 0));
    }
}
