//! Filter functions for the Mapper algorithm.
//!
//! A filter function maps high-dimensional data points to a lower-dimensional
//! space (typically ℝ or ℝ^d). This filtered representation is then covered
//! by overlapping intervals, forming the basis of the Mapper construction.
//!
//! # Built-in Filters
//!
//! - **PCA projection**: Projects onto the first principal component using an
//!   iterative power method for the top eigenvector.
//! - **Distance from centroid**: L² distance of each point from the dataset mean.
//! - **Eccentricity**: Mean distance from each point to all other points.
//! - **Kernel density estimate**: Gaussian KDE evaluated at each point.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A single data point, represented as a vector of f64 coordinates.
pub type DataPoint = Vec<f64>;

/// A filter function maps a data point to one or more filter values.
///
/// Single-dimensional filters return `Vec<f64>` with one element;
/// multi-dimensional filters return multiple elements.
#[derive(Serialize, Deserialize)]
pub struct FilterFunction {
    /// Human-readable name for this filter.
    pub name: String,
    /// The filter implementation. Receives a reference to the full dataset
    /// and the index of the point to evaluate.
    ///
    /// For built-in filters, the closure captures precomputed values
    /// (e.g., the centroid for distance-from-centroid).
    #[serde(skip)]
    #[allow(clippy::type_complexity)]
    pub compute: Option<Box<dyn Fn(&[DataPoint], usize) -> Vec<f64>>>,
}

impl fmt::Debug for FilterFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterFunction")
            .field("name", &self.name)
            .field(
                "compute",
                &if self.compute.is_some() {
                    Some("<closure>")
                } else {
                    None
                },
            )
            .finish()
    }
}

impl FilterFunction {
    /// Create a filter from a closure.
    pub fn new<F>(name: impl Into<String>, compute: F) -> Self
    where
        F: Fn(&[DataPoint], usize) -> Vec<f64> + 'static,
    {
        Self {
            name: name.into(),
            compute: Some(Box::new(compute)),
        }
    }

    /// Apply this filter to a single point within the dataset.
    ///
    /// # Panics
    /// Panics if the filter has no compute function.
    pub fn apply(&self, data: &[DataPoint], index: usize) -> Vec<f64> {
        (self
            .compute
            .as_ref()
            .expect("FilterFunction has no compute closure"))(data, index)
    }
}

/// Result of applying a filter to an entire dataset.
///
/// Each inner `Vec<f64>` corresponds to one data point and contains one or more
/// filter values (1 for scalar filters, d for multi-dimensional filters).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterValues {
    /// One Vec<f64> per data point.
    pub values: Vec<Vec<f64>>,
    /// Name of the filter that produced these values.
    pub filter_name: String,
}

impl FilterValues {
    /// Apply a filter function to every point in the dataset.
    pub fn from_filter(data: &[DataPoint], filter: &FilterFunction) -> Self {
        let values: Vec<Vec<f64>> = (0..data.len()).map(|i| filter.apply(data, i)).collect();
        Self {
            values,
            filter_name: filter.name.clone(),
        }
    }

    /// Dimensionality of the filter output (number of values per point).
    pub fn dimensionality(&self) -> usize {
        self.values.first().map_or(0, |v| v.len())
    }

    /// For 1D filters: return (min, max) of all filter values.
    pub fn range_1d(&self) -> (f64, f64) {
        let scalars: Vec<f64> = self.values.iter().map(|v| v[0]).collect();
        let min = scalars.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scalars.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ─── Built-in filters ──────────────────────────────────────────────────────

/// PCA projection onto the first principal component.
///
/// Uses the iterative power method to approximate the dominant eigenvector,
/// then projects each point onto it. No external linear algebra crate needed.
///
/// # Complexity
/// O(n · d · iterations) where n = number of points, d = dimensionality.
pub fn pca_projection_filter(iterations: usize) -> FilterFunction {
    FilterFunction::new("pca_projection", move |data, _index| {
        if data.is_empty() || data[0].is_empty() {
            return vec![0.0];
        }
        let dim = data[0].len();
        // Power method: find dominant eigenvector of covariance matrix
        let mut v: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0) / (dim as f64)).collect();
        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }

        // Compute mean
        let mean = compute_centroid(data);

        // Centered data as covariance-vector products
        for _ in 0..iterations {
            // v = (1/n) * Σ (x_i - μ)(x_i - μ)^T v
            let mut new_v = vec![0.0; dim];
            for point in data {
                let centered: Vec<f64> =
                    point.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();
                let dot: f64 = centered.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                for (j, nv) in new_v.iter_mut().enumerate() {
                    *nv += centered[j] * dot;
                }
            }
            let n = data.len() as f64;
            for x in new_v.iter_mut() {
                *x /= n;
            }
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in new_v.iter_mut() {
                    *x /= norm;
                }
            }
            v = new_v;
        }

        // Project each point — but this closure is called per-point.
        // We need to recompute the eigenvector each time (suboptimal but correct).
        // Instead, let's project just the requested point using the eigenvector.
        // We recompute v each call, which is expensive but stateless.
        // For PCA, we just project point[_index] onto v.
        let point = &data[_index];
        let centered: Vec<f64> = point.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();
        let projection: f64 = centered.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        vec![projection]
    })
}

/// Distance from the dataset centroid.
///
/// For each point, computes the L² (Euclidean) distance to the mean of the dataset.
///
/// # Complexity
/// O(n · d) for centroid computation, O(d) per point for distance.
pub fn distance_from_centroid_filter() -> FilterFunction {
    FilterFunction::new("distance_from_centroid", |data, index| {
        let centroid = compute_centroid(data);
        let dist = euclidean_distance(&data[index], &centroid);
        vec![dist]
    })
}

/// Eccentricity filter: mean distance from each point to all others.
///
/// Points in dense regions have low eccentricity; outliers have high eccentricity.
///
/// # Complexity
/// O(n² · d) for the full dataset.
pub fn eccentricity_filter() -> FilterFunction {
    FilterFunction::new("eccentricity", |data, index| {
        if data.len() <= 1 {
            return vec![0.0];
        }
        let mut total_dist = 0.0;
        for (_j, point) in data.iter().enumerate() {
            if _j != index {
                total_dist += euclidean_distance(&data[index], point);
            }
        }
        vec![total_dist / (data.len() - 1) as f64]
    })
}

/// Kernel density estimate (Gaussian kernel).
///
/// Evaluates a Gaussian KDE at each point. High values indicate dense regions.
/// The `bandwidth` parameter controls the smoothing width.
///
/// # Complexity
/// O(n² · d) for the full dataset.
pub fn kde_filter(bandwidth: f64) -> FilterFunction {
    FilterFunction::new("kde", move |data, index| {
        if data.is_empty() {
            return vec![0.0];
        }
        let dim = data[0].len() as f64;
        let h = if bandwidth <= 0.0 { 1.0 } else { bandwidth };
        let mut density = 0.0;
        for point in data.iter() {
            let dist_sq = euclidean_distance_sq(&data[index], point);
            density += (-dist_sq / (2.0 * h * h)).exp();
        }
        // Normalize by n * (2π)^(d/2) * h^d
        let n = data.len() as f64;
        let norm = n * (2.0 * std::f64::consts::PI).powf(dim / 2.0) * h.powf(dim);
        vec![density / norm]
    })
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Compute the centroid (mean) of a set of data points.
pub fn compute_centroid(data: &[DataPoint]) -> DataPoint {
    if data.is_empty() {
        return vec![];
    }
    let dim = data[0].len();
    let mut centroid = vec![0.0; dim];
    for point in data {
        for (i, val) in point.iter().enumerate() {
            centroid[i] += val;
        }
    }
    let n = data.len() as f64;
    for val in centroid.iter_mut() {
        *val /= n;
    }
    centroid
}

/// Euclidean distance between two points.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    euclidean_distance_sq(a, b).sqrt()
}

/// Squared Euclidean distance (avoids sqrt when only comparisons needed).
pub fn euclidean_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Precompute all pairwise Euclidean distances into a flat distance matrix.
///
/// Returns a Vec<Vec<f64>> where `dist[i][j]` is the distance between points i and j.
pub fn pairwise_distances(data: &[DataPoint]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&data[i], &data[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<DataPoint> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![5.0, 5.0],
        ]
    }

    #[test]
    fn test_centroid() {
        let data = sample_data();
        let c = compute_centroid(&data);
        assert_eq!(c.len(), 2);
        assert!((c[0] - 7.0 / 5.0).abs() < 1e-10);
        assert!((c[1] - 7.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_distances() {
        let data = vec![vec![0.0], vec![3.0]];
        let dist = pairwise_distances(&data);
        assert!((dist[0][1] - 3.0).abs() < 1e-10);
        assert!((dist[1][0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_from_centroid() {
        let data = sample_data();
        let filter = distance_from_centroid_filter();
        let values = FilterValues::from_filter(&data, &filter);
        assert_eq!(values.len(), 5);
        assert_eq!(values.dimensionality(), 1);
        // Point [5,5] is furthest from centroid
        let last = values.values[4][0];
        let first = values.values[0][0];
        assert!(last > first);
    }

    #[test]
    fn test_eccentricity() {
        let data = sample_data();
        let filter = eccentricity_filter();
        let values = FilterValues::from_filter(&data, &filter);
        // Point [5,5] is an outlier — should have highest eccentricity
        let outlier_ecc = values.values[4][0];
        for v in &values.values[..4] {
            assert!(outlier_ecc > v[0]);
        }
    }

    #[test]
    fn test_kde_filter() {
        let data = sample_data();
        let filter = kde_filter(1.0);
        let values = FilterValues::from_filter(&data, &filter);
        assert_eq!(values.len(), 5);
        // Denser region (first 4 points near origin) should have higher KDE
        let origin_density = values.values[0][0];
        let outlier_density = values.values[4][0];
        assert!(origin_density > outlier_density);
    }

    #[test]
    fn test_pca_projection() {
        let data = sample_data();
        let filter = pca_projection_filter(50);
        let values = FilterValues::from_filter(&data, &filter);
        assert_eq!(values.len(), 5);
        assert_eq!(values.dimensionality(), 1);
    }

    #[test]
    fn test_filter_values_range() {
        let data = sample_data();
        let filter = distance_from_centroid_filter();
        let values = FilterValues::from_filter(&data, &filter);
        let (min, max) = values.range_1d();
        assert!(min <= max);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<DataPoint> = vec![];
        let filter = distance_from_centroid_filter();
        let values = FilterValues::from_filter(&data, &filter);
        assert!(values.is_empty());
    }
}
