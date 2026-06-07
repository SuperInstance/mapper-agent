//! Overlapping cover for the Mapper algorithm.
//!
//! After applying a filter function, the filter values are divided into
//! overlapping intervals. Each interval defines a "bin" that pulls in
//! data points whose filter values fall within its range.
//!
//! # Overlap
//!
//! Adjacent intervals share a fraction of their range (the overlap percentage).
//! This ensures that points near interval boundaries appear in multiple bins,
//! creating the connections that form the Mapper graph's edges.

use crate::filter::FilterValues;
use serde::{Deserialize, Serialize};

/// Configuration for constructing an overlapping cover.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoverConfig {
    /// Number of intervals to divide the filter range into.
    pub num_intervals: usize,
    /// Overlap fraction between adjacent intervals (0.0 to 1.0).
    /// Typical values: 0.1–0.3 (10–30%).
    pub overlap: f64,
}

impl Default for CoverConfig {
    fn default() -> Self {
        Self {
            num_intervals: 10,
            overlap: 0.2,
        }
    }
}

impl CoverConfig {
    pub fn new(num_intervals: usize, overlap: f64) -> Self {
        Self {
            num_intervals: num_intervals.max(1),
            overlap: overlap.clamp(0.0, 0.99),
        }
    }
}

/// A single interval in the overlapping cover.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Interval {
    /// Index of this interval in the cover.
    pub index: usize,
    /// Start of the interval (inclusive).
    pub start: f64,
    /// End of the interval (inclusive).
    pub end: f64,
    /// Overlap region with the previous interval.
    pub overlap_left: f64,
    /// Overlap region with the next interval.
    pub overlap_right: f64,
    /// Indices of data points whose filter values fall in this interval.
    pub point_indices: Vec<usize>,
}

impl Interval {
    /// Check if a filter value falls within this interval.
    pub fn contains(&self, value: f64) -> bool {
        value >= self.start && value <= self.end
    }

    /// Width of the interval.
    pub fn width(&self) -> f64 {
        self.end - self.start
    }
}

/// An overlapping cover of the filter value range.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlappingCover {
    /// Configuration used to create this cover.
    pub config: CoverConfig,
    /// The intervals.
    pub intervals: Vec<Interval>,
    /// Minimum filter value covered.
    pub filter_min: f64,
    /// Maximum filter value covered.
    pub filter_max: f64,
}

impl OverlappingCover {
    /// Construct an overlapping cover from 1D filter values.
    ///
    /// For multi-dimensional filters, this uses only the first component.
    pub fn from_filter_values(filter_values: &FilterValues, config: &CoverConfig) -> Self {
        if filter_values.is_empty() {
            return Self {
                config: config.clone(),
                intervals: vec![],
                filter_min: 0.0,
                filter_max: 0.0,
            };
        }

        let (fmin, fmax) = filter_values.range_1d();
        let n = config.num_intervals;
        let overlap = config.overlap;

        // Compute interval width and step
        let range = fmax - fmin;
        if range < 1e-15 {
            // All filter values are the same — single interval
            let interval = Interval {
                index: 0,
                start: fmin,
                end: fmax,
                overlap_left: 0.0,
                overlap_right: 0.0,
                point_indices: (0..filter_values.len()).collect(),
            };
            return Self {
                config: config.clone(),
                intervals: vec![interval],
                filter_min: fmin,
                filter_max: fmax,
            };
        }

        // Each interval has width w. Step between starts = w * (1 - overlap).
        // n * w * (1 - overlap) + w * overlap = range
        // w * (n * (1 - overlap) + overlap) = range
        // w = range / (n * (1 - overlap) + overlap)
        let w = range / (n as f64 * (1.0 - overlap) + overlap);
        let step = w * (1.0 - overlap);

        let mut intervals = Vec::with_capacity(n);
        for i in 0..n {
            let start = fmin + i as f64 * step;
            let mut end = start + w;
            // Extend last interval to cover fmax exactly
            if i == n - 1 {
                end = fmax;
            }
            let _ol = if i > 0 {
                start - (fmin + (i - 1) as f64 * step + w)
            } else {
                0.0
            };
            // overlap_right is computed implicitly by the next interval's start < this end
            intervals.push(Interval {
                index: i,
                start,
                end,
                overlap_left: if i > 0 {
                    (fmin + (i - 1) as f64 * step + w) - start
                } else {
                    0.0
                },
                overlap_right: if i < n - 1 {
                    end - (fmin + (i + 1) as f64 * step)
                } else {
                    0.0
                },
                point_indices: vec![],
            });
        }

        // Assign points to intervals
        for (pt_idx, vals) in filter_values.values.iter().enumerate() {
            let v = vals[0]; // 1D
            for interval in &mut intervals {
                if interval.contains(v) {
                    interval.point_indices.push(pt_idx);
                }
            }
        }

        Self {
            config: config.clone(),
            intervals,
            filter_min: fmin,
            filter_max: fmax,
        }
    }

    /// Verify that every data point is in at least one interval.
    pub fn verify_coverage(&self, num_points: usize) -> bool {
        let mut covered = vec![false; num_points];
        for interval in &self.intervals {
            for &idx in &interval.point_indices {
                if idx < num_points {
                    covered[idx] = true;
                }
            }
        }
        covered.iter().all(|&c| c)
    }

    /// Number of intervals.
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Whether the cover is empty.
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{DataPoint, FilterFunction, FilterValues};

    fn simple_filter_values() -> FilterValues {
        // 10 points, filter values 0.0 to 9.0
        let data: Vec<DataPoint> = (0..10).map(|i| vec![i as f64]).collect();
        let filter = FilterFunction::new("identity", |data, i| vec![data[i][0]]);
        FilterValues::from_filter(&data, &filter)
    }

    #[test]
    fn test_cover_basic() {
        let fv = simple_filter_values();
        let config = CoverConfig::new(3, 0.3); // More overlap
        let cover = OverlappingCover::from_filter_values(&fv, &config);
        assert_eq!(cover.len(), 3);
        assert!(
            cover.verify_coverage(10),
            "Every point should be covered: {:?}",
            cover
                .intervals
                .iter()
                .map(|i| (i.start, i.end, &i.point_indices))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_cover_overlap() {
        let fv = simple_filter_values();
        let config = CoverConfig::new(5, 0.3);
        let cover = OverlappingCover::from_filter_values(&fv, &config);
        // With 30% overlap, some points should be in multiple intervals
        let total_assignments: usize = cover.intervals.iter().map(|i| i.point_indices.len()).sum();
        assert!(
            total_assignments > 10,
            "With overlap, total assignments should exceed point count"
        );
    }

    #[test]
    fn test_cover_single_interval() {
        // All points have same filter value
        let data: Vec<DataPoint> = (0..5).map(|_| vec![1.0]).collect();
        let filter = FilterFunction::new("const", |_data, _i| vec![1.0]);
        let fv = FilterValues::from_filter(&data, &filter);
        let config = CoverConfig::new(5, 0.2);
        let cover = OverlappingCover::from_filter_values(&fv, &config);
        assert_eq!(cover.len(), 1);
        assert!(cover.verify_coverage(5));
    }

    #[test]
    fn test_cover_empty() {
        let fv = FilterValues {
            values: vec![],
            filter_name: "test".into(),
        };
        let config = CoverConfig::new(5, 0.2);
        let cover = OverlappingCover::from_filter_values(&fv, &config);
        assert!(cover.is_empty());
    }

    #[test]
    fn test_interval_contains() {
        let interval = Interval {
            index: 0,
            start: 0.0,
            end: 5.0,
            overlap_left: 0.0,
            overlap_right: 1.0,
            point_indices: vec![],
        };
        assert!(interval.contains(0.0));
        assert!(interval.contains(2.5));
        assert!(interval.contains(5.0));
        assert!(!interval.contains(-0.1));
        assert!(!interval.contains(5.1));
    }

    #[test]
    fn test_interval_width() {
        let interval = Interval {
            index: 0,
            start: 1.0,
            end: 3.5,
            overlap_left: 0.0,
            overlap_right: 0.0,
            point_indices: vec![],
        };
        assert!((interval.width() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_cover_config_default() {
        let c = CoverConfig::default();
        assert_eq!(c.num_intervals, 10);
        assert!((c.overlap - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_verification_fails() {
        // Manually construct a cover with missing points
        let cover = OverlappingCover {
            config: CoverConfig::new(2, 0.1),
            intervals: vec![Interval {
                index: 0,
                start: 0.0,
                end: 5.0,
                overlap_left: 0.0,
                overlap_right: 0.0,
                point_indices: vec![0, 1], // only covers 2 of 10 points
            }],
            filter_min: 0.0,
            filter_max: 10.0,
        };
        assert!(!cover.verify_coverage(10));
    }
}
