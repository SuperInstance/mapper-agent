//! Mapper algorithm pipeline.
//!
//! The Mapper algorithm (Singh, Mémoli, Carlsson 2007) constructs a simplicial
//! complex that captures the topological structure of a dataset:
//!
//! ```text
//! M(X, f, U) = Nerve(Cluster(f⁻¹(U_i) ∩ X))
//! ```
//!
//! Steps:
//! 1. Apply filter function f to obtain filter values.
//! 2. Create overlapping cover U of the filter range.
//! 3. Cluster points within each cover element.
//! 4. Build the nerve of the resulting clusters.

use crate::cluster::{ClusterConfig, IntervalClusters, build_cluster_map, cluster_all_intervals};
use crate::cover::{CoverConfig, OverlappingCover};
use crate::filter::{DataPoint, FilterFunction, FilterValues};
use crate::nerve::{SimplicialComplex, build_nerve};
use crate::visualization::MapperGraph;
use serde::{Deserialize, Serialize};

/// Configuration for the full Mapper pipeline.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MapperConfig {
    /// Cover configuration.
    pub cover: CoverConfig,
    /// Clustering configuration.
    pub cluster: ClusterConfig,
}

impl MapperConfig {
    pub fn new(cover: CoverConfig, cluster: ClusterConfig) -> Self {
        Self { cover, cluster }
    }
}

/// Result of running the Mapper pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapperResult {
    /// The filter values computed in step 1.
    pub filter_values: FilterValues,
    /// The overlapping cover from step 2.
    pub cover: OverlappingCover,
    /// The clusters from step 3.
    pub clusters: Vec<IntervalClusters>,
    /// The nerve (simplicial complex) from step 4.
    pub nerve: SimplicialComplex,
    /// The mapper graph (visualization-friendly).
    pub graph: MapperGraph,
    /// Number of input data points.
    pub num_points: usize,
}

/// Run the full Mapper pipeline.
///
/// # Arguments
/// * `data` — Input data points.
/// * `filter` — Filter function to apply.
/// * `config` — Pipeline configuration (cover + clustering).
///
/// # Returns
/// A `MapperResult` containing all intermediate results and the final graph.
pub fn mapper_pipeline(
    data: &[DataPoint],
    filter: &FilterFunction,
    config: &MapperConfig,
) -> MapperResult {
    // Step 1: Apply filter
    let filter_values = FilterValues::from_filter(data, filter);

    // Step 2: Create overlapping cover
    let cover = OverlappingCover::from_filter_values(&filter_values, &config.cover);

    // Step 3: Cluster within each interval
    let interval_point_sets: Vec<Vec<usize>> = cover
        .intervals
        .iter()
        .map(|interval| interval.point_indices.clone())
        .collect();
    let clusters = cluster_all_intervals(&interval_point_sets, data, &config.cluster);

    // Step 4: Build nerve
    let cluster_points = build_cluster_map(&clusters);
    let nerve = build_nerve(&cluster_points);

    // Build mapper graph
    let graph = MapperGraph::from_nerve(&nerve, data);

    MapperResult {
        filter_values,
        cover,
        clusters,
        nerve,
        graph,
        num_points: data.len(),
    }
}

/// Convenience function: run Mapper with default configuration.
pub fn mapper_default(data: &[DataPoint], filter: &FilterFunction) -> MapperResult {
    mapper_pipeline(data, filter, &MapperConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::ClusterConfig;
    use crate::cover::CoverConfig;
    use crate::filter::{distance_from_centroid_filter, pca_projection_filter};

    /// Generate two well-separated Gaussian-ish clusters in 2D.
    fn two_cluster_data() -> Vec<DataPoint> {
        let mut data = Vec::new();
        // Cluster A: near (0, 0)
        for i in 0..15 {
            let t = i as f64 * 0.2;
            data.push(vec![t.sin() * 0.5, t.cos() * 0.5]);
        }
        // Cluster B: near (10, 10)
        for i in 0..15 {
            let t = i as f64 * 0.2;
            data.push(vec![10.0 + t.sin() * 0.5, 10.0 + t.cos() * 0.5]);
        }
        data
    }

    #[test]
    fn test_pipeline_basic() {
        let data = two_cluster_data();
        let filter = distance_from_centroid_filter();
        let config = MapperConfig::new(CoverConfig::new(5, 0.3), ClusterConfig::new(2.0));
        let result = mapper_pipeline(&data, &filter, &config);
        assert!(result.num_points == 30);
        assert!(!result.graph.nodes.is_empty());
    }

    #[test]
    fn test_pipeline_default_config() {
        let data = two_cluster_data();
        let filter = pca_projection_filter(30);
        let result = mapper_default(&data, &filter);
        assert_eq!(result.num_points, 30);
    }

    #[test]
    fn test_pipeline_single_point() {
        let data = vec![vec![0.0, 0.0]];
        let filter = distance_from_centroid_filter();
        let config = MapperConfig::default();
        let result = mapper_pipeline(&data, &filter, &config);
        assert_eq!(result.num_points, 1);
        assert_eq!(result.graph.nodes.len(), 1);
    }

    #[test]
    fn test_pipeline_empty() {
        let data: Vec<DataPoint> = vec![];
        let filter = distance_from_centroid_filter();
        let config = MapperConfig::default();
        let result = mapper_pipeline(&data, &filter, &config);
        assert_eq!(result.num_points, 0);
        assert!(result.graph.nodes.is_empty());
    }

    #[test]
    fn test_pipeline_identical_points() {
        let data: Vec<DataPoint> = (0..10).map(|_| vec![1.0, 1.0]).collect();
        let filter = distance_from_centroid_filter();
        let config = MapperConfig::new(CoverConfig::new(3, 0.2), ClusterConfig::new(1.0));
        let result = mapper_pipeline(&data, &filter, &config);
        // All points identical → single cluster
        assert!(result.graph.nodes.len() <= 1);
    }

    #[test]
    fn test_pipeline_preserves_data_count() {
        let data = two_cluster_data();
        let filter = distance_from_centroid_filter();
        let result = mapper_default(&data, &filter);
        // All data points should be in some cluster
        let mut covered: Vec<bool> = vec![false; data.len()];
        for node in &result.graph.nodes {
            for &idx in &node.member_indices {
                covered[idx] = true;
            }
        }
        assert!(
            covered.iter().all(|&c| c),
            "Every data point must appear in at least one node"
        );
    }

    #[test]
    fn test_pipeline_two_clusters_separate() {
        let data = two_cluster_data();
        let filter = pca_projection_filter(50);
        let config = MapperConfig::new(CoverConfig::new(3, 0.1), ClusterConfig::new(1.0));
        let result = mapper_pipeline(&data, &filter, &config);
        // With well-separated clusters and low overlap, we expect ≥ 2 nodes
        assert!(result.graph.nodes.len() >= 2);
    }
}
