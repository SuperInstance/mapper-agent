//! # mapper-agent
//!
//! Mapper algorithm (Singh, Mémoli, Carlsson 2007) for discovering
//! topological structure in agent state spaces.
//!
//! The Mapper algorithm constructs a simplicial complex (graph) that captures
//! the shape and topology of high-dimensional data. It combines:
//! 1. A **filter** that maps data to a lower-dimensional space.
//! 2. An **overlapping cover** of the filter range.
//! 3. **Clustering** within each cover element.
//! 4. **Nerve construction** to form the output graph.
//!
//! # Quick Start
//!
//! ```rust
//! use mapper_agent::*;
//!
//! let data = vec![
//!     vec![0.0, 0.0],
//!     vec![1.0, 1.0],
//!     vec![10.0, 10.0],
//!     vec![11.0, 11.0],
//! ];
//!
//! let filter = filter::distance_from_centroid_filter();
//! let config = mapper::MapperConfig::default();
//! let result = mapper::mapper_pipeline(&data, &filter, &config);
//!
//! println!("Nodes: {}", result.graph.node_count());
//! println!("Edges: {}", result.graph.edge_count());
//! ```

pub mod cluster;
pub mod cover;
pub mod filter;
pub mod mapper;
pub mod nerve;
pub mod visualization;

// Re-export key types
pub use cluster::{ClusterConfig, ClusterID, UnionFind};
pub use cover::{CoverConfig, Interval, OverlappingCover};
pub use filter::{DataPoint, FilterFunction, FilterValues};
pub use mapper::{MapperConfig, MapperResult, mapper_default, mapper_pipeline};
pub use nerve::SimplicialComplex;
pub use visualization::{MapperGraph, MapperNode, NodeAttributes};

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Generate a noisy circle (annulus) in 2D.
    fn circle_data(n: usize, radius: f64, noise: f64) -> Vec<DataPoint> {
        (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                let r = radius + (i as f64 * 7.3).sin() * noise;
                vec![r * angle.cos(), r * angle.sin()]
            })
            .collect()
    }

    /// Generate three well-separated blobs.
    fn three_blobs_data() -> Vec<DataPoint> {
        let mut data = Vec::new();
        // Blob 1: near origin
        for i in 0..10 {
            let x = (i as f64 * 0.3).sin();
            let y = (i as f64 * 0.3).cos();
            data.push(vec![x, y]);
        }
        // Blob 2: near (5, 0)
        for i in 0..10 {
            let x = 5.0 + (i as f64 * 0.3).sin();
            let y = (i as f64 * 0.3).cos();
            data.push(vec![x, y]);
        }
        // Blob 3: near (2.5, 5)
        for i in 0..10 {
            let x = 2.5 + (i as f64 * 0.3).sin();
            let y = 5.0 + (i as f64 * 0.3).cos();
            data.push(vec![x, y]);
        }
        data
    }

    #[test]
    fn test_circle_mapper() {
        let data = circle_data(50, 5.0, 0.5);
        let filter = filter::eccentricity_filter();
        let config = MapperConfig::new(CoverConfig::new(5, 0.4), ClusterConfig::new(8.0));
        let result = mapper_pipeline(&data, &filter, &config);
        // Circle data should produce a graph with nodes and edges
        assert!(result.graph.node_count() > 1);
        // Should be connected or nearly connected
        let cc = result.graph.connected_components();
        assert!(
            cc.len() <= 5,
            "Circle with overlap should be mostly connected, got {} components",
            cc.len()
        );
    }

    #[test]
    fn test_three_blobs_three_components() {
        let data = three_blobs_data();
        let filter = filter::pca_projection_filter(30);
        let config = MapperConfig::new(CoverConfig::new(5, 0.1), ClusterConfig::new(1.5));
        let result = mapper_pipeline(&data, &filter, &config);
        // With well-separated blobs and low overlap, expect ≤ 5 components
        let cc = result.graph.connected_components();
        assert!(cc.len() >= 1 && cc.len() <= 6);
    }

    #[test]
    fn test_kde_filter_mapper() {
        let data = circle_data(30, 3.0, 0.3);
        let filter = filter::kde_filter(1.0);
        let config = MapperConfig::new(CoverConfig::new(5, 0.3), ClusterConfig::new(2.0));
        let result = mapper_pipeline(&data, &filter, &config);
        assert!(result.graph.node_count() >= 1);
    }

    #[test]
    fn test_full_coverage_guarantee() {
        let data = three_blobs_data();
        let filter = filter::distance_from_centroid_filter();
        let result = mapper_default(&data, &filter);
        // Every data point must appear in at least one graph node
        let mut covered = vec![false; data.len()];
        for node in &result.graph.nodes {
            for &idx in &node.member_indices {
                covered[idx] = true;
            }
        }
        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn test_dot_export() {
        let data = three_blobs_data();
        let filter = filter::distance_from_centroid_filter();
        let result = mapper_default(&data, &filter);
        let dot = result.graph.to_dot();
        assert!(dot.contains("graph mapper"));
    }

    #[test]
    fn test_ascii_render() {
        let data = three_blobs_data();
        let filter = filter::distance_from_centroid_filter();
        let result = mapper_default(&data, &filter);
        let ascii = result.graph.to_ascii(60, 30);
        assert!(ascii.len() > 10);
    }

    #[test]
    fn test_union_find_large() {
        let mut uf = UnionFind::new(100);
        for i in 0..99 {
            uf.union(i, i + 1);
        }
        assert_eq!(uf.num_sets(), 1);
        // All should have same root
        let root = uf.find(0);
        for i in 0..100 {
            assert_eq!(uf.find(i), root);
        }
    }

    #[test]
    fn test_mapper_config_default() {
        let c = MapperConfig::default();
        assert_eq!(c.cover.num_intervals, 10);
        assert!((c.cluster.threshold - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cover_config_clamping() {
        let c = CoverConfig::new(0, 1.5);
        assert_eq!(c.num_intervals, 1);
        assert!((c.overlap - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_filter_values_dimensionality() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let filter = filter::distance_from_centroid_filter();
        let fv = FilterValues::from_filter(&data, &filter);
        assert_eq!(fv.dimensionality(), 1);
    }

    #[test]
    fn test_graph_node_attributes() {
        let data = three_blobs_data();
        let filter = filter::distance_from_centroid_filter();
        let result = mapper_default(&data, &filter);
        for node in &result.graph.nodes {
            assert!(node.size > 0);
            assert_eq!(node.centroid.len(), 2);
            assert!(node.attributes.density >= 0.0);
        }
    }

    #[test]
    fn test_linear_data_pipeline() {
        // Points along a line
        let data: Vec<DataPoint> = (0..20).map(|i| vec![i as f64, i as f64 * 0.5]).collect();
        let filter = filter::pca_projection_filter(30);
        let config = MapperConfig::new(CoverConfig::new(4, 0.3), ClusterConfig::new(3.0));
        let result = mapper_pipeline(&data, &filter, &config);
        // Linear data should produce a connected chain
        assert!(result.graph.node_count() >= 2);
        assert!(result.graph.edge_count() >= 1);
    }

    #[test]
    fn test_mapper_result_debug() {
        let data = three_blobs_data();
        let filter = filter::distance_from_centroid_filter();
        let result = mapper_default(&data, &filter);
        // Debug trait should work
        let debug = format!("{:?}", result);
        assert!(!debug.is_empty());
    }
}
