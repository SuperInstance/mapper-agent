//! Clustering within cover intervals.
//!
//! Points that fall in the same cover interval are clustered using
//! single-linkage hierarchical clustering. Two clusters merge when
//! the minimum distance between any pair of points (one from each cluster)
//! falls below a threshold δ.
//!
//! # Union-Find
//!
//! The implementation uses an iterative union-find data structure with
//! path compression and union by rank. This avoids recursion and provides
//! near-amortized-constant-time operations.

use crate::filter::{DataPoint, euclidean_distance};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifier for a cluster: (interval_index, cluster_index_within_interval).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterID {
    pub interval: usize,
    pub cluster: usize,
}

impl ClusterID {
    pub fn new(interval: usize, cluster: usize) -> Self {
        Self { interval, cluster }
    }
}

/// Result of clustering within a single cover interval.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntervalClusters {
    /// Index of the cover interval.
    pub interval_index: usize,
    /// Clusters: each is a Vec of point indices.
    pub clusters: Vec<Vec<usize>>,
}

/// Configuration for the clustering step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Distance threshold δ for single-linkage merging.
    pub threshold: f64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self { threshold: 1.0 }
    }
}

impl ClusterConfig {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.max(0.0),
        }
    }
}

// ─── Union-Find (iterative, with path compression) ─────────────────────────

/// Iterative union-find with path compression and union by rank.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the root of x iteratively with path compression.
    pub fn find(&mut self, mut x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression
        while self.parent[x] != root {
            let next = self.parent[x];
            self.parent[x] = root;
            x = next;
        }
        root
    }

    /// Union two sets. Returns true if they were separate.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        // Union by rank
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
        true
    }

    /// Number of distinct sets.
    pub fn num_sets(&self) -> usize {
        let mut roots = vec![false; self.parent.len()];
        for i in 0..self.parent.len() {
            let mut r = i;
            while self.parent[r] != r {
                r = self.parent[r];
            }
            roots[r] = true;
        }
        roots.iter().filter(|&&b| b).count()
    }
}

// ─── Single-linkage clustering ─────────────────────────────────────────────

/// Cluster points within a single cover interval using single-linkage.
///
/// Two points are in the same cluster if there exists a chain of points
/// where each consecutive pair has distance ≤ threshold.
pub fn cluster_points(points: &[usize], data: &[DataPoint], threshold: f64) -> Vec<Vec<usize>> {
    if points.is_empty() {
        return vec![];
    }
    if points.len() == 1 {
        return vec![points.to_vec()];
    }

    let n = points.len();
    let mut uf = UnionFind::new(n);

    // Check all pairs; merge if distance ≤ threshold
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(&data[points[i]], &data[points[j]]);
            if dist <= threshold {
                uf.union(i, j);
            }
        }
    }

    // Collect clusters
    let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let root = uf.find(i);
        cluster_map.entry(root).or_default().push(points[i]);
    }

    cluster_map.into_values().collect()
}

/// Cluster all points across all cover intervals.
///
/// Returns one `IntervalClusters` per interval that has at least one cluster.
pub fn cluster_all_intervals(
    interval_point_sets: &[Vec<usize>],
    data: &[DataPoint],
    config: &ClusterConfig,
) -> Vec<IntervalClusters> {
    interval_point_sets
        .iter()
        .enumerate()
        .map(|(interval_idx, points)| {
            let clusters = cluster_points(points, data, config.threshold);
            IntervalClusters {
                interval_index: interval_idx,
                clusters,
            }
        })
        .filter(|ic| !ic.clusters.is_empty())
        .collect()
}

/// Build a map from ClusterID to the set of point indices in that cluster.
pub fn build_cluster_map(all_clusters: &[IntervalClusters]) -> HashMap<ClusterID, Vec<usize>> {
    let mut map = HashMap::new();
    for ic in all_clusters {
        for (cidx, points) in ic.clusters.iter().enumerate() {
            map.insert(ClusterID::new(ic.interval_index, cidx), points.clone());
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<DataPoint> {
        vec![
            vec![0.0, 0.0], // 0
            vec![0.5, 0.5], // 1
            vec![1.0, 1.0], // 2
            vec![5.0, 5.0], // 3 — far from others
            vec![5.5, 5.5], // 4 — close to 3
        ]
    }

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.num_sets(), 5);
        assert!(uf.union(0, 1));
        assert!(!uf.union(0, 1)); // already merged
        assert_eq!(uf.num_sets(), 4);
    }

    #[test]
    fn test_union_find_path_compression() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);
        // Path compression should flatten
        let root = uf.find(3);
        assert_eq!(uf.find(0), root);
        assert_eq!(uf.find(1), root);
        assert_eq!(uf.find(2), root);
    }

    #[test]
    fn test_cluster_two_groups() {
        let data = sample_data();
        let points = vec![0, 1, 2, 3, 4];
        let clusters = cluster_points(&points, &data, 2.0);
        // Points 0,1,2 should cluster together; 3,4 together
        assert_eq!(clusters.len(), 2);
        for c in &clusters {
            if c.contains(&0) {
                assert!(c.contains(&1));
                assert!(c.contains(&2));
            }
            if c.contains(&3) {
                assert!(c.contains(&4));
            }
        }
    }

    #[test]
    fn test_cluster_single_point() {
        let data = sample_data();
        let clusters = cluster_points(&[0], &data, 1.0);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec![0]);
    }

    #[test]
    fn test_cluster_empty() {
        let clusters = cluster_points(&[], &sample_data(), 1.0);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_all_merged() {
        let data = sample_data();
        let clusters = cluster_points(&[0, 1, 2, 3, 4], &data, 100.0);
        // All points within threshold of each other (via chains)
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 5);
    }

    #[test]
    fn test_cluster_all_intervals() {
        let data = sample_data();
        let intervals: Vec<Vec<usize>> = vec![
            vec![0, 1, 2], // cluster 1
            vec![3, 4],    // cluster 2
        ];
        let config = ClusterConfig::new(2.0);
        let results = cluster_all_intervals(&intervals, &data, &config);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_build_cluster_map() {
        let data = sample_data();
        let intervals: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![3, 4]];
        let config = ClusterConfig::new(2.0);
        let results = cluster_all_intervals(&intervals, &data, &config);
        let map = build_cluster_map(&results);
        assert!(!map.is_empty());
        // Each cluster should have a ClusterID
        for (id, points) in &map {
            assert!(!points.is_empty());
            assert_eq!(id.cluster, 0); // only one cluster per interval in this case
        }
    }

    #[test]
    fn test_cluster_id() {
        let id = ClusterID::new(3, 7);
        assert_eq!(id.interval, 3);
        assert_eq!(id.cluster, 7);
    }
}
