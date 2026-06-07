//! Nerve construction for the Mapper algorithm.
//!
//! The nerve of a cover is a simplicial complex where:
//! - Each cluster (within a cover interval) becomes a vertex.
//! - Two clusters that share data points are connected by an edge.
//! - Higher-order overlaps (3+ clusters sharing points) form higher simplices.
//!
//! By the Nerve Theorem, the nerve is homotopy equivalent to the union of
//! the cover elements, provided all finite intersections are contractible.
//!
//! Nerve(U) ≃ ∪U  when all finite intersections are contractible.

use crate::cluster::ClusterID;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A simplicial complex: a collection of simplices (subsets of vertices).
///
/// Represented as `Vec<Vec<usize>>` where each inner vec is a simplex
/// (sorted, vertex indices).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimplicialComplex {
    /// Vertices, indexed by usize.
    pub vertices: Vec<VertexInfo>,
    /// Simplices: each is a sorted Vec<usize> of vertex indices.
    /// 0-simplices (vertices) are implicit from `vertices`.
    /// 1-simplices are edges, 2-simplices are triangles, etc.
    pub simplices: Vec<Vec<usize>>,
}

/// Metadata for a vertex in the nerve complex.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VertexInfo {
    /// The cluster this vertex represents.
    pub cluster_id: ClusterID,
    /// Data point indices belonging to this cluster.
    pub point_indices: Vec<usize>,
}

/// Build the nerve of the cover from clusters and their point memberships.
///
/// # Arguments
/// * `cluster_points` — Map from ClusterID to the set of point indices.
///
/// # Returns
/// A `SimplicialComplex` where vertices are clusters and simplices represent
/// overlapping point sets.
pub fn build_nerve(cluster_points: &HashMap<ClusterID, Vec<usize>>) -> SimplicialComplex {
    if cluster_points.is_empty() {
        return SimplicialComplex {
            vertices: vec![],
            simplices: vec![],
        };
    }

    // Assign vertex indices
    let mut id_to_vertex: HashMap<ClusterID, usize> = HashMap::new();
    let mut vertices: Vec<VertexInfo> = Vec::new();
    for (i, (cid, points)) in cluster_points.iter().enumerate() {
        id_to_vertex.insert(*cid, i);
        vertices.push(VertexInfo {
            cluster_id: *cid,
            point_indices: points.clone(),
        });
    }

    let _n = vertices.len();

    // Build point → set of vertex indices that contain it
    let mut point_to_vertices: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (vi, info) in vertices.iter().enumerate() {
        for &pt in &info.point_indices {
            point_to_vertices.entry(pt).or_default().insert(vi);
        }
    }

    // Collect all maximal simplices from shared points
    let mut simplex_set: HashSet<Vec<usize>> = HashSet::new();

    for vertex_set in point_to_vertices.values() {
        if vertex_set.len() >= 2 {
            let mut simplex: Vec<usize> = vertex_set.iter().cloned().collect();
            simplex.sort();
            simplex_set.insert(simplex);
        }
    }

    // Also add all edges from pairwise overlaps (subset of the above)
    // And ensure all sub-simplices are represented
    let mut all_simplices: HashSet<Vec<usize>> = HashSet::new();

    for simplex in &simplex_set {
        // Add all subsets of size >= 2
        let k = simplex.len();
        if k <= 16 {
            // Enumerate all non-empty subsets of size >= 2
            let n_subsets = 1u32 << k;
            for mask in 0..n_subsets {
                let subset: Vec<usize> = simplex
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| (mask >> bit) & 1 == 1)
                    .map(|(_, &v)| v)
                    .collect();
                if subset.len() >= 2 {
                    all_simplices.insert(subset);
                }
            }
        } else {
            // For very large simplices, just add the full simplex and all pairs
            all_simplices.insert(simplex.clone());
            for i in 0..k {
                for j in (i + 1)..k {
                    all_simplices.insert(vec![simplex[i], simplex[j]]);
                }
            }
        }
    }

    let mut simplices: Vec<Vec<usize>> = all_simplices.into_iter().collect();
    // Sort by dimension (size), then lexicographically
    simplices.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

    SimplicialComplex {
        vertices,
        simplices,
    }
}

/// Extract edges (1-simplices) from the nerve complex.
pub fn edges(complex: &SimplicialComplex) -> Vec<(usize, usize)> {
    complex
        .simplices
        .iter()
        .filter(|s| s.len() == 2)
        .map(|s| (s[0], s[1]))
        .collect()
}

/// Extract triangles (2-simplices) from the nerve complex.
pub fn triangles(complex: &SimplicialComplex) -> Vec<(usize, usize, usize)> {
    complex
        .simplices
        .iter()
        .filter(|s| s.len() == 3)
        .map(|s| (s[0], s[1], s[2]))
        .collect()
}

/// Count simplices by dimension.
pub fn simplex_counts_by_dim(complex: &SimplicialComplex) -> HashMap<usize, usize> {
    let mut counts = HashMap::new();
    // 0-simplices = vertices
    counts.insert(0, complex.vertices.len());
    for s in &complex.simplices {
        *counts.entry(s.len() - 1).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clusters() -> HashMap<ClusterID, Vec<usize>> {
        let mut m = HashMap::new();
        // Cluster 0 in interval 0: points {0, 1, 2}
        m.insert(ClusterID::new(0, 0), vec![0, 1, 2]);
        // Cluster 1 in interval 1: points {2, 3}  — shares point 2 with cluster (0,0)
        m.insert(ClusterID::new(1, 0), vec![2, 3]);
        // Cluster 2 in interval 2: points {4, 5}  — no overlap
        m.insert(ClusterID::new(2, 0), vec![4, 5]);
        m
    }

    #[test]
    fn test_build_nerve_basic() {
        let clusters = make_clusters();
        let nerve = build_nerve(&clusters);
        assert_eq!(nerve.vertices.len(), 3);
        // Edge between the two clusters sharing point 2
        let edge_list = edges(&nerve);
        assert_eq!(edge_list.len(), 1);
        // Find which vertices share point 2
        let sharing: Vec<usize> = nerve
            .vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| v.point_indices.contains(&2))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(sharing.len(), 2);
        let (a, b) = (sharing[0].min(sharing[1]), sharing[0].max(sharing[1]));
        assert!(edge_list.contains(&(a, b)));
    }

    #[test]
    fn test_nerve_no_overlap() {
        let mut clusters = HashMap::new();
        clusters.insert(ClusterID::new(0, 0), vec![0, 1]);
        clusters.insert(ClusterID::new(1, 0), vec![2, 3]);
        let nerve = build_nerve(&clusters);
        assert_eq!(edges(&nerve).len(), 0);
    }

    #[test]
    fn test_nerve_empty() {
        let clusters: HashMap<ClusterID, Vec<usize>> = HashMap::new();
        let nerve = build_nerve(&clusters);
        assert!(nerve.vertices.is_empty());
        assert!(nerve.simplices.is_empty());
    }

    #[test]
    fn test_nerve_triangle() {
        let mut clusters = HashMap::new();
        // Three clusters all sharing point 0
        clusters.insert(ClusterID::new(0, 0), vec![0, 1]);
        clusters.insert(ClusterID::new(1, 0), vec![0, 2]);
        clusters.insert(ClusterID::new(2, 0), vec![0, 3]);
        let nerve = build_nerve(&clusters);
        let tris = triangles(&nerve);
        assert_eq!(tris.len(), 1);
    }

    #[test]
    fn test_simplex_counts() {
        let clusters = make_clusters();
        let nerve = build_nerve(&clusters);
        let counts = simplex_counts_by_dim(&nerve);
        assert_eq!(counts.get(&&0), Some(&3)); // 3 vertices
        assert!(counts.get(&&1).map_or(false, |&c| c >= 1)); // at least 1 edge
    }

    #[test]
    fn test_single_cluster() {
        let mut clusters = HashMap::new();
        clusters.insert(ClusterID::new(0, 0), vec![0, 1, 2]);
        let nerve = build_nerve(&clusters);
        assert_eq!(nerve.vertices.len(), 1);
        assert!(nerve.simplices.is_empty());
    }
}
