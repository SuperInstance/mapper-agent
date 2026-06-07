//! Visualization and graph analysis for Mapper output.
//!
//! The `MapperGraph` is the primary output of the Mapper algorithm: an
//! undirected graph where each node represents a cluster of data points
//! and edges connect clusters that share data points.
//!
//! # Layout
//!
//! Force-directed layout positions nodes using an iterative (NOT recursive)
//! spring-embedding algorithm. This is suitable for ASCII rendering or
//! export to external graph tools.

use crate::filter::{DataPoint, compute_centroid, euclidean_distance};
use crate::nerve::{SimplicialComplex, edges};
use serde::{Deserialize, Serialize};

/// A node in the Mapper graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapperNode {
    /// Unique node index.
    pub index: usize,
    /// Number of data points in this cluster.
    pub size: usize,
    /// Centroid of the data points in this cluster.
    pub centroid: Vec<f64>,
    /// Indices of member data points.
    pub member_indices: Vec<usize>,
    /// Attributes for visualization (density, filter value, custom).
    pub attributes: NodeAttributes,
}

/// Attributes for coloring and labeling graph nodes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeAttributes {
    /// Cluster density (number of points / area proxy).
    pub density: f64,
    /// Average filter value of members (for 1D filters).
    pub avg_filter_value: f64,
    /// Custom metric value.
    pub custom: f64,
}

/// The Mapper graph: adjacency list + node metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapperGraph {
    /// Nodes of the graph.
    pub nodes: Vec<MapperNode>,
    /// Adjacency list: `adj[i]` contains indices of neighbors of node i.
    pub adj: Vec<Vec<usize>>,
    /// Edge list for export.
    pub edge_list: Vec<(usize, usize)>,
}

impl MapperGraph {
    /// Build a MapperGraph from a nerve complex and the original data.
    pub fn from_nerve(nerve: &SimplicialComplex, data: &[DataPoint]) -> Self {
        let n = nerve.vertices.len();
        let mut nodes = Vec::with_capacity(n);
        let mut adj = vec![vec![]; n];

        for (i, info) in nerve.vertices.iter().enumerate() {
            let members = &info.point_indices;
            let member_data: Vec<&DataPoint> = members.iter().map(|&idx| &data[idx]).collect();
            let centroid = if member_data.is_empty() {
                vec![]
            } else {
                compute_centroid(&member_data.iter().map(|&d| d.clone()).collect::<Vec<_>>())
            };

            let density = if members.is_empty() {
                0.0
            } else if members.len() == 1 {
                1.0
            } else {
                // Average pairwise distance as density proxy (inverse)
                let mut total = 0.0;
                let mut count = 0;
                for a in 0..members.len() {
                    for b in (a + 1)..members.len() {
                        total += euclidean_distance(&data[members[a]], &data[members[b]]);
                        count += 1;
                    }
                }
                if total > 0.0 {
                    count as f64 / total
                } else {
                    1.0
                }
            };

            let avg_filter = if members.is_empty() {
                0.0
            } else {
                // Use first coordinate as proxy
                members
                    .iter()
                    .map(|&idx| data[idx].iter().sum::<f64>() / data[idx].len() as f64)
                    .sum::<f64>()
                    / members.len() as f64
            };

            nodes.push(MapperNode {
                index: i,
                size: members.len(),
                centroid,
                member_indices: members.clone(),
                attributes: NodeAttributes {
                    density,
                    avg_filter_value: avg_filter,
                    custom: 0.0,
                },
            });
        }

        // Build adjacency from nerve edges
        let edge_list = edges(nerve);
        for &(a, b) in &edge_list {
            if a < n && b < n {
                if !adj[a].contains(&b) {
                    adj[a].push(b);
                }
                if !adj[b].contains(&a) {
                    adj[b].push(a);
                }
            }
        }

        Self {
            nodes,
            adj,
            edge_list,
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edge_list.len()
    }

    /// Export as edge list string (for graphviz/neato).
    ///
    /// Format:
    /// ```text
    /// graph mapper {
    ///   0 -- 1;
    ///   1 -- 2;
    /// }
    /// ```
    pub fn to_dot(&self) -> String {
        let mut s = String::from("graph mapper {\n");
        for node in &self.nodes {
            s.push_str(&format!(
                "  {} [label=\"n{} (size={})\"];\n",
                node.index, node.index, node.size
            ));
        }
        for &(a, b) in &self.edge_list {
            s.push_str(&format!("  {} -- {};\n", a, b));
        }
        s.push_str("}\n");
        s
    }

    /// Export as simple edge list (TSV: src dst per line).
    pub fn to_edge_list(&self) -> String {
        self.edge_list
            .iter()
            .map(|(a, b)| format!("{}\t{}", a, b))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Find connected components using iterative BFS.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        if n == 0 {
            return vec![];
        }
        let mut visited = vec![false; n];
        let mut components = vec![];

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut component = vec![];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;

            while let Some(node) = queue.pop_front() {
                component.push(node);
                for &neighbor in &self.adj[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            components.push(component);
        }
        components
    }

    /// Force-directed layout using iterative spring embedding.
    ///
    /// Returns (x, y) positions for each node.
    ///
    /// # Algorithm
    /// 1. Initialize positions randomly (seeded by node index).
    /// 2. For each iteration:
    ///    a. Compute repulsive forces between all pairs.
    ///    b. Compute attractive forces along edges.
    ///    c. Update positions with temperature cooling.
    ///
    /// NOT recursive. Entirely iterative.
    pub fn force_directed_layout(
        &self,
        iterations: usize,
        width: f64,
        height: f64,
    ) -> Vec<(f64, f64)> {
        let n = self.nodes.len();
        if n == 0 {
            return vec![];
        }

        // Initialize positions in a circle
        let mut pos: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (
                    width / 2.0 + width / 3.0 * angle.cos(),
                    height / 2.0 + height / 3.0 * angle.sin(),
                )
            })
            .collect();

        let k = (width * height / (n as f64).max(1.0)).sqrt(); // optimal distance
        let mut temperature = width / 2.0;
        let cooling = temperature / (iterations as f64 + 1.0);

        for _ in 0..iterations {
            let mut displacement: Vec<(f64, f64)> = vec![(0.0, 0.0); n];

            // Repulsive forces between all pairs
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let dx = pos[i].0 - pos[j].0;
                    let dy = pos[i].1 - pos[j].1;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.01);
                    let force = k * k / dist;
                    displacement[i].0 += dx / dist * force;
                    displacement[i].1 += dy / dist * force;
                }
            }

            // Attractive forces along edges
            for &(a, b) in &self.edge_list {
                let dx = pos[a].0 - pos[b].0;
                let dy = pos[a].1 - pos[b].1;
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);
                let force = dist * dist / k;
                displacement[a].0 -= dx / dist * force;
                displacement[a].1 -= dy / dist * force;
                displacement[b].0 += dx / dist * force;
                displacement[b].1 += dy / dist * force;
            }

            // Apply displacement with temperature limiting
            for i in 0..n {
                let dist = (displacement[i].0 * displacement[i].0
                    + displacement[i].1 * displacement[i].1)
                    .sqrt()
                    .max(0.01);
                let scale = temperature.min(dist) / dist;
                pos[i].0 += displacement[i].0 * scale;
                pos[i].1 += displacement[i].1 * scale;
                // Keep within bounds
                pos[i].0 = pos[i].0.clamp(0.0, width);
                pos[i].1 = pos[i].1.clamp(0.0, height);
            }

            temperature = (temperature - cooling).max(0.01);
        }

        pos
    }

    /// Render the graph as ASCII art using force-directed layout.
    pub fn to_ascii(&self, width: usize, height: usize) -> String {
        if self.nodes.is_empty() {
            return "(empty graph)\n".to_string();
        }

        let positions = self.force_directed_layout(100, width as f64, height as f64);

        // Create canvas
        let mut canvas = vec![vec![' '; width]; height];

        // Draw edges
        for &(a, b) in &self.edge_list {
            draw_line(
                &mut canvas,
                positions[a].0 as isize,
                positions[a].1 as isize,
                positions[b].0 as isize,
                positions[b].1 as isize,
                '-',
            );
        }

        // Draw nodes
        for (i, node) in self.nodes.iter().enumerate() {
            let x = positions[i].0 as isize;
            let y = positions[i].1 as isize;
            if y >= 0 && y < height as isize && x >= 0 && x < width as isize {
                let ch = if node.size > 5 { 'O' } else { 'o' };
                canvas[y as usize][x as usize] = ch;
            }
        }

        // Render
        let mut s = String::new();
        for row in &canvas {
            s.push_str(&row.iter().collect::<String>());
            s.push('\n');
        }
        s
    }
}

/// Draw a line on a canvas using Bresenham's algorithm (iterative).
fn draw_line(canvas: &mut [Vec<char>], x0: isize, y0: isize, x1: isize, y1: isize, ch: char) {
    let h = canvas.len() as isize;
    let w = if h > 0 { canvas[0].len() as isize } else { 0 };

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let mut cx = x0;
    let mut cy = y0;

    loop {
        if cy >= 0 && cy < h && cx >= 0 && cx < w {
            let existing = canvas[cy as usize][cx as usize];
            if existing == ' ' {
                canvas[cy as usize][cx as usize] = ch;
            }
        }
        if cx == x1 && cy == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::ClusterID;
    use crate::nerve::build_nerve;
    use std::collections::HashMap;

    fn test_data() -> Vec<DataPoint> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![5.0, 5.0]]
    }

    fn test_nerve() -> SimplicialComplex {
        let mut clusters = HashMap::new();
        clusters.insert(ClusterID::new(0, 0), vec![0, 1]);
        clusters.insert(ClusterID::new(1, 0), vec![1, 2]);
        build_nerve(&clusters)
    }

    #[test]
    fn test_mapper_graph_from_nerve() {
        let nerve = test_nerve();
        let data = test_data();
        let graph = MapperGraph::from_nerve(&nerve, &data);
        assert_eq!(graph.node_count(), 2);
        assert!(graph.edge_count() >= 1);
    }

    #[test]
    fn test_connected_components_single() {
        let nerve = test_nerve();
        let data = test_data();
        let graph = MapperGraph::from_nerve(&nerve, &data);
        let cc = graph.connected_components();
        assert_eq!(cc.len(), 1);
    }

    #[test]
    fn test_connected_components_disconnected() {
        let mut clusters = HashMap::new();
        clusters.insert(ClusterID::new(0, 0), vec![0]);
        clusters.insert(ClusterID::new(1, 0), vec![1]);
        let nerve = build_nerve(&clusters);
        let data = test_data();
        let graph = MapperGraph::from_nerve(&nerve, &data);
        let cc = graph.connected_components();
        assert_eq!(cc.len(), 2);
    }

    #[test]
    fn test_to_dot() {
        let nerve = test_nerve();
        let data = test_data();
        let graph = MapperGraph::from_nerve(&nerve, &data);
        let dot = graph.to_dot();
        assert!(dot.starts_with("graph mapper {"));
        assert!(dot.contains("--"));
    }

    #[test]
    fn test_to_edge_list() {
        let nerve = test_nerve();
        let data = test_data();
        let graph = MapperGraph::from_nerve(&nerve, &data);
        let el = graph.to_edge_list();
        assert!(!el.is_empty() || graph.edge_count() == 0);
    }

    #[test]
    fn test_force_directed_layout() {
        let nerve = test_nerve();
        let data = test_data();
        let graph = MapperGraph::from_nerve(&nerve, &data);
        let positions = graph.force_directed_layout(50, 80.0, 40.0);
        assert_eq!(positions.len(), 2);
        // Positions should be within bounds
        for (x, y) in &positions {
            assert!(*x >= 0.0 && *x <= 80.0);
            assert!(*y >= 0.0 && *y <= 40.0);
        }
    }

    #[test]
    fn test_ascii_render() {
        let nerve = test_nerve();
        let data = test_data();
        let graph = MapperGraph::from_nerve(&nerve, &data);
        let ascii = graph.to_ascii(40, 20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_empty_graph() {
        let nerve = SimplicialComplex {
            vertices: vec![],
            simplices: vec![],
        };
        let data: Vec<DataPoint> = vec![];
        let graph = MapperGraph::from_nerve(&nerve, &data);
        assert_eq!(graph.node_count(), 0);
        assert!(graph.to_ascii(20, 10).contains("empty"));
    }
}
