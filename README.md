# mapper-agent

A pure Rust implementation of the **Mapper algorithm** (Singh, Mémoli, Carlsson 2007) for discovering topological structure in high-dimensional agent state spaces. Zero external dependencies beyond `serde`.

---

## Table of Contents

1. [Overview](#overview)
2. [Theory](#theory)
   - [The Mapper Construction](#the-mapper-construction)
   - [The Nerve Theorem](#the-nerve-theorem)
   - [Filters](#filters)
   - [Overlapping Covers](#overlapping-covers)
   - [Clustering](#clustering)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Module Reference](#module-reference)
6. [Examples](#examples)
   - [Basic Mapper Pipeline](#example-1-basic-mapper-pipeline)
   - [Custom Filter Function](#example-2-custom-filter-function)
   - [Adjusting Cover and Clustering](#example-3-adjusting-cover-and-clustering)
   - [Graph Visualization and Analysis](#example-4-graph-visualization-and-analysis)
7. [Performance](#performance)
8. [Design Decisions](#design-decisions)
9. [Comparison with Other Methods](#comparison-with-other-methods)
10. [Practical Applications](#practical-applications)
11. [API Documentation](#api-documentation)
12. [References](#references)
13. [License](#license)

---

## Overview

The Mapper algorithm is a tool from **topological data analysis (TDA)** that constructs a combinatorial graph — a simplicial complex — from high-dimensional data. Unlike dimensionality reduction techniques (PCA, t-SNE, UMAP) that produce embeddings, Mapper produces a graph that captures the *shape* of the data: its clusters, loops, flares, and voids.

**Key insight:** Mapper reveals *topology*, not just geometry. Two datasets can have similar point clouds but different topological structures (e.g., a ring vs. a filled disk). Mapper distinguishes these.

### When to Use Mapper

- **Agent state space exploration:** Understand the landscape of agent behaviors — are states clustered in blobs? Arranged in a ring? Connected by narrow bridges?
- **Anomaly detection:** Points that create isolated nodes or long branches in the Mapper graph are structurally anomalous.
- **Shape discovery:** Identify the intrinsic topology of your data (connected components, loops, branching structures).
- **Multi-scale analysis:** By varying the cover resolution and clustering threshold, you explore data at multiple granularities.

### What mapper-agent Provides

| Feature | Description |
|---------|-------------|
| **Filter functions** | PCA projection, distance from centroid, eccentricity, kernel density estimation |
| **Overlapping covers** | Configurable number of intervals and overlap percentage |
| **Single-linkage clustering** | With iterative union-find (path compression, union by rank) |
| **Nerve construction** | Full simplicial complex with higher-order simplices |
| **Graph analysis** | Connected components, node attributes, force-directed layout |
| **Visualization** | DOT/graphviz export, ASCII rendering, edge lists |
| **Serialization** | All types derive `Serialize` + `Deserialize` via serde |

### Design Principles

- **Concrete types:** `Vec<f64>` for data points, `Vec<Vec<f64>>` for distance matrices, `HashMap` for cluster maps. No abstract topology traits.
- **Iterative algorithms:** Union-find with iterative path compression. Force-directed layout via iterative spring embedding. No recursion.
- **Zero external deps:** Only `serde` for serialization. Pure Rust math.
- **Edition 2024:** Modern Rust idioms.

---

## Theory

### The Mapper Construction

Given a dataset X ⊂ ℝᵈ, the Mapper algorithm produces a simplicial complex M(X, f, U):

```
M(X, f, U) = Nerve(Cluster(f⁻¹(Uᵢ) ∩ X))
```

Where:
- **f : X → ℝᵏ** is a *filter function* that maps each data point to a lower-dimensional space
- **U = {U₁, U₂, ..., Uₙ}** is an *overlapping cover* of the filter range f(X)
- **Cluster** is a clustering algorithm applied to the preimage f⁻¹(Uᵢ) ∩ X of each cover element
- **Nerve** constructs a simplicial complex from the overlap pattern of the resulting clusters

**Intuition:** Mapper slices the data along the filter axis, clusters within each slice, and then connects clusters that share data points (via overlapping cover elements). The resulting graph is a "skeleton" of the data's shape.

### The Nerve Theorem

The mathematical foundation of Mapper is the **Nerve Theorem**:

```
Nerve(U) ≃ ∪U    when all finite intersections are contractible
```

This means the nerve (a combinatorial object) has the same *homotopy type* as the union of cover elements (a topological object). In practical terms: if your cover is fine enough and your data is well-behaved, the Mapper graph faithfully represents the topology of your data.

**Implications:**
- If your data lies on a circle (S¹), Mapper recovers a cycle graph
- If your data has k disconnected components, Mapper produces k connected components
- If your data has a branching structure, Mapper shows branches
- Loops in the Mapper graph correspond to topological loops in the data

### Filters

The filter f : X → ℝᵏ reduces dimensionality while preserving the structure you care about:

- **PCA projection** (f(x) = ⟨x - μ, v₁⟩): Captures the direction of maximum variance. Good for elongated data.
- **Distance from centroid** (f(x) = ‖x - μ‖): Measures how "central" each point is. Reveals radial structure.
- **Eccentricity** (f(x) = (1/n)Σ‖x - xᵢ‖): Mean distance to all other points. Outliers get high eccentricity.
- **Kernel density estimation** (f(x) = (1/n)Σ K_σ(x - xᵢ)): Estimates local density. Dense regions get high values.

The choice of filter is the primary *lens* through which you view your data. Different filters reveal different structures.

### Overlapping Covers

Given filter range [min, max], we divide it into n intervals with overlap p:

```
Interval width:     w = range / (n·(1-p) + p)
Step between starts: s = w·(1-p)
```

With 20% overlap, adjacent intervals share 20% of their range. Points in the overlap region belong to both intervals, creating the connections that form edges in the Mapper graph.

**Why overlap matters:** Without overlap, adjacent intervals are disconnected and the Mapper graph is just a collection of disjoint cluster nodes. Overlap creates *bridges* between adjacent regions, enabling the graph to capture continuous structures.

### Clustering

Within each cover interval, we apply **single-linkage hierarchical clustering** with threshold δ:

Two points are in the same cluster if there exists a chain of points x₁, x₂, ..., xₖ where each consecutive pair has distance ≤ δ. This is equivalent to: the minimum spanning tree distance between the points is ≤ δ.

**Implementation:** Union-find with iterative path compression and union by rank. For each pair of points within the interval, if their distance ≤ δ, we union their sets. This is O(n² · k) where n is the total number of points and k is the number of intervals.

**Why single-linkage:** Single-linkage is *chaining* — it can merge clusters through narrow bridges. This is desirable for Mapper because we want to capture connected structures. Complete-linkage or average-linkage would be too conservative, potentially missing legitimate connections.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mapper Algorithm Pipeline                    │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Input    │    │  Filter   │    │  Cover    │    │ Cluster  │  │
│  │ Vec<f64>  │───▶│ f: X→ℝᵏ  │───▶│ Overlap   │───▶│ Single-  │  │
│  │ data pts  │    │ PCA/cent/ │    │ Intervals │    │ linkage  │  │
│  │           │    │ ecc/kde   │    │           │    │ union-find│  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│                                                       │         │
│  ┌──────────┐    ┌──────────┐                         │         │
│  │ Mapper   │    │ Visualize│    ┌──────────┐         │         │
│  │ Graph    │◀───│ DOT/ASCII│◀───│  Nerve    │◀────────┘         │
│  │ adj list │    │ layout   │    │ Simplicial│                    │
│  │ nodes    │    │          │    │ Complex   │                    │
│  └──────────┘    └──────────┘    └──────────┘                    │
│                                                                   │
│  Steps: data → filter → cover → cluster → nerve → mapper graph   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Raw Data Points                Filter Values
[0.0, 0.0] ─┐                 0.71 ─┐
[1.0, 0.0] ─┤  PCA filter     0.00 ─┤  Cover into
[0.0, 1.0] ─┤  ──────────▶   1.41 ─┤  overlapping
[5.0, 5.0] ─┤                 7.07 ─┤  intervals
[5.5, 5.5] ─┘                 7.78 ─┘
                                        │
                     ┌──────────────────┤
                     ▼                  ▼
              Interval 0:          Interval 1:
              {0,1,2}              {3,4}
                     │                  │
                     ▼                  ▼
              Cluster A:           Cluster B:
              {0,1,2}              {3,4}
                     │                  │
                     └──────┬───────────┘
                            ▼
                     Nerve: A──B
                     (edge if overlap)
```

---

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
mapper-agent = "0.1"
```

Basic usage:

```rust
use mapper_agent::*;

// Create some data points
let data = vec![
    vec![0.0, 0.0],
    vec![1.0, 1.0],
    vec![10.0, 10.0],
    vec![11.0, 11.0],
];

// Use a built-in filter
let filter = filter::distance_from_centroid_filter();

// Run the Mapper pipeline with defaults
let result = mapper::mapper_default(&data, &filter);

println!("Graph has {} nodes and {} edges",
    result.graph.node_count(),
    result.graph.edge_count());

// Export for visualization
println!("{}", result.graph.to_dot());
```

---

## Module Reference

| Module | Description | Key Types | Example Usage |
|--------|-------------|-----------|---------------|
| `filter` | Filter functions (PCA, centroid dist, eccentricity, KDE) | `FilterFunction`, `FilterValues`, `DataPoint` | `filter::eccentricity_filter()` |
| `cover` | Overlapping interval covers | `OverlappingCover`, `CoverConfig`, `Interval` | `CoverConfig::new(10, 0.2)` |
| `cluster` | Single-linkage clustering with union-find | `ClusterID`, `ClusterConfig`, `UnionFind` | `ClusterConfig::new(1.5)` |
| `nerve` | Nerve construction (simplicial complex) | `SimplicialComplex` | `nerve::build_nerve(&clusters)` |
| `mapper` | Full Mapper pipeline | `MapperConfig`, `MapperResult` | `mapper::mapper_pipeline(...)` |
| `visualization` | Graph analysis and rendering | `MapperGraph`, `MapperNode`, `NodeAttributes` | `graph.to_dot()`, `graph.to_ascii(60, 30)` |

### Type Aliases

```rust
/// A data point: Vec<f64> of coordinates
pub type DataPoint = Vec<f64>;
```

---

## Examples

### Example 1: Basic Mapper Pipeline

```rust
use mapper_agent::*;
use mapper_agent::filter;
use mapper_agent::cover::CoverConfig;
use mapper_agent::cluster::ClusterConfig;

// Generate a synthetic point cloud: two well-separated blobs
let mut data = Vec::new();

// Blob 1: 15 points near (0, 0)
for i in 0..15 {
    let angle = (i as f64) * 0.4;
    data.push(vec![angle.cos() * 0.5, angle.sin() * 0.5]);
}

// Blob 2: 15 points near (10, 10)
for i in 0..15 {
    let angle = (i as f64) * 0.4;
    data.push(vec![10.0 + angle.cos() * 0.5, 10.0 + angle.sin() * 0.5]);
}

// Choose a filter function
let filter = filter::distance_from_centroid_filter();

// Configure the pipeline
let config = mapper::MapperConfig::new(
    CoverConfig::new(5, 0.25),   // 5 intervals, 25% overlap
    ClusterConfig::new(2.0),     // merge clusters within distance 2.0
);

// Run the Mapper algorithm
let result = mapper::mapper_pipeline(&data, &filter, &config);

// Analyze the result
println!("Input: {} data points", result.num_points);
println!("Graph: {} nodes, {} edges",
    result.graph.node_count(),
    result.graph.edge_count());
println!("Connected components: {}",
    result.graph.connected_components().len());

// Each node represents a cluster
for node in &result.graph.nodes {
    println!("  Node {}: {} points, density={:.2}",
        node.index, node.size, node.attributes.density);
}

// Print the graphviz DOT representation
println!("{}", result.graph.to_dot());
```

### Example 2: Custom Filter Function

```rust
use mapper_agent::*;
use mapper_agent::filter::FilterFunction;

// Suppose we're analyzing agent states where each state is
// [position_x, position_y, velocity, health, ammo]
// We want a filter that captures "tactical posture" — the ratio
// of health to velocity (high = defensive, low = aggressive)

let agent_states = vec![
    vec![0.0, 0.0, 5.0, 100.0, 50.0],   // fast, healthy
    vec![1.0, 1.0, 4.0, 80.0, 40.0],     // moderate
    vec![2.0, 2.0, 2.0, 60.0, 30.0],     // slow, moderate
    vec![10.0, 10.0, 8.0, 20.0, 10.0],   // fast, damaged
    vec![11.0, 11.0, 7.0, 15.0, 5.0],    // fast, very damaged
];

// Custom filter: health / velocity ratio
let tactical_filter = FilterFunction::new(
    "tactical_posture",
    |data, index| {
        let velocity = data[index][2].max(0.1); // avoid div-by-zero
        let health = data[index][3];
        vec![health / velocity] // defensive score
    },
);

let config = mapper::MapperConfig::default();
let result = mapper::mapper_pipeline(&agent_states, &tactical_filter, &config);

println!("Tactical posture analysis:");
println!("  {} behavior clusters detected", result.graph.node_count());
for node in &result.graph.nodes {
    println!("  Cluster {}: {} agents, avg posture={:.1}",
        node.index,
        node.size,
        node.attributes.avg_filter_value);
}
```

### Example 3: Adjusting Cover and Clustering

```rust
use mapper_agent::*;
use mapper_agent::filter;
use mapper_agent::cover::CoverConfig;
use mapper_agent::cluster::ClusterConfig;

// Generate data on a circle (annulus)
let circle_data: Vec<Vec<f64>> = (0..100)
    .map(|i| {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 100.0;
        let r = 5.0 + (i as f64 * 3.7).sin() * 0.3; // noisy radius
        vec![r * angle.cos(), r * angle.sin()]
    })
    .collect();

let filter = filter::eccentricity_filter();

// Fine-grained analysis: many intervals, high overlap, tight clustering
let fine_config = mapper::MapperConfig::new(
    CoverConfig::new(12, 0.35),
    ClusterConfig::new(1.5),
);

// Coarse analysis: few intervals, low overlap, loose clustering
let coarse_config = mapper::MapperConfig::new(
    CoverConfig::new(4, 0.15),
    ClusterConfig::new(4.0),
);

let fine_result = mapper::mapper_pipeline(&circle_data, &filter, &fine_config);
let coarse_result = mapper::mapper_pipeline(&circle_data, &filter, &coarse_config);

println!("Fine-grained: {} nodes, {} edges, {} components",
    fine_result.graph.node_count(),
    fine_result.graph.edge_count(),
    fine_result.graph.connected_components().len());

println!("Coarse: {} nodes, {} edges, {} components",
    coarse_result.graph.node_count(),
    coarse_result.graph.edge_count(),
    coarse_result.graph.connected_components().len());

// A circle should ideally produce a cyclic graph
// The fine-grained analysis captures this better
```

### Example 4: Extracting and Visualizing the Mapper Graph

```rust
use mapper_agent::*;
use mapper_agent::filter;

// Simple dataset
let data = vec![
    vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0],
    vec![5.0, 5.0], vec![5.5, 5.5], vec![6.0, 6.0],
    vec![2.5, 2.5], // bridge point between clusters
];

let filter = filter::pca_projection_filter(50);
let result = mapper::mapper_default(&data, &filter);

// 1. DOT format for graphviz/neato rendering
let dot = result.graph.to_dot();
println!("DOT representation:\n{}", dot);
// Render with: echo "$DOT" | dot -Tpng -o mapper.png

// 2. Edge list (TSV format)
let edges = result.graph.to_edge_list();
println!("Edge list:\n{}", edges);

// 3. ASCII art rendering
let ascii = result.graph.to_ascii(60, 20);
println!("ASCII visualization:\n{}", ascii);

// 4. Connected components
let components = result.graph.connected_components();
println!("Connected components: {}", components.len());
for (i, component) in components.iter().enumerate() {
    println!("  Component {}: {} nodes — {:?}", i, component.len(), component);
}

// 5. Node analysis
for node in &result.graph.nodes {
    println!("Node {}: size={}, density={:.3}, members={:?}",
        node.index,
        node.size,
        node.attributes.density,
        node.member_indices);
}

// 6. Access the raw simplicial complex
let nerve = &result.nerve;
println!("Simplicial complex:");
println!("  Vertices: {}", nerve.vertices.len());
println!("  Total simplices: {}", nerve.simplices.len());
let counts = nerve::simplex_counts_by_dim(nerve);
for (dim, count) in &counts {
    println!("  {}-simplices: {}", dim, count);
}
```

---

## Performance

### Complexity Analysis

| Step | Complexity | Notes |
|------|-----------|-------|
| Filter (PCA) | O(n · d · iter) | iter = power method iterations (typically 20-50) |
| Filter (centroid) | O(n · d) | Single pass to compute mean, then O(d) per point |
| Filter (eccentricity) | O(n² · d) | All pairwise distances |
| Filter (KDE) | O(n² · d) | Kernel evaluation at each point |
| Cover construction | O(n · k) | k = number of intervals; each point checked against each interval |
| Clustering | O(n² · k) | For each interval: all pairs of points within it |
| Nerve construction | O(n² · k²) | Checking pairwise cluster overlaps |
| Force-directed layout | O(V² · iter + E · iter) | V = nodes, E = edges, iter = layout iterations |

### Scaling Notes

- **10–1000 points:** Near-instantaneous for all operations. This is the sweet spot for agent state analysis.
- **1K–10K points:** Eccentricity and KDE filters become noticeable (O(n²)). Consider PCA or centroid filters for larger datasets. Clustering within intervals helps — each interval only contains a subset of points.
- **10K+ points:** Consider sampling or approximate nearest neighbors. The current implementation is exact and not optimized for very large datasets.

### Memory

- Distance matrices: O(n²) when precomputed. The clustering step uses on-demand distance computation instead.
- Mapper graph: O(V + E) where V = number of clusters, E = number of overlapping pairs.
- Simplicial complex: In the worst case, O(2^V) for the full nerve, but in practice much smaller due to limited overlaps.

---

## Design Decisions

### Why Single-Linkage Clustering?

Single-linkage has a known weakness (chaining — it can merge clusters through narrow bridges). For Mapper, this is actually a *feature*:

1. **Topological sensitivity:** Mapper needs to detect when regions are connected, even through thin bridges. Single-linkage captures this.
2. **Nerve compatibility:** The nerve theorem requires that cover intersections are contractible. Single-linkage produces "blob-like" clusters that satisfy this condition more reliably than other methods.
3. **Simplicity:** No need for cluster count estimation or linkage tree cutting. Just set a distance threshold δ.

### Why Union-Find (Iterative)?

1. **No recursion:** Avoids stack overflow on large datasets. Path compression is implemented iteratively.
2. **Near-constant amortized time:** With path compression and union by rank, each operation is O(α(n)) ≈ O(1) where α is the inverse Ackermann function.
3. **Memory efficient:** Only two arrays: `parent` and `rank`.

### Why f64 Filters?

1. **Precision:** Agent state spaces often span multiple orders of magnitude. f32 would lose precision in difference calculations.
2. **Standard practice:** Scientific computing in Rust overwhelmingly uses f64. The slight memory overhead is negligible.
3. **Distance calculations:** Euclidean distance accumulates rounding errors; f64 provides adequate precision.

### Why Vec<f64> for Data Points?

1. **Concrete and predictable:** No trait objects, no vtables, no generic bounds. Just vectors of floats.
2. **Serde compatibility:** `Vec<f64>` serializes trivially.
3. **Cache-friendly:** Contiguous memory layout for numerical operations.

---

## Comparison with Other Methods

### Mapper vs t-SNE

| Aspect | Mapper | t-SNE |
|--------|--------|-------|
| Output | Graph (simplicial complex) | 2D/3D embedding |
| Preserves | Topology (connectivity, loops) | Local neighborhood structure |
| Deterministic | Yes (given same parameters) | No (random initialization) |
| Parameters | Filter, cover, clustering | Perplexity, learning rate |
| Scale | Moderate (O(n² · k)) | Slow (O(n²) per iteration) |
| Interpretability | Graph structure is directly interpretable | Embedding requires visual inspection |

**When Mapper wins:** You need to detect loops, flares, or disconnected components. t-SNE may break loops or create artificial clusters.

**When t-SNE wins:** You need a visual 2D representation for human inspection of local structure.

### Mapper vs UMAP

| Aspect | Mapper | UMAP |
|--------|--------|------|
| Output | Graph (simplicial complex) | 2D/3D embedding + graph |
| Preserves | Topology (exactly, via nerve theorem) | Local + global structure (approximately) |
| Mathematical basis | Nerve theorem (homotopy equivalence) | Riemannian geometry + fuzzy simplicial sets |
| Parameters | Filter, cover, clustering | n_neighbors, min_dist |
| Speed | O(n² · k) | O(n · k · log n) with approximate NN |

**When Mapper wins:** You need guaranteed topological correctness (under the nerve theorem conditions) or multi-resolution analysis.

**When UMAP wins:** You need a fast, general-purpose embedding with good global structure preservation.

### Mapper vs PCA

| Aspect | Mapper | PCA |
|--------|--------|------|
| Output | Graph | Linear projection |
| Captures | Nonlinear topology | Linear variance |
| Sensitivity | Can detect loops, clusters | Only detects elongation |
| Parameters | Filter + cover + cluster | Number of components |

**When Mapper wins:** Your data has nonlinear structure (rings, Y-shapes, clusters of varying density).

**When PCA wins:** You need a fast, deterministic linear summary of variance. PCA is also useful *as a filter* within Mapper.

---

## Practical Applications

### Agent Behavior Clustering

Map agent states (position, velocity, health, resources) through the Mapper pipeline to identify distinct behavioral modes. Nodes with high density represent common behaviors; isolated nodes represent rare or anomalous states.

```rust
// Agent state: [x, y, vx, vy, health, ammo]
let states = collect_agent_states(); // Vec<Vec<f64>>
let filter = filter::pca_projection_filter(30);
let result = mapper::mapper_default(&states, &filter);

// Find anomalous behaviors (low-density, isolated nodes)
for node in &result.graph.nodes {
    if node.size == 1 || node.attributes.density < 0.1 {
        println!("Anomalous state cluster: {:?}", node.member_indices);
    }
}
```

### Anomaly Detection

Points that appear in isolated nodes (no edges) or at the tips of branches are structurally anomalous — they don't fit into the main topological structure of the data.

```rust
let components = result.graph.connected_components();
for component in &components {
    if component.len() == 1 {
        let node = &result.graph.nodes[component[0]];
        println!("Isolated cluster: {} points", node.size);
        println!("  Members: {:?}", node.member_indices);
    }
}
```

### State Space Exploration

Use Mapper to understand the topology of an agent's reachable state space. If the Mapper graph has loops, the agent can cycle through states. If it's a tree, the agent's decisions are irreversible.

### Shape of Data

Mapper answers the question: "What *shape* does my data have?"

- **One connected component with no cycles:** Tree-like structure (branching)
- **One component with one cycle:** Ring/loop structure
- **Multiple components:** Disconnected clusters
- **Long chains:** Sequential/progressive structure

---

## API Documentation

### Core Types

#### `DataPoint`
```rust
pub type DataPoint = Vec<f64>;
```
A data point represented as a vector of f64 coordinates.

#### `FilterFunction`
```rust
pub struct FilterFunction {
    pub name: String,
    pub compute: Option<Box<dyn Fn(&[DataPoint], usize) -> Vec<f64>>>,
}
```
A filter function that maps a data point (by index within a dataset) to filter values.

#### `FilterValues`
```rust
pub struct FilterValues {
    pub values: Vec<Vec<f64>>,  // One Vec<f64> per data point
    pub filter_name: String,
}
```
Result of applying a filter to an entire dataset.

#### `CoverConfig`
```rust
pub struct CoverConfig {
    pub num_intervals: usize,  // Number of intervals (default: 10)
    pub overlap: f64,          // Overlap fraction (default: 0.2)
}
```

#### `ClusterConfig`
```rust
pub struct ClusterConfig {
    pub threshold: f64,  // Distance threshold δ (default: 1.0)
}
```

#### `ClusterID`
```rust
pub struct ClusterID {
    pub interval: usize,  // Cover interval index
    pub cluster: usize,   // Cluster index within the interval
}
```

#### `MapperConfig`
```rust
pub struct MapperConfig {
    pub cover: CoverConfig,
    pub cluster: ClusterConfig,
}
```

#### `MapperResult`
```rust
pub struct MapperResult {
    pub filter_values: FilterValues,
    pub cover: OverlappingCover,
    pub clusters: Vec<IntervalClusters>,
    pub nerve: SimplicialComplex,
    pub graph: MapperGraph,
    pub num_points: usize,
}
```

#### `MapperGraph`
```rust
pub struct MapperGraph {
    pub nodes: Vec<MapperNode>,
    pub adj: Vec<Vec<usize>>,
    pub edge_list: Vec<(usize, usize)>,
}
```

### Built-in Filters

| Filter | Function | Output | Complexity |
|--------|----------|--------|------------|
| PCA Projection | `pca_projection_filter(iterations)` | 1D: first principal component | O(n·d·iter) |
| Distance from Centroid | `distance_from_centroid_filter()` | 1D: L² distance to mean | O(n·d) |
| Eccentricity | `eccentricity_filter()` | 1D: mean distance to all others | O(n²·d) |
| Kernel Density | `kde_filter(bandwidth)` | 1D: Gaussian KDE | O(n²·d) |

### Graph Methods

| Method | Description |
|--------|-------------|
| `node_count()` | Number of nodes |
| `edge_count()` | Number of edges |
| `to_dot()` | Graphviz DOT representation |
| `to_edge_list()` | TSV edge list |
| `to_ascii(w, h)` | ASCII art rendering |
| `connected_components()` | Vec of connected components |
| `force_directed_layout(iter, w, h)` | Node positions via spring embedding |

---

## References

1. **Singh, G., Mémoli, F., & Carlsson, G.** (2007). Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition. *Eurographics Symposium on Point-Based Graphics*.

2. **Carlsson, G.** (2009). Topology and Data. *Bulletin of the American Mathematical Society*, 46(2), 255–308.

3. **Lum, P. Y., Singh, G., Lehman, A., Ishkanov, T., Vejdemo-Johansson, M., Alagappan, M., ... & Carlsson, G.** (2013). Extracting insights from the shape of complex data using topology. *Scientific Reports*, 3, 1236.

4. **Edelsbrunner, H., & Harer, J.** (2010). *Computational Topology: An Introduction*. American Mathematical Society.

5. **Chazal, F., Fasy, B. T., Lecci, F., Rinaldo, A., & Wasserman, L.** (2014). Stochastic Convergence of Persistence Landscapes and Silhouettes. *Annual Symposium on Computational Geometry*, 474–483.

6. **Munch, E.** (2017). A User's Guide to Topological Data Analysis. *Journal of Learning Analytics*, 4(2), 47–61.

7. **Bauer, U.** (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. *Journal of Applied and Computational Topology*, 5, 391–423.

8. **Bubenik, P., & Dłotko, P.** (2017). A persistence landscapes toolbox for topological statistics. *Journal of Symbolic Computation*, 78, 91–114.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
