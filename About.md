<div align="center">

# Google Summer of Code 2026

## GPU-Accelerated Spatial Searching & Mesh Intersection Engine

### CGAL · CUDA · cuBQL · Exact Geometric Predicates

<br>

<img src="images/PolyScope2026Aug22.png" alt="Interactive Polyscope viewport showing GPU-accelerated mesh intersection" width="950">

<br>

**YuryUoL** · **CGAL** · **Google Summer of Code 2026**

</div>

---

## Project Information

| | |
|---|---|
| **Contributor** | Yury |
| **GitHub** | [YuryUoL](https://github.com/YuryUoL) |
| **Organization** | [CGAL — Computational Geometry Algorithms Library](https://www.cgal.org/) |
| **Program** | Google Summer of Code 2026 |
| **Project** | GPU-Accelerated Spatial Searching and Mesh Intersection Engine |
| **Development Branch** | [`gsoc2026-gpu-spatial-search-YuryUoL`](https://github.com/CGAL/cgal-public-dev/tree/gsoc2026-gpu-spatial-search-YuryUoL) |
| **Standalone Repository** | [`YuryUoL/gpu-cubql-intersector-standalone`](https://github.com/YuryUoL/gpu-cubql-intersector-standalone) |
| **Primary Technologies** | C++ · CUDA · cuBQL · CGAL · TBB · Polyscope |

---

# 1. Project Overview

Detecting exact triangle-triangle intersections and extracting 3D intersection curves between surface meshes can become a significant computational bottleneck in geometric processing pipelines.

The primary goal of this project was to develop a **high-throughput, hybrid GPU/CPU spatial search and mesh intersection engine for CGAL**.

The implementation combines GPU-based spatial searching through **cuBQL** with the exact geometric predicates provided by **CGAL**. The computationally intensive broad phase is moved to CUDA, while geometrically uncertain cases can be passed back to the CPU for exact evaluation.

The resulting architecture separates the problem into two major components:

- **GPU:** high-throughput spatial searching and candidate generation
- **CPU:** exact geometric predicates and final correctness checks

This allows the GPU to process large numbers of spatial candidates in parallel while retaining the exact arithmetic capabilities required for robust computational geometry.

### Key Objectives

- 🚀 Accelerate spatial searching using CUDA.
- 🌲 Exploit GPU Bounding Volume Hierarchies (BVHs).
- 🔀 Develop parallel dual-tree traversal.
- ⚡ Develop batched single-tree traversal for large candidate sets.
- 🔢 Evaluate geometric predicates efficiently on the GPU.
- 🎯 Preserve exact geometric correctness through CGAL fallbacks.
- 🧵 Exploit CPU parallelism using Intel TBB for exact predicates.
- 🖥️ Provide an interactive Polyscope-based visualization environment.
- 📊 Provide profiling and benchmarking facilities for the complete pipeline.

---

# 2. System Architecture

The overall pipeline follows a heterogeneous GPU/CPU design:

```text
                         ┌──────────────────────┐
                         │      Input Meshes    │
                         │       Mesh A + B     │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │   GPU BVH Building   │
                         │        cuBQL          │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │ Initial Level         │
                         │ Cross-Check (N, M)    │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │   Dual-Tree Traversal │
                         │       GPU Broad Phase │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │ Batched Single-Tree   │
                         │       Traversal       │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │ GPU Predicate Filter  │
                         │ Float / Double / ...  │
                         └───────────┬──────────┘
                                     │
                          uncertain candidates
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │   CGAL Exact CPU     │
                         │ Predicates + TBB      │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │  Exact Intersection  │
                         │       Curves         │
                         └──────────────────────┘
```

The central design principle is to avoid performing expensive exact geometric computations on every possible triangle pair.

Instead, the pipeline progressively reduces the candidate set:

> **BVH pruning → GPU spatial traversal → GPU numerical filtering → CPU exact predicates**

This allows the majority of obviously non-intersecting pairs to be discarded before expensive exact arithmetic is required.

---

# 3. Algorithmic Methodology

The engine provides two different execution strategies depending on the structure of the input problem and the available GPU memory.

## 3.1 Standard Pipeline — `compute`

The standard pipeline constructs full BVHs for both input meshes and performs hierarchical GPU traversal.

### Step 1 — Initial Level Cross-Check

Bounding Volume Hierarchies are constructed for both input meshes.

The traversal engine descends to configurable levels:

- `N` for Mesh A
- `M` for Mesh B

At these levels, a parallel cross-check is performed between candidate nodes from the two trees.

Node pairs whose bounding volumes do not overlap are immediately discarded.

This creates a reduced set of potentially intersecting subtree pairs.

### Step 2 — Dual-Tree Traversal

The surviving node pairs enter a parallel dual-tree traversal stage.

For every candidate pair, the traversal expands both trees simultaneously.

For binary BVHs, this produces up to four child combinations:

```text
(A_left,  B_left)
(A_left,  B_right)
(A_right, B_left)
(A_right, B_right)
```

Each generated pair is tested for bounding-volume overlap.

Non-overlapping pairs are immediately eliminated.

The process continues until the traversal reaches the desired query subtrees.

### Step 3 — Batched Single-Tree Traversal

After the dual-tree traversal reaches suitable subtrees, the surviving candidate pairs are partitioned into batches.

Each batch launches parallel single-tree traversals across the GPU.

The traversal proceeds directly toward leaf nodes and ultimately produces candidate triangle pairs whose bounding boxes overlap.

This approach allows a large number of spatial queries to be processed simultaneously while maintaining high GPU parallelism.

The combination of dual-tree and batched single-tree traversal is one of the central algorithmic components developed during the project.

### Step 4 — GPU/CPU Predicate Evaluation

The generated triangle pairs are passed through geometric filtering.

The first stage uses GPU floating-point predicates.

Depending on the selected predicate mode, uncertain cases can proceed through additional numerical precision or be transferred to the CPU.

The general filtering hierarchy is:

```text
GPU Float
   │
   ├── clearly classified ───────────────► result
   │
   └── uncertain
          │
          ▼
     GPU Double / Extended Filter
          │
          ├── clearly classified ───────► result
          │
          └── uncertain
                 │
                 ▼
          CGAL Exact Predicate
                 │
                 ▼
             final result
```

The CPU fallback uses CGAL's exact geometric machinery, including `CGAL::Exact_predicates_exact_constructions_kernel` where appropriate, together with multi-threading through Intel TBB.

---

# 4. Partial Forest Pipeline — `standaloneCompute`

A full BVH is not always the optimal representation for the problem.

When the interesting geometry is concentrated in localized regions, constructing the complete hierarchy from the root can introduce unnecessary memory allocation and construction work.

The `standaloneCompute` pipeline therefore supports a reduced **partial-forest representation**.

### Skipping Root-to-Level Nodes

Instead of constructing and traversing the entire hierarchy from the root down to levels `N` and `M`, the pipeline can begin directly from the relevant hierarchy levels.

### Dynamic Forest Construction

A collection of relevant subtrees is constructed as a dynamic forest.

Conceptually:

```text
Full BVH

                 Root
                /    \
              ...    ...
             /        \
            N          N
           /|\        /|\
          ...         ...

                ↓

Partial Forest

        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │ Subtree │   │ Subtree │   │ Subtree │
        │    A    │   │    B    │   │    C    │
        └─────────┘   └─────────┘   └─────────┘
```

Only the relevant subtrees participate in the subsequent search.

### On-the-Fly Pruning

The cross-check and single-tree traversal operate directly on this reduced forest.

This can reduce:

- GPU memory usage
- BVH construction work
- temporary allocations
- unnecessary traversal through irrelevant regions

The partial-forest approach is particularly useful when the intersecting regions represent only a relatively small part of the overall meshes.

---

# 5. Interactive Application

The primary application, `MainApp`, integrates the intersection engine with an interactive **Polyscope** 3D viewport.

The application allows meshes to be loaded, transformed, intersected, and inspected interactively.

Mesh transformations can be performed directly through viewport gizmos, allowing the intersection curves to be recomputed as the relative position and orientation of the meshes changes.

<div align="center">

<img src="images/PolyScope2026Aug22.png" alt="Interactive Polyscope mesh intersection visualization" width="950">

<br>

<em>Interactive Polyscope viewport showing transformed meshes and computed intersection geometry.</em>

</div>

The same application also provides command-line controls for benchmarking, profiling, GPU memory management, and predicate experimentation.

---

# 6. Command Reference

## Mesh I/O and Preparation

| Command | Description |
|---|---|
| `load <meshA> [meshB] [qLevel] [rLevel] [leafThresh] [scale]` | Fast parallel mesh loading, normalization, and GPU/BVH setup. |
| `loadOld ...` | Legacy sequential CGAL stream loader used for performance comparison. |

## Pipeline Execution

| Command | Description |
|---|---|
| `compute [batchMult] [dualTreeSteps] [predMode] [showCurves]` | Executes the full GPU dual-tree / batched single-tree intersection pipeline. |
| `standaloneCompute [qLevel] [rLevel] [batchMult] [steps] [thresh] [mode]` | Executes the partial-BVH forest intersection pipeline. |
| `ComputeCGALClassic` | Executes the standard CPU CGAL PMP intersection implementation as a baseline and verification path. |

## Interactive Viewport

| Command | Description |
|---|---|
| `gizmo <showA(0/1)> [showB(0/1)]` | Enables or disables the 3D transformation gizmos for Mesh A and Mesh B. |
| `transform <rotA> <transA> <rotB> <transB>` | Applies explicit transformation matrices to the two meshes. |

## GPU Memory and BVH Control

| Command | Description |
|---|---|
| `clear` | Frees GPU buffers and cuBQL trees while retaining the host-side meshes. |
| `reconstruct <qLevel> <rLevel>` | Reallocates device memory and reconstructs BVHs from cached CPU data. |

## Profiling and Diagnostics

| Command | Description |
|---|---|
| `profile` | Displays a graphical timing breakdown of the pipeline phases. |
| `stats` | Reports geometric precision statistics and triangle degeneracy information. |
| `scale` | Displays normalization and geometric scale information. |
| `runSuite` | Runs automated benchmark parameter sweeps. |
| `testConfig` | Configures and executes individual benchmark configurations. |
| `tbb <threads>` | Changes the number of CPU threads available to the CGAL exact predicate stage. |

---

# 7. GPU Predicate Evaluation Modes

The GPU predicate system supports multiple evaluation modes.

| Mode | Evaluation Pipeline | Purpose |
|---:|---|---|
| **0** | Float → CPU fallback | Basic GPU floating-point filtering. |
| **1** | Float → Simulated Double → CPU fallback | Increased numerical precision before CPU fallback. |
| **2** | Float → Inexact Big Integer quantization | Experimental GPU integer-based filtering. |
| **3** | Float → Exact Big Integer quantization → CPU fallback | Experimental exact integer-based GPU filtering. |
| **4** | Float → Real Double → CPU fallback | GPU floating-point filtering using double precision. |

These modes provide a framework for investigating the trade-off between GPU throughput, numerical precision, and CPU fallback frequency.

---

# 8. Numerical Robustness

A central challenge of GPU acceleration for computational geometry is that GPU floating-point arithmetic cannot simply replace the exact predicate machinery traditionally used by CGAL.

A fast numerical answer is not sufficient if an incorrect predicate classification can alter the topology of the resulting intersection.

The implementation therefore uses a staged approach.

### Broad Phase

The GPU first performs spatial pruning using bounding volumes.

Pairs that cannot intersect are removed without evaluating expensive geometric predicates.

### Numerical Filtering

Remaining triangle pairs are subjected to GPU numerical predicates.

Clearly classified cases can be resolved without CPU involvement.

### Exact Fallback

Numerically uncertain cases can be passed to CGAL's exact predicate implementation.

This creates a heterogeneous architecture in which the GPU handles the high-volume work while the CPU provides the final correctness guarantee for difficult geometric cases.

Conceptually:

```text
                         ALL TRIANGLE PAIRS
                                │
                                ▼
                       ┌─────────────────┐
                       │  GPU BVH Search │
                       └────────┬────────┘
                                │
                         candidate pairs
                                │
                                ▼
                       ┌─────────────────┐
                       │ GPU Predicates  │
                       └────────┬────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
             classified                   uncertain
                  │                           │
                  ▼                           ▼
               result                ┌────────────────┐
                                     │ CGAL Exact CPU │
                                     │   + TBB        │
                                     └───────┬────────┘
                                             │
                                             ▼
                                           result
```

The goal is not simply to replace CPU computation with GPU computation, but to use each architecture where it is most effective.

---

# 9. Current State and Visual Results

The current implementation is capable of tracking, computing, and visualizing exact triangle intersection curves while the input meshes are interactively translated and rotated.

The Polyscope viewport provides:

- interactive mesh visualization
- transformation gizmos
- mesh translation and rotation
- visualization of intersection curves
- real-time pipeline execution
- performance profiling
- geometric statistics

<div align="center">

<img src="images/PolyScope2026Aug22.png" alt="Polyscope interactive GPU mesh intersection demo" width="1000">

<br>

<em>Interactive Polyscope viewport showing transformed meshes and computed intersection geometry.</em>

</div>

The integrated `profile` functionality makes it possible to inspect the time spent in the different stages of the GPU/CPU pipeline.

This is useful for identifying bottlenecks in:

- BVH construction
- cross-checking
- dual-tree traversal
- single-tree traversal
- GPU predicate evaluation
- candidate transfer
- CPU exact predicate evaluation

---

# 10. Performance-Oriented Design

A major theme of the project was identifying where the actual computational bottlenecks occur rather than assuming that the GPU portion alone determines overall performance.

Several design decisions were introduced to reduce both computation and GPU memory traffic.

### Hierarchical Candidate Reduction

The dual-tree traversal aggressively removes pairs whose bounding volumes cannot intersect.

### Batched GPU Traversal

Candidate subtrees are grouped into batches to expose greater GPU parallelism.

### Partial Forest Construction

The `standaloneCompute` pipeline avoids constructing irrelevant portions of the hierarchy.

### GPU-Side Filtering

Cheap numerical predicates are evaluated before candidates are transferred to the CPU.

### Parallel Exact Fallback

Candidates that require exact evaluation can be processed using multiple CPU threads through Intel TBB.

Together, these techniques form a heterogeneous pipeline designed to minimize expensive work at every stage.

---

# 11. What's Left to Do

The current implementation establishes the spatial-search and intersection foundation, but several important extensions remain.

## Mesh Boolean Operations

The extracted intersection curves can serve as the basis for complete surface arrangement and mesh Boolean operations.

Future work includes:

- Union
- Intersection
- Difference / Subtraction
- Surface splitting
- Arrangement construction
- Consistent classification of resulting surface patches

## Academic Publication

A formal research paper would document:

- the dual-tree traversal strategy
- the batched single-tree traversal design
- partial forest construction
- GPU predicate filtering
- heterogeneous CPU/GPU execution
- performance characteristics
- comparisons with classical CPU approaches

## Upstream CGAL Integration

The implementation should be further refined into a generic CGAL-compatible interface.

This includes:

- stabilizing the API
- separating experimental infrastructure from reusable components
- providing appropriate CGAL-style templates and concepts
- documenting GPU requirements
- integrating the implementation with CGAL's package structure
- preparing the code for upstream review and eventual integration

---

# 12. Upstream Code Status

The project development is hosted on the CGAL public development repository.

### Development Branch

[`CGAL/cgal-public-dev: gsoc2026-gpu-spatial-search-YuryUoL`](https://github.com/CGAL/cgal-public-dev/tree/gsoc2026-gpu-spatial-search-YuryUoL)

The branch contains the implementation developed throughout the GSoC project.

### Standalone Repository

[`YuryUoL/gpu-cubql-intersector-standalone`](https://github.com/YuryUoL/gpu-cubql-intersector-standalone)

The standalone repository provides an environment for experimenting with and demonstrating the GPU intersection engine independently of the main CGAL development tree.

### Integration Status

The implementation currently resides in the dedicated GSoC feature branch.

Further API stabilization and CGAL-oriented packaging are required before the implementation can be considered ready for upstream integration into the main CGAL codebase.

---

# 13. Challenges & Lessons Learned

## CUDA Engineering and Algorithmic Design

One of the main outcomes of the project was gaining extensive experience in designing and implementing high-performance CUDA algorithms for computational geometry.

GPU programming introduces challenges that do not appear in ordinary CPU implementations, including:

- irregular tree traversal
- memory coalescing
- thread divergence
- dynamic candidate generation
- device-side memory management
- synchronization
- GPU occupancy
- balancing work between GPU and CPU

Designing the spatial traversal around these constraints required rethinking how hierarchical spatial searches can be mapped onto massively parallel hardware.

## Dual-Tree and Batched Single-Tree Traversal

Adapting dual-tree traversal to the GPU was one of the central algorithmic challenges.

Traditional recursive tree traversal does not map directly onto CUDA because recursive work can generate highly irregular amounts of computation.

The implementation therefore uses explicit traversal structures and parallel expansion to expose large amounts of thread-level parallelism.

The subsequent batched single-tree traversal provides another layer of parallelism when many independent subtree queries are available.

## Extending and Modifying cuBQL

The project required working deeply with **cuBQL**, rather than using the library purely as a black-box component.

The requirements of the CGAL intersection pipeline motivated changes and extensions involving:

- traversal routines
- tree-level access
- memory layouts
- multi-level cross-checking
- partial hierarchy construction
- customized traversal behavior

Working at this level provided practical experience with adapting a GPU spatial-search library to a specialized computational-geometry workload.

## Bridging GPU Performance and Exact Arithmetic

Another major challenge was reconciling GPU performance with the numerical guarantees expected from computational geometry.

A straightforward implementation could perform all predicates using GPU floating-point arithmetic, but this risks numerical errors near geometric degeneracies.

The project therefore investigated a hierarchy of numerical filters:

```text
Float
  ↓
Double / Extended Precision
  ↓
Integer-Based Filtering
  ↓
CGAL Exact Arithmetic
```

The important lesson was that GPU acceleration does not require abandoning exact computation.

Instead, inexpensive GPU filters can reduce the number of candidates that require expensive exact CPU evaluation.

## Memory and Construction Costs

Another important lesson was that GPU acceleration does not automatically make every part of the algorithm faster.

In particular, BVH construction and device-memory allocation can become significant components of total runtime.

This motivated the development of the partial forest approach, which avoids constructing hierarchy regions that are irrelevant to the current intersection query.

The resulting design treats memory consumption and construction time as first-class components of the algorithm rather than secondary implementation details.

---

# 14. Key Contributions

The main contributions of the project can be summarized as follows.

### GPU Spatial Search

A CUDA-based spatial-search pipeline was developed around GPU BVHs and cuBQL.

### Dual-Tree Traversal

A parallel dual-tree traversal strategy was developed for efficiently reducing overlapping BVH node pairs.

### Batched Single-Tree Traversal

A second traversal stage was developed to process many subtree queries in parallel and efficiently generate triangle-level candidates.

### Partial Forest Pipeline

A reduced hierarchy representation was developed to avoid unnecessary construction and memory usage for localized intersection problems.

### GPU Predicate Experiments

Multiple GPU predicate evaluation modes were implemented to explore the trade-off between speed and numerical precision.

### Exact CPU Fallback

The GPU filtering stages were integrated with CGAL exact predicates, allowing uncertain cases to be handled using robust CPU arithmetic.

### Interactive Visualization

A Polyscope-based interactive environment was developed for inspecting mesh transformations, intersection curves, and pipeline performance.

### Profiling and Benchmarking

The application includes tools for timing pipeline stages and sweeping traversal and predicate configurations.

---

# 15. Conclusion

This project explored how GPU massively parallel computation can be combined with the exact geometric machinery of CGAL to accelerate surface-mesh intersection.

The resulting architecture separates the problem into stages that are well suited to different hardware:

```text
                    GPU
                     │
        ┌────────────┴────────────┐
        │                         │
   Spatial Search          Numerical Filtering
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
              Candidate Pairs
                     │
                     ▼
                    CPU
                     │
             Exact Predicates
                     │
                     ▼
             Correct Geometry
```

The project demonstrated that high-throughput GPU spatial searching and exact computational geometry can be combined in a single heterogeneous pipeline.

The GPU provides the parallel throughput required to process large numbers of spatial candidates, while CGAL provides the robust exact predicates necessary for reliable geometric computation.

The work establishes a foundation for further development toward GPU-accelerated spatial-search and mesh-processing functionality within CGAL.

---

# 16. Acknowledgements

I would like to thank the **CGAL community and mentors** for their guidance and support throughout Google Summer of Code 2026.

The project provided an opportunity to work at the intersection of:

- computational geometry
- GPU programming
- spatial data structures
- exact arithmetic
- high-performance computing

and to explore how these areas can be combined into practical high-performance geometric algorithms.

---

<div align="center">

## Google Summer of Code 2026

### GPU-Accelerated Spatial Searching & Mesh Intersection Engine

**YuryUoL · CGAL**

[Development Branch](https://github.com/CGAL/cgal-public-dev/tree/gsoc2026-gpu-spatial-search-YuryUoL)
&nbsp; · &nbsp;
[Standalone Repository](https://github.com/YuryUoL/gpu-cubql-intersector-standalone)

</div>
