## 🚀 High-Performance Build Instructions (Crucial for Evaluation)

This application implements a high-performance hybrid GPU/CPU mesh intersection pipeline:
* **GPU Broad-Phase Search:** Uses **cuBQL** for CUDA-accelerated Bounding Volume Hierarchy (BVH) construction and dual-tree spatial traversal.
* **Exact CPU Filtering:** Leverages **CGAL** exact geometric predicates (`CGAL_Core`, GMP/MPFR, TBB) for exact triangle intersection resolution and curve extraction without numerical drift.
* **Interactive Viewport:** Uses **Polyscope** for real-time 3D visualization, allowing dynamic translation and rotation of surface meshes while displaying live intersection curves.

Because CGAL and cuBQL rely heavily on arithmetic interval filtering and template-heavy multi-precision operations, compiling in **Debug mode results in significant performance degradation**. To evaluate the true speed of the pipeline, the project **must** be built in `Release` mode. Compiler optimizations (`-O3 -march=native --use_fast_math -DNDEBUG`) are injected automatically by target helper functions in CMake.

### Prerequisites
* **CUDA Toolkit:** 12.x+ (default target architecture: `120` / Blackwell, configurable via `-DCMAKE_CUDA_ARCHITECTURES`)
* **CGAL:** 5.x+ (`Core` component)
* **System Dependencies:** TBB, GMP, MPFR, `readline` (`pkg-config`)
* **cuBQL:** Shared libraries (`libcuBQL_cuda_float2.so`, `libcuBQL_cuda_float3.so`)

---

### Compilation Sequence

Use explicit source (`-S .`) and build (`-B build`) target flags to keep the directory clean. Custom dependency locations can be passed using `-D` flags if CGAL or cuBQL are not installed in standard system paths:

```bash
# 1. Configure the standalone build in Release mode
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCGAL_DIR=/path/to/cgal-main \
      -DCUBQL_ROOT=/path/to/Cubql \
      -DCUBQL_BUILD=/path/to/Cubql/build

# 2. Compile the primary interactive executable
cmake --build build --target MainApp -j$(nproc)
