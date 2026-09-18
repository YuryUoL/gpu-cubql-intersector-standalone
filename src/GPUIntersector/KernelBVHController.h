#pragma once

// ============================================================================
// KernelBVHController.h
//
// This header is intentionally free of CUDA/Thrust/cuBQL/TBB headers so that
// it can be #included and compiled from plain CPU-side translation units
// (i.e. with a normal host compiler, not nvcc). All CUDA-resident state
// (device pointers, cuBQL BVH structs, Thrust vectors, the CUDA stream,
// etc.) lives in a private, forward-declared `Impl` struct that is only
// ever defined in KernelBVHController.cu. From the outside, this class is
// just a handle (via std::unique_ptr<Impl>) - a classic Pimpl / opaque
// pointer pattern.
//
// NOTE on <vector_types.h>: double3 / uint3 / int2 are plain-old-data
// structs declared in the CUDA Toolkit's <vector_types.h>. That header has
// no device code and no dependency on nvcc, Thrust, or cuBQL - it's safe
// to include from a host-only compiler as long as the CUDA include path is
// visible to the build. If you'd rather not depend on any CUDA header at
// all from CPU-only translation units, swap these for your own POD
// mirrors (e.g. a `struct Double3 { double x, y, z; };`) and convert at
// the .cu boundary instead.
// ============================================================================

#include <memory>
#include <vector_types.h>  // double3, uint3, int2 (POD only, no device code)

// CPU-side project types that appear by value / by reference in the public
// API. These headers must stay CUDA/Thrust/cuBQL free themselves for this
// header to remain safely includable from CPU-only code.
#include "CPU/CgalDefinitions.h"   // Mesh, Point3
#include "common/ExecutionStats.h"  // ExecutionStats

class KernelBVHController {
public:
    KernelBVHController();
    ~KernelBVHController();

    // Prevent accidental copying (the class owns GPU resources)
    KernelBVHController(const KernelBVHController&) = delete;
    KernelBVHController& operator=(const KernelBVHController&) = delete;

    // 1. Setup & Construction - Double Precision Only
    void construct(
        Mesh& meshAcpu, Mesh& meshBcpu,
        const Point3& centerA, const Point3& centerB,
        const double3* hVertsA, int numVertsA, const uint3* hIndicesA, int numTrianglesA, int levelA,
        const double3* hVertsB, int numVertsB, const uint3* hIndicesB, int numTrianglesB, int levelB,
        int leafThreshold, ExecutionStats& stats);

    void construct(
        Mesh& meshAcpu, Mesh& meshBcpu,
        const double3* hVertsA, int numVertsA, const uint3* hIndicesA, int numTrianglesA, int levelA,
        const double3* hVertsB, int numVertsB, const uint3* hIndicesB, int numTrianglesB, int levelB,
        int leafThreshold, ExecutionStats& stats);

    // 2. Execution Pipeline - Double Precision Only
    void runIntersectionPipeline(
        int batchMultiplier, int numberOfDualTreeSteps, int activateAsyncDownload,
        int2*& outFinalExactPairs,       // Fast raw pointer return
        size_t& outFinalCount, ExecutionStats& stats, int gpuDouble);

    // 3. Deallocates all GPU resources safely
    void cleanup();
    void clearGPU();
    void reconstructGPU(ExecutionStats& stats, int a, int b);
    void reconstructGPU(ExecutionStats& stats);

    // Dynamic Dual Point Cloud Transformation
    void setTransformBoth(double3 rotDegA, double3 transA,
                          double3 rotDegB, double3 transB,
                          float& timeGPU, float& timeCPU,
                          float& timeTransformVerts,
                          float& timeAssembleTris,
                          float& timeGenBoxes);

    // Legacy Translation Interfaces
    void setTranslation(double xB, double yB, double zB);
    // void setTranslationCPUHostUpload(double xB, double yB, double zB);

    // Centroid Getters (defined in the .cu - Impl is incomplete here)
    Point3 getCenterA() const;
    Point3 getCenterB() const;

    bool isGPUAllocated() const;

private:
    // Everything CUDA/Thrust/cuBQL/TBB-related (device pointers, streams,
    // cuBQL BVH structs, Thrust device_vectors, cached host pointers, etc.)
    // is defined only in KernelBVHController.cu.
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};