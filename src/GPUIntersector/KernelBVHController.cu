#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1

// Standard C++ System Headers
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

// CUDA & Thrust Headers
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>

// Intel TBB Headers
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

// Controller Interface (clean, CPU-facing)
#include "GPUIntersector/KernelBVHController.h"

// Core cuBQL & External Utilities
// NOTE: bvh.h must be included before builder/cuda.h - cuda.h's templates
// reference BinaryBVH<T,D>/box_t<T,D>, which bvh.h declares. Getting this
// backwards causes cuda.h to fail to parse, corrupting the rest of the
// cuBQL namespace parse (this is what produced the "BinaryBVH is undefined" /
// "cuBQL::Triangle has no member" cascade).
#include <cuBQL/bvh.h>
#include <cuBQL/builder/cuda.h>
#include <loadOBJ.h>
#include <third-party/cubql/fixedBoxQueryv2.h>
#include <third-party/cubql/level_cut_builder.h>
#include <third-party/cubql/refit_forest.h>

// Custom Project Modules
#include "GPU_exact_predicates/TriangleDouble.h"
#include "common/utils.h"
#include "single_tree/SingleTreeBatchIntersector.h"
#include "traversal/DualTreeStep.h"
#include "traversal/crossCheck.h"
#include "traversal/rapidDescendKernel.h"

#ifndef _ALLOC
#define _ALLOC(ptr, count, stream, memResource) \
  CUBQL_CUDA_CALL(MallocAsync((void**)&(ptr), (size_t)(count) * sizeof(*(ptr)), stream))
#endif

#ifndef _FREE
#define _FREE(ptr, stream, memResource) CUBQL_CUDA_CALL(FreeAsync((ptr), stream))
#endif

// ============================================================================
// KernelBVHController::Impl
//
// All CUDA-resident / library-specific state lives here. This struct is
// only ever seen from this translation unit, so KernelBVHController.h stays
// free of CUDA, Thrust, cuBQL and TBB includes and can be consumed by plain
// CPU-side code.
// ============================================================================
struct KernelBVHController::Impl {
    // Configuration Parameters
    int m_leafThreshold = 0;
    int m_levelA = 0;
    int m_levelB = 0;
    int m_numTrianglesA = 0;
    int m_numTrianglesB = 0;
    int m_numVertsA = 0;
    int m_numVertsB = 0;

    // Pre-computed Mesh Centroids
    Point3 m_centerA{0, 0, 0};
    Point3 m_centerB{0, 0, 0};

    // Active transformations relative to pristine baseline states
    double3 m_rotA{0.0, 0.0, 0.0};
    double3 m_transA{0.0, 0.0, 0.0};
    double3 m_rotB{0.0, 0.0, 0.0};
    double3 m_transB{0.0, 0.0, 0.0};

    // Legacy shift markers
    double shiftX = 0.0;
    double shiftY = 0.0;
    double shiftZ = 0.0;

    Mesh* m_meshAcpu = nullptr;
    Mesh* m_meshBcpu = nullptr;

    // CUDA Streams & Memory Resource
    cudaStream_t m_stream = nullptr;
    cuBQL::DeviceMemoryResource m_memResource;

    // Core Geometry & Topology Persistent Device Buffers
    cuBQL::Triangle* m_dMeshA = nullptr;
    float2*          m_dMeshMetricsA = nullptr;
    cuBQL::box3f*    m_dBoxesA = nullptr;

    cuBQL::Triangle* m_dMeshB = nullptr;
    float2*          m_dMeshMetricsB = nullptr;
    cuBQL::box3f*    m_dBoxesB = nullptr;

    // Cached host data needed for GPU reconstruction
    const double3* m_hVertsA = nullptr;
    const uint3*   m_hIndicesA = nullptr;
    const double3* m_hVertsB = nullptr;
    const uint3*   m_hIndicesB = nullptr;

    // Persistent Raw GPU & CPU Buffers for Mesh A
    double3*             m_dVertsA = nullptr;       // Active transformed vertices
    double3*             m_dVertsAOrig = nullptr;   // Pristine baseline vertices
    uint3*               m_dIndicesA = nullptr;     // Triangle vertex indices
    float*               m_dVertErrorsA = nullptr;  // Precision error bounds
    std::vector<Point3>  m_origPointsA;             // Baseline CGAL host points

    // Persistent Raw GPU & CPU Buffers for Mesh B
    double3*             m_dVertsB = nullptr;       // Active transformed vertices
    double3*             m_dVertsBOrig = nullptr;   // Pristine baseline vertices
    uint3*               m_dIndicesB = nullptr;     // Triangle vertex indices
    float*               m_dVertErrorsB = nullptr;  // Precision error bounds
    std::vector<Point3>  m_origPointsB;             // Baseline CGAL host points

    // BVH Structures (cuBQL v2_2 format)
    cuBQL::bvh3f m_bvhA;
    cuBQL::bvh3f m_bvhB;

    uint32_t m_hOutMarkedCountA_Full = 0;
    uint32_t m_hOutMarkedCountB_Full = 0;
    thrust::device_vector<uint32_t> m_dMarkedNodeIndicesA_Full;
    thrust::device_vector<uint32_t> m_dMarkedNodeIndicesB_Full;

    // Persistent Thrust Vectors
    thrust::device_vector<uint32_t> m_dNodeDescendantCountsA;
    thrust::device_vector<uint32_t> m_dNodeDescendantCountsB;

    thrust::device_vector<uint32_t> m_dReverseMapB;
    thrust::device_vector<uint32_t> m_dOutPairsA;
    thrust::device_vector<uint32_t> m_dOutPairsB;

    thrust::device_vector<uint32_t> m_dOutOffsetsA;
    thrust::device_vector<uint32_t> m_dOutPrimsFlatA;
    thrust::device_vector<uint32_t> m_dOutOffsetsB;
    thrust::device_vector<uint32_t> m_dOutPrimsFlatB;
};

// ============================================================================
// CPU Host Helpers (Parallel TBB Transformations)
// ============================================================================

/**
 * @brief Applies rigid body transformation to a CGAL Surface Mesh in parallel.
 * @param mesh Pointer to the target CGAL Surface Mesh.
 * @param origPoints Vector of original vertex coordinates in canonical space.
 * @param center Center of mass / pivot point for rotation.
 * @param rotDeg Euler rotation angles in degrees (X, Y, Z).
 * @param trans Translation offset vector.
 */
static void transformCgalMesh(Mesh* mesh,
                               const std::vector<Point3>& origPoints,
                               Point3 center,
                               double3 rotDeg,
                               double3 trans) {
  if(!mesh || origPoints.empty()) return;

  auto pmap = mesh->points();
  const size_t numPoints = mesh->num_vertices();

  // Precompute trigonometric terms for standard Euler rotation (Rz * Ry * Rx)
  constexpr double DEG_TO_RAD = 3.14159265358979323846 / 180.0;
  const double radX = rotDeg.x * DEG_TO_RAD;
  const double radY = rotDeg.y * DEG_TO_RAD;
  const double radZ = rotDeg.z * DEG_TO_RAD;

  const double cx = std::cos(radX), sx = std::sin(radX);
  const double cy = std::cos(radY), sy = std::sin(radY);
  const double cz = std::cos(radZ), sz = std::sin(radZ);

  const double r00 = cy * cz, r01 = sx * sy * cz - cx * sz, r02 = cx * sy * cz + sx * sz;
  const double r10 = cy * sz, r11 = sx * sy * sz + cx * cz, r12 = cx * sy * sz - sx * cz;
  const double r20 = -sy,    r21 = sx * cy,               r22 = cx * cy;

  const double cx_p = center.x(), cy_p = center.y(), cz_p = center.z();

  tbb::parallel_for(tbb::blocked_range<size_t>(0, numPoints),
                    [mesh, &pmap, &origPoints, r00, r01, r02, r10, r11, r12, r20, r21, r22,
                     cx_p, cy_p, cz_p, trans](const tbb::blocked_range<size_t>& range) {
                      for(size_t i = range.begin(); i != range.end(); ++i) {
                        Mesh::Vertex_index vd(static_cast<uint32_t>(i));
                        if(!mesh->is_removed(vd)) {
                          const Point3& pOrig = origPoints[i];
                          const double x = pOrig.x() - cx_p;
                          const double y = pOrig.y() - cy_p;
                          const double z = pOrig.z() - cz_p;

                          const double rx = r00 * x + r01 * y + r02 * z;
                          const double ry = r10 * x + r11 * y + r12 * z;
                          const double rz = r20 * x + r21 * y + r22 * z;

                          pmap[vd] = Point3(rx + cx_p + trans.x,
                                            ry + cy_p + trans.y,
                                            rz + cz_p + trans.z);
                        }
                      }
                    });
}

/**
 * @brief Translates CGAL mesh in parallel and extracts double3 host array for GPU uploads.
 */
static void translateCgalAndExtractDouble3(Mesh* mesh,
                                          const std::vector<Point3>& origPoints,
                                          double3* hOutVerts,
                                          double xB, double yB, double zB) {
  if(!mesh || origPoints.empty() || !hOutVerts) return;

  auto pmap = mesh->points();
  const size_t numPoints = mesh->num_vertices();

  tbb::parallel_for(tbb::blocked_range<size_t>(0, numPoints),
                    [mesh, &pmap, &origPoints, hOutVerts, xB, yB, zB](const tbb::blocked_range<size_t>& range) {
                      for(size_t i = range.begin(); i != range.end(); ++i) {
                        Mesh::Vertex_index vd(static_cast<uint32_t>(i));
                        if(!mesh->is_removed(vd)) {
                          const Point3& pOrig = origPoints[i];
                          const Point3 pTranslated(pOrig.x() + xB, pOrig.y() + yB, pOrig.z() + zB);

                          pmap[vd] = pTranslated;
                          hOutVerts[i] = make_double3(pTranslated.x(), pTranslated.y(), pTranslated.z());
                        }
                      }
                    });
}

// ============================================================================
// Controller Construction & Destruction
// ============================================================================

KernelBVHController::KernelBVHController() : m_impl(std::make_unique<Impl>()) {
  CUBQL_CUDA_CALL(StreamCreate(&m_impl->m_stream));
}

KernelBVHController::~KernelBVHController() {
  cleanup();
  if(m_impl->m_stream) {
    cudaStreamDestroy(m_impl->m_stream);
    m_impl->m_stream = nullptr;
  }
}

// ============================================================================
// Simple Accessors
// ============================================================================

Point3 KernelBVHController::getCenterA() const { return m_impl->m_centerA; }
Point3 KernelBVHController::getCenterB() const { return m_impl->m_centerB; }

bool KernelBVHController::isGPUAllocated() const {
  return (m_impl->m_dVertsA != nullptr && m_impl->m_dVertsB != nullptr);
}

// ============================================================================
// Dynamic Rigid Body Transformation Pipeline
// ============================================================================

void KernelBVHController::setTransformBoth(double3 rotDegA,
                                           double3 transA,
                                           double3 rotDegB,
                                           double3 transB,
                                           float& timeGPU,
                                           float& timeCPU,
                                           float& timeTransformVerts,
                                           float& timeAssembleTris,
                                           float& timeGenBoxes) {
  timeGPU = 0.0f;
  timeCPU = 0.0f;
  timeTransformVerts = 0.0f;
  timeAssembleTris = 0.0f;
  timeGenBoxes = 0.0f;

  const bool movedA = (rotDegA.x != m_impl->m_rotA.x || rotDegA.y != m_impl->m_rotA.y || rotDegA.z != m_impl->m_rotA.z ||
                       transA.x  != m_impl->m_transA.x || transA.y  != m_impl->m_transA.y || transA.z  != m_impl->m_transA.z);

  const bool movedB = (rotDegB.x != m_impl->m_rotB.x || rotDegB.y != m_impl->m_rotB.y || rotDegB.z != m_impl->m_rotB.z ||
                       transB.x  != m_impl->m_transB.x || transB.y  != m_impl->m_transB.y || transB.z  != m_impl->m_transB.z);

  if(!movedA && !movedB) return;

  const bool runA = (movedA && m_impl->m_numVertsA > 0 && m_impl->m_dVertsA && m_impl->m_dVertsAOrig);
  const bool runB = (movedB && m_impl->m_numVertsB > 0 && m_impl->m_dVertsB && m_impl->m_dVertsBOrig);
  const bool gpuWorkQueued = runA || runB;

  cudaEvent_t evStartGPU, evStopGPU;
  cudaEvent_t evStartTV, evStopTV;
  cudaEvent_t evStartAssemble, evStopAssemble;
  cudaEvent_t evStartRefit, evStopRefit;

  if(gpuWorkQueued) {
    cudaEventCreate(&evStartGPU);
    cudaEventCreate(&evStopGPU);
    cudaEventCreate(&evStartTV);
    cudaEventCreate(&evStopTV);
    cudaEventCreate(&evStartAssemble);
    cudaEventCreate(&evStopAssemble);
    cudaEventCreate(&evStartRefit);
    cudaEventCreate(&evStopRefit);

    cudaEventRecord(evStartGPU, m_impl->m_stream);

    // 1. Transform Vertices on GPU
    cudaEventRecord(evStartTV, m_impl->m_stream);
    if(runA) {
      Mat3x3 RA = makeRotationMatrixDeg(rotDegA.x, rotDegA.y, rotDegA.z);
      double3 cA = make_double3(m_impl->m_centerA.x(), m_impl->m_centerA.y(), m_impl->m_centerA.z());
      launchTransformVertices(m_impl->m_dVertsA, m_impl->m_dVertsAOrig, m_impl->m_numVertsA, RA, cA, transA, m_impl->m_stream);
    }
    if(runB) {
      Mat3x3 RB = makeRotationMatrixDeg(rotDegB.x, rotDegB.y, rotDegB.z);
      double3 cB = make_double3(m_impl->m_centerB.x(), m_impl->m_centerB.y(), m_impl->m_centerB.z());
      launchTransformVertices(m_impl->m_dVertsB, m_impl->m_dVertsBOrig, m_impl->m_numVertsB, RB, cB, transB, m_impl->m_stream);
    }
    cudaEventRecord(evStopTV, m_impl->m_stream);

    // 2. Generate Bounding Boxes
    cudaEventRecord(evStartAssemble, m_impl->m_stream);
    if(runA) {
      launchGenerateBoxesTris(m_impl->m_dBoxesA, m_impl->m_dVertsA, m_impl->m_dIndicesA, m_impl->m_numTrianglesA, m_impl->m_stream);
    }
    if(runB) {
      launchGenerateBoxesTris(m_impl->m_dBoxesB, m_impl->m_dVertsB, m_impl->m_dIndicesB, m_impl->m_numTrianglesB, m_impl->m_stream);
    }
    cudaEventRecord(evStopAssemble, m_impl->m_stream);

    // 3. Refit cuBQL Acceleration Trees
    cudaEventRecord(evStartRefit, m_impl->m_stream);
    if(runA) {
      cuBQL::cuda::refit(m_impl->m_bvhA, m_impl->m_dBoxesA, m_impl->m_stream);
    }
    if(runB) {
      cuBQL::cuda::refit(m_impl->m_bvhB, m_impl->m_dBoxesB, m_impl->m_stream);
    }
    cudaEventRecord(evStopRefit, m_impl->m_stream);

    cudaEventRecord(evStopGPU, m_impl->m_stream);
  }

  if(movedA) {
    m_impl->m_rotA = rotDegA;
    m_impl->m_transA = transA;
  }
  if(movedB) {
    m_impl->m_rotB = rotDegB;
    m_impl->m_transB = transB;
    m_impl->shiftX = transB.x;
    m_impl->shiftY = transB.y;
    m_impl->shiftZ = transB.z;
  }

  // 4. Synchronize GPU & Collect Execution Timings
  if(gpuWorkQueued) {
    cudaEventSynchronize(evStopGPU);

    cudaEventElapsedTime(&timeGPU, evStartGPU, evStopGPU);
    cudaEventElapsedTime(&timeTransformVerts, evStartTV, evStopTV);
    cudaEventElapsedTime(&timeAssembleTris, evStartAssemble, evStopAssemble);
    cudaEventElapsedTime(&timeGenBoxes, evStartRefit, evStopRefit);

    cudaEventDestroy(evStartGPU);
    cudaEventDestroy(evStopGPU);
    cudaEventDestroy(evStartTV);
    cudaEventDestroy(evStopTV);
    cudaEventDestroy(evStartAssemble);
    cudaEventDestroy(evStopAssemble);
    cudaEventDestroy(evStartRefit);
    cudaEventDestroy(evStopRefit);
  }
}

void KernelBVHController::setTranslation(double xB, double yB, double zB) {
  float tGPU, tCPU, tTV, tAT, tGB;
  setTransformBoth(m_impl->m_rotA, m_impl->m_transA, m_impl->m_rotB, make_double3(xB, yB, zB), tGPU, tCPU, tTV, tAT, tGB);
}

// ============================================================================
// BVH Construction & Resource Allocation
// ============================================================================

void KernelBVHController::construct(Mesh& meshAcpu,
                                    Mesh& meshBcpu,
                                    const double3* hVertsA,
                                    int numVertsA,
                                    const uint3* hIndicesA,
                                    int numTrianglesA,
                                    int levelA,
                                    const double3* hVertsB,
                                    int numVertsB,
                                    const uint3* hIndicesB,
                                    int numTrianglesB,
                                    int levelB,
                                    int leafThreshold,
                                    ExecutionStats& stats) {
  construct(meshAcpu, meshBcpu, Point3(0, 0, 0), Point3(0, 0, 0), hVertsA, numVertsA, hIndicesA,
            numTrianglesA, levelA, hVertsB, numVertsB, hIndicesB, numTrianglesB, levelB,
            leafThreshold, stats);
}

void KernelBVHController::construct(Mesh& meshAcpu,
                                    Mesh& meshBcpu,
                                    const Point3& centerA,
                                    const Point3& centerB,
                                    const double3* hVertsA,
                                    int numVertsA,
                                    const uint3* hIndicesA,
                                    int numTrianglesA,
                                    int levelA,
                                    const double3* hVertsB,
                                    int numVertsB,
                                    const uint3* hIndicesB,
                                    int numTrianglesB,
                                    int levelB,
                                    int leafThreshold,
                                    ExecutionStats& stats) {
  if(numTrianglesA <= 0 || numTrianglesB <= 0) return;

  cleanup();

  m_impl->m_meshAcpu = &meshAcpu;
  m_impl->m_meshBcpu = &meshBcpu;
  m_impl->m_numTrianglesA = numTrianglesA;
  m_impl->m_numVertsA = numVertsA;
  m_impl->m_levelA = levelA;
  m_impl->m_numTrianglesB = numTrianglesB;
  m_impl->m_numVertsB = numVertsB;
  m_impl->m_levelB = levelB;
  m_impl->m_leafThreshold = leafThreshold;

  m_impl->m_hVertsA = hVertsA;
  m_impl->m_hIndicesA = hIndicesA;
  m_impl->m_hVertsB = hVertsB;
  m_impl->m_hIndicesB = hIndicesB;

  m_impl->m_centerA = centerA;
  m_impl->m_centerB = centerB;

  m_impl->m_rotA = make_double3(0.0, 0.0, 0.0);
  m_impl->m_transA = make_double3(0.0, 0.0, 0.0);
  m_impl->m_rotB = make_double3(0.0, 0.0, 0.0);
  m_impl->m_transB = make_double3(0.0, 0.0, 0.0);
  m_impl->shiftX = m_impl->shiftY = m_impl->shiftZ = 0.0;

  cuBQL::BuildConfig buildConfig;
  buildConfig.makeLeafThreshold = m_impl->m_leafThreshold;

  cudaEvent_t evAllocStart, evAllocStop;
  cudaEvent_t evBuildAStart, evBuildAStop;
  cudaEvent_t evBuildBStart, evBuildBStop;

  CUBQL_CUDA_CALL(EventCreate(&evAllocStart));
  CUBQL_CUDA_CALL(EventCreate(&evAllocStop));
  CUBQL_CUDA_CALL(EventCreate(&evBuildAStart));
  CUBQL_CUDA_CALL(EventCreate(&evBuildAStop));
  CUBQL_CUDA_CALL(EventCreate(&evBuildBStart));
  CUBQL_CUDA_CALL(EventCreate(&evBuildBStop));

  CUBQL_CUDA_CALL(EventRecord(evAllocStart, m_impl->m_stream));

  // Allocate and upload Mesh A GPU buffers
  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dVertsA, m_impl->m_numVertsA * sizeof(double3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dVertsA, hVertsA, m_impl->m_numVertsA * sizeof(double3), cudaMemcpyHostToDevice, m_impl->m_stream));

  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dVertsAOrig, m_impl->m_numVertsA * sizeof(double3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dVertsAOrig, hVertsA, m_impl->m_numVertsA * sizeof(double3), cudaMemcpyHostToDevice, m_impl->m_stream));

  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dIndicesA, m_impl->m_numTrianglesA * sizeof(uint3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dIndicesA, hIndicesA, m_impl->m_numTrianglesA * sizeof(uint3), cudaMemcpyHostToDevice, m_impl->m_stream));

  m_impl->m_dVertErrorsA = nullptr;

  // Allocate and upload Mesh B GPU buffers
  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dVertsB, m_impl->m_numVertsB * sizeof(double3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dVertsB, hVertsB, m_impl->m_numVertsB * sizeof(double3), cudaMemcpyHostToDevice, m_impl->m_stream));

  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dVertsBOrig, m_impl->m_numVertsB * sizeof(double3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dVertsBOrig, hVertsB, m_impl->m_numVertsB * sizeof(double3), cudaMemcpyHostToDevice, m_impl->m_stream));

  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dIndicesB, m_impl->m_numTrianglesB * sizeof(uint3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dIndicesB, hIndicesB, m_impl->m_numTrianglesB * sizeof(uint3), cudaMemcpyHostToDevice, m_impl->m_stream));

  m_impl->m_dVertErrorsB = nullptr;

  // Allocate AABB bounding box buffers
  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dBoxesA, m_impl->m_numTrianglesA * sizeof(cuBQL::box3f), m_impl->m_stream));
  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dBoxesB, m_impl->m_numTrianglesB * sizeof(cuBQL::box3f), m_impl->m_stream));

  // Compute initial primitive bounding boxes
  generateBoxesTrisKernel<<<cuBQL::divRoundUp(m_impl->m_numTrianglesA, 256), 256, 0, m_impl->m_stream>>>(
      m_impl->m_dBoxesA, m_impl->m_dVertsA, m_impl->m_dIndicesA, m_impl->m_numTrianglesA);

  generateBoxesTrisKernel<<<cuBQL::divRoundUp(m_impl->m_numTrianglesB, 256), 256, 0, m_impl->m_stream>>>(
      m_impl->m_dBoxesB, m_impl->m_dVertsB, m_impl->m_dIndicesB, m_impl->m_numTrianglesB);

  CUBQL_CUDA_CALL(EventRecord(evAllocStop, m_impl->m_stream));

  // Allocate Thrust dynamic tracking vectors
  const double tThrustInitStart = cuBQL::getCurrentTime();
  const uint32_t maxNodesA = 2 * m_impl->m_numTrianglesA;
  const uint32_t maxNodesB = 2 * m_impl->m_numTrianglesB;

  m_impl->m_dMarkedNodeIndicesA_Full.resize(maxNodesA, 0);
  m_impl->m_dNodeDescendantCountsA.resize(maxNodesA, 0);
  m_impl->m_dMarkedNodeIndicesB_Full.resize(maxNodesB, 0);
  m_impl->m_dNodeDescendantCountsB.resize(maxNodesB, 0);
  m_impl->m_dReverseMapB.resize(maxNodesB, 0);

  stats.thrustInitOverheadMs = (cuBQL::getCurrentTime() - tThrustInitStart) * 1000.0;

  // Build custom level-cut BVHs and perform initial refit
  CUBQL_CUDA_CALL(EventRecord(evBuildAStart, m_impl->m_stream));
  cuBQL::ext::level_cut::build_custom(m_impl->m_bvhA, m_impl->m_dBoxesA, m_impl->m_numTrianglesA, buildConfig, (uint32_t)m_impl->m_levelA,
                                       thrust::raw_pointer_cast(m_impl->m_dMarkedNodeIndicesA_Full.data()),
                                       thrust::raw_pointer_cast(m_impl->m_dNodeDescendantCountsA.data()),
                                       &m_impl->m_hOutMarkedCountA_Full, m_impl->m_stream, m_impl->m_memResource);
  cuBQL::cuda::refit(m_impl->m_bvhA, m_impl->m_dBoxesA, m_impl->m_stream);
  CUBQL_CUDA_CALL(EventRecord(evBuildAStop, m_impl->m_stream));

  CUBQL_CUDA_CALL(EventRecord(evBuildBStart, m_impl->m_stream));
  cuBQL::ext::level_cut::build_custom(m_impl->m_bvhB, m_impl->m_dBoxesB, m_impl->m_numTrianglesB, buildConfig, (uint32_t)m_impl->m_levelB,
                                       thrust::raw_pointer_cast(m_impl->m_dMarkedNodeIndicesB_Full.data()),
                                       thrust::raw_pointer_cast(m_impl->m_dNodeDescendantCountsB.data()),
                                       &m_impl->m_hOutMarkedCountB_Full, m_impl->m_stream, m_impl->m_memResource);
  cuBQL::cuda::refit(m_impl->m_bvhB, m_impl->m_dBoxesB, m_impl->m_stream);
  CUBQL_CUDA_CALL(EventRecord(evBuildBStop, m_impl->m_stream));

  CUBQL_CUDA_CALL(StreamSynchronize(m_impl->m_stream));

  float msAlloc = 0.0f, msBuildA = 0.0f, msBuildB = 0.0f;
  CUBQL_CUDA_CALL(EventElapsedTime(&msAlloc, evAllocStart, evAllocStop));
  CUBQL_CUDA_CALL(EventElapsedTime(&msBuildA, evBuildAStart, evBuildAStop));
  CUBQL_CUDA_CALL(EventElapsedTime(&msBuildB, evBuildBStart, evBuildBStop));

  stats.initialAllocAndCopyMs = static_cast<double>(msAlloc);
  stats.buildRefitMeshAMs = static_cast<double>(msBuildA);
  stats.buildRefitMeshBMs = static_cast<double>(msBuildB);

  CUBQL_CUDA_CALL(EventDestroy(evAllocStart));
  CUBQL_CUDA_CALL(EventDestroy(evAllocStop));
  CUBQL_CUDA_CALL(EventDestroy(evBuildAStart));
  CUBQL_CUDA_CALL(EventDestroy(evBuildAStop));
  CUBQL_CUDA_CALL(EventDestroy(evBuildBStart));
  CUBQL_CUDA_CALL(EventDestroy(evBuildBStop));
}

// ============================================================================
// Intersection Execution Pipeline
// ============================================================================

void KernelBVHController::runIntersectionPipeline(int batchMultiplier,
                                                  int numberOfDualTreeSteps,
                                                  int activateAsyncDownload,
                                                  int2*& outFinalExactPairs,
                                                  size_t& outFinalCount,
                                                  ExecutionStats& stats,
                                                  int gpuDouble) {
  const double tPipelineStart = cuBQL::getCurrentTime();

  m_impl->m_dOutPairsA.clear();
  m_impl->m_dOutPairsB.clear();

  uint32_t m_hOutMarkedCountA = m_impl->m_hOutMarkedCountA_Full;
  uint32_t m_hOutMarkedCountB = m_impl->m_hOutMarkedCountB_Full;
  thrust::device_vector<uint32_t> m_dMarkedNodeIndicesA = m_impl->m_dMarkedNodeIndicesA_Full;
  thrust::device_vector<uint32_t> m_dMarkedNodeIndicesB = m_impl->m_dMarkedNodeIndicesB_Full;

  // Stage 1: Level-Cut Criss-Cross BVH Cross-Check
  const double tCrossStart = cuBQL::getCurrentTime();
  const uint32_t totalIntersections = executeCrissCrossIntersection(
      m_impl->m_bvhA, m_dMarkedNodeIndicesA, m_hOutMarkedCountA,
      m_impl->m_bvhB, m_dMarkedNodeIndicesB, m_hOutMarkedCountB,
      m_impl->m_dOutPairsA, m_impl->m_dOutPairsB);

  CUBQL_CUDA_CALL(StreamSynchronize(m_impl->m_stream));
  const double tCrossEnd = cuBQL::getCurrentTime();

  const uint64_t totalPossiblePairs = static_cast<uint64_t>(m_hOutMarkedCountA) * m_hOutMarkedCountB;
  const double intersectionPercentage =
      totalPossiblePairs > 0 ? (static_cast<double>(totalIntersections) / totalPossiblePairs) * 100.0 : 0.0;

  // Stage 2: Dual-Tree Traversal Descent
  const double tDualStepStart = cuBQL::getCurrentTime();
  if(numberOfDualTreeSteps > 0) {
    executeDualTreeStep(numberOfDualTreeSteps, m_impl->m_levelA, m_impl->m_levelB, m_impl->m_dOutPairsA, m_impl->m_dOutPairsB,
                        m_dMarkedNodeIndicesA, m_dMarkedNodeIndicesB,
                        m_impl->m_dNodeDescendantCountsA, m_impl->m_dNodeDescendantCountsB,
                        m_hOutMarkedCountA, m_hOutMarkedCountB, m_impl->m_bvhA, m_impl->m_bvhB);

    CUBQL_CUDA_CALL(StreamSynchronize(m_impl->m_stream));
  }
  const double tDualStepEnd = cuBQL::getCurrentTime();
  stats.dualTreeStepMs = (numberOfDualTreeSteps > 0) ? (tDualStepEnd - tDualStepStart) * 1000.0 : 0.0;

  const int totalBatches = static_cast<int>(m_impl->m_dOutPairsA.size());
  IntersectionTimeTracker tracker;
  uint64_t finalCandidatePairs = 0;

  // Stage 3: Parallel BFS Rapid Descent & Target Reverse Mapping
  const double tGpuBfsStart = cuBQL::getCurrentTime();
  executeRapidDescentBFS(m_impl->m_bvhB, m_impl->m_numTrianglesB, m_dMarkedNodeIndicesB,
                         m_impl->m_dNodeDescendantCountsB, m_hOutMarkedCountB,
                         m_impl->m_dOutOffsetsB, m_impl->m_dOutPrimsFlatB);

  if(m_hOutMarkedCountB > 0) {
    constexpr int blockSize = 256;
    const int gridSize = (m_hOutMarkedCountB + blockSize - 1) / blockSize;
    populateReverseMapBKernel<<<gridSize, blockSize, 0, m_impl->m_stream>>>(
        thrust::raw_pointer_cast(m_impl->m_dReverseMapB.data()),
        thrust::raw_pointer_cast(m_dMarkedNodeIndicesB.data()),
        m_hOutMarkedCountB);
  }

  CUBQL_CUDA_CALL(StreamSynchronize(m_impl->m_stream));
  const double tGpuBfsEnd = cuBQL::getCurrentTime();

  // Stage 4: Batched Exact Predicate Traversal Pass
  if(m_impl->m_meshAcpu && m_impl->m_meshBcpu) {
    finalCandidatePairs = executeSingleTreeBatchedTraversalWithPredicates(
        *m_impl->m_meshAcpu, *m_impl->m_meshBcpu, batchMultiplier, totalBatches, m_impl->m_dOutPairsA, m_impl->m_dOutPairsB,
        m_impl->m_dReverseMapB, m_dMarkedNodeIndicesB, m_impl->m_dOutOffsetsB, m_impl->m_dOutPrimsFlatB,
        m_impl->m_dNodeDescendantCountsB, m_hOutMarkedCountB, m_impl->m_bvhA, m_impl->m_dBoxesA, m_impl->m_dBoxesB,
        m_impl->m_dVertsA, m_impl->m_dIndicesA, m_impl->m_dVertsB, m_impl->m_dIndicesB, outFinalExactPairs, outFinalCount,
        tracker, m_impl->m_centerA, m_impl->m_centerB, m_impl->m_rotA, m_impl->m_transA, m_impl->m_rotB, m_impl->m_transB, gpuDouble, m_impl->m_stream);
  }

  const double tPipelineEnd = cuBQL::getCurrentTime();

  // Stage 5: Metrics Record Keeping
  stats.meshATotalNodes = m_impl->m_bvhA.numNodes;
  stats.meshAExtractedTargets = m_hOutMarkedCountA;
  stats.meshBTotalNodes = m_impl->m_bvhB.numNodes;
  stats.meshBExtractedTargets = m_hOutMarkedCountB;
  stats.totalIntersections = totalIntersections;
  stats.totalPossiblePairs = totalPossiblePairs;
  stats.intersectionPercentage = intersectionPercentage;
  stats.gpuCrossCheckEngineMs = (tCrossEnd - tCrossStart) * 1000.0;
  stats.parallelDfsDescentBMs = (tGpuBfsEnd - tGpuBfsStart) * 1000.0;
  stats.GPUTotalTime = (tPipelineEnd - tPipelineStart) * 1000.0;
  stats.totalCrissCrossBatches = totalBatches;
  stats.finalAabbCandidatePairs = finalCandidatePairs;
  stats.loopTracker = tracker;
}

// ============================================================================
// Resource Cleanup & GPU Re-initialization
// ============================================================================

void KernelBVHController::cleanup() {
  clearGPU();

  m_impl->m_hVertsA = nullptr;
  m_impl->m_hIndicesA = nullptr;
  m_impl->m_hVertsB = nullptr;
  m_impl->m_hIndicesB = nullptr;

  m_impl->m_meshAcpu = nullptr;
  m_impl->m_meshBcpu = nullptr;

  m_impl->m_origPointsA.clear();
  m_impl->m_origPointsA.shrink_to_fit();
  m_impl->m_origPointsB.clear();
  m_impl->m_origPointsB.shrink_to_fit();

  m_impl->m_centerA = Point3(0, 0, 0);
  m_impl->m_centerB = Point3(0, 0, 0);
  m_impl->m_rotA = make_double3(0.0, 0.0, 0.0);
  m_impl->m_transA = make_double3(0.0, 0.0, 0.0);
  m_impl->m_rotB = make_double3(0.0, 0.0, 0.0);
  m_impl->m_transB = make_double3(0.0, 0.0, 0.0);
  m_impl->shiftX = m_impl->shiftY = m_impl->shiftZ = 0.0;
}

void KernelBVHController::clearGPU() {
  if(m_impl->m_stream) {
    CUBQL_CUDA_CALL(StreamSynchronize(m_impl->m_stream));
  }

  // Release Mesh A GPU buffers
  if(m_impl->m_dMeshA)        { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dMeshA, m_impl->m_stream)); m_impl->m_dMeshA = nullptr; }
  if(m_impl->m_dMeshMetricsA) { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dMeshMetricsA, m_impl->m_stream)); m_impl->m_dMeshMetricsA = nullptr; }
  if(m_impl->m_dBoxesA)       { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dBoxesA, m_impl->m_stream)); m_impl->m_dBoxesA = nullptr; }
  if(m_impl->m_dVertsA)       { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dVertsA, m_impl->m_stream)); m_impl->m_dVertsA = nullptr; }
  if(m_impl->m_dVertsAOrig)   { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dVertsAOrig, m_impl->m_stream)); m_impl->m_dVertsAOrig = nullptr; }
  if(m_impl->m_dIndicesA)     { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dIndicesA, m_impl->m_stream)); m_impl->m_dIndicesA = nullptr; }
  if(m_impl->m_dVertErrorsA)  { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dVertErrorsA, m_impl->m_stream)); m_impl->m_dVertErrorsA = nullptr; }

  // Release Mesh B GPU buffers
  if(m_impl->m_dMeshB)        { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dMeshB, m_impl->m_stream)); m_impl->m_dMeshB = nullptr; }
  if(m_impl->m_dMeshMetricsB) { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dMeshMetricsB, m_impl->m_stream)); m_impl->m_dMeshMetricsB = nullptr; }
  if(m_impl->m_dBoxesB)       { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dBoxesB, m_impl->m_stream)); m_impl->m_dBoxesB = nullptr; }
  if(m_impl->m_dVertsB)       { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dVertsB, m_impl->m_stream)); m_impl->m_dVertsB = nullptr; }
  if(m_impl->m_dVertsBOrig)   { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dVertsBOrig, m_impl->m_stream)); m_impl->m_dVertsBOrig = nullptr; }
  if(m_impl->m_dIndicesB)     { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dIndicesB, m_impl->m_stream)); m_impl->m_dIndicesB = nullptr; }
  if(m_impl->m_dVertErrorsB)  { CUBQL_CUDA_CALL(FreeAsync(m_impl->m_dVertErrorsB, m_impl->m_stream)); m_impl->m_dVertErrorsB = nullptr; }

  // Free cuBQL acceleration trees
  cuBQL::cuda::free(m_impl->m_bvhA, m_impl->m_stream, m_impl->m_memResource);
  cuBQL::cuda::free(m_impl->m_bvhB, m_impl->m_stream, m_impl->m_memResource);

  // Clear Thrust device vectors
  m_impl->m_dMarkedNodeIndicesA_Full.clear(); m_impl->m_dMarkedNodeIndicesA_Full.shrink_to_fit();
  m_impl->m_dNodeDescendantCountsA.clear();   m_impl->m_dNodeDescendantCountsA.shrink_to_fit();
  m_impl->m_dMarkedNodeIndicesB_Full.clear(); m_impl->m_dMarkedNodeIndicesB_Full.shrink_to_fit();
  m_impl->m_dNodeDescendantCountsB.clear();   m_impl->m_dNodeDescendantCountsB.shrink_to_fit();

  m_impl->m_dReverseMapB.clear();             m_impl->m_dReverseMapB.shrink_to_fit();
  m_impl->m_dOutPairsA.clear();               m_impl->m_dOutPairsA.shrink_to_fit();
  m_impl->m_dOutPairsB.clear();               m_impl->m_dOutPairsB.shrink_to_fit();
  m_impl->m_dOutOffsetsB.clear();             m_impl->m_dOutOffsetsB.shrink_to_fit();
  m_impl->m_dOutPrimsFlatB.clear();           m_impl->m_dOutPrimsFlatB.shrink_to_fit();
  m_impl->m_dOutOffsetsA.clear();             m_impl->m_dOutOffsetsA.shrink_to_fit();
  m_impl->m_dOutPrimsFlatA.clear();           m_impl->m_dOutPrimsFlatA.shrink_to_fit();

  if(m_impl->m_stream) {
    CUBQL_CUDA_CALL(StreamSynchronize(m_impl->m_stream));
  }
}

void KernelBVHController::reconstructGPU(ExecutionStats& stats, int levelA, int levelB) {
  m_impl->m_levelA = levelA;
  m_impl->m_levelB = levelB;
  this->reconstructGPU(stats);
}

void KernelBVHController::reconstructGPU(ExecutionStats& stats) {
  if(!m_impl->m_hVertsA || !m_impl->m_hIndicesA || !m_impl->m_hVertsB || !m_impl->m_hIndicesB ||
     m_impl->m_numTrianglesA <= 0 || m_impl->m_numTrianglesB <= 0) {
    std::cerr << "[KernelBVHController] Error: Cannot reconstruct GPU resources. Host pointers missing.\n";
    return;
  }

  clearGPU();

  cuBQL::BuildConfig buildConfig;
  buildConfig.makeLeafThreshold = m_impl->m_leafThreshold;

  const double tReconstructStart = cuBQL::getCurrentTime();

  // Re-allocate & copy Mesh A
  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dVertsA, m_impl->m_numVertsA * sizeof(double3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dVertsA, m_impl->m_hVertsA, m_impl->m_numVertsA * sizeof(double3), cudaMemcpyHostToDevice, m_impl->m_stream));

  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dVertsAOrig, m_impl->m_numVertsA * sizeof(double3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dVertsAOrig, m_impl->m_hVertsA, m_impl->m_numVertsA * sizeof(double3), cudaMemcpyHostToDevice, m_impl->m_stream));

  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dIndicesA, m_impl->m_numTrianglesA * sizeof(uint3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dIndicesA, m_impl->m_hIndicesA, m_impl->m_numTrianglesA * sizeof(uint3), cudaMemcpyHostToDevice, m_impl->m_stream));

  // Re-allocate & copy Mesh B
  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dVertsB, m_impl->m_numVertsB * sizeof(double3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dVertsB, m_impl->m_hVertsB, m_impl->m_numVertsB * sizeof(double3), cudaMemcpyHostToDevice, m_impl->m_stream));

  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dVertsBOrig, m_impl->m_numVertsB * sizeof(double3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dVertsBOrig, m_impl->m_hVertsB, m_impl->m_numVertsB * sizeof(double3), cudaMemcpyHostToDevice, m_impl->m_stream));

  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dIndicesB, m_impl->m_numTrianglesB * sizeof(uint3), m_impl->m_stream));
  CUBQL_CUDA_CALL(MemcpyAsync(m_impl->m_dIndicesB, m_impl->m_hIndicesB, m_impl->m_numTrianglesB * sizeof(uint3), cudaMemcpyHostToDevice, m_impl->m_stream));

  // Compute bounding boxes
  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dBoxesA, m_impl->m_numTrianglesA * sizeof(cuBQL::box3f), m_impl->m_stream));
  CUBQL_CUDA_CALL(MallocAsync((void**)&m_impl->m_dBoxesB, m_impl->m_numTrianglesB * sizeof(cuBQL::box3f), m_impl->m_stream));

  generateBoxesTrisKernel<<<cuBQL::divRoundUp(m_impl->m_numTrianglesA, 256), 256, 0, m_impl->m_stream>>>(
      m_impl->m_dBoxesA, m_impl->m_dVertsA, m_impl->m_dIndicesA, m_impl->m_numTrianglesA);

  generateBoxesTrisKernel<<<cuBQL::divRoundUp(m_impl->m_numTrianglesB, 256), 256, 0, m_impl->m_stream>>>(
      m_impl->m_dBoxesB, m_impl->m_dVertsB, m_impl->m_dIndicesB, m_impl->m_numTrianglesB);

  // Resize Thrust vectors
  const uint32_t maxNodesA = 2 * m_impl->m_numTrianglesA;
  const uint32_t maxNodesB = 2 * m_impl->m_numTrianglesB;

  m_impl->m_dMarkedNodeIndicesA_Full.resize(maxNodesA, 0);
  m_impl->m_dNodeDescendantCountsA.resize(maxNodesA, 0);
  m_impl->m_dMarkedNodeIndicesB_Full.resize(maxNodesB, 0);
  m_impl->m_dNodeDescendantCountsB.resize(maxNodesB, 0);
  m_impl->m_dReverseMapB.resize(maxNodesB, 0);

  // Rebuild BVH structures
  cuBQL::ext::level_cut::build_custom(m_impl->m_bvhA, m_impl->m_dBoxesA, m_impl->m_numTrianglesA, buildConfig, (uint32_t)m_impl->m_levelA,
                                       thrust::raw_pointer_cast(m_impl->m_dMarkedNodeIndicesA_Full.data()),
                                       thrust::raw_pointer_cast(m_impl->m_dNodeDescendantCountsA.data()),
                                       &m_impl->m_hOutMarkedCountA_Full, m_impl->m_stream, m_impl->m_memResource);
  cuBQL::cuda::refit(m_impl->m_bvhA, m_impl->m_dBoxesA, m_impl->m_stream);

  cuBQL::ext::level_cut::build_custom(m_impl->m_bvhB, m_impl->m_dBoxesB, m_impl->m_numTrianglesB, buildConfig, (uint32_t)m_impl->m_levelB,
                                       thrust::raw_pointer_cast(m_impl->m_dMarkedNodeIndicesB_Full.data()),
                                       thrust::raw_pointer_cast(m_impl->m_dNodeDescendantCountsB.data()),
                                       &m_impl->m_hOutMarkedCountB_Full, m_impl->m_stream, m_impl->m_memResource);
  cuBQL::cuda::refit(m_impl->m_bvhB, m_impl->m_dBoxesB, m_impl->m_stream);

  CUBQL_CUDA_CALL(StreamSynchronize(m_impl->m_stream));

  // Re-apply transformations if previously active
  if(m_impl->m_rotA.x != 0.0 || m_impl->m_rotA.y != 0.0 || m_impl->m_rotA.z != 0.0 ||
     m_impl->m_transA.x != 0.0 || m_impl->m_transA.y != 0.0 || m_impl->m_transA.z != 0.0 ||
     m_impl->m_rotB.x != 0.0 || m_impl->m_rotB.y != 0.0 || m_impl->m_rotB.z != 0.0 ||
     m_impl->m_transB.x != 0.0 || m_impl->m_transB.y != 0.0 || m_impl->m_transB.z != 0.0) {
    float tGPU, tCPU, tTV, tAT, tGB;
    const double3 currentRotA = m_impl->m_rotA, currentTransA = m_impl->m_transA;
    const double3 currentRotB = m_impl->m_rotB, currentTransB = m_impl->m_transB;

    m_impl->m_rotA = m_impl->m_transA = m_impl->m_rotB = m_impl->m_transB = make_double3(0, 0, 0);
    setTransformBoth(currentRotA, currentTransA, currentRotB, currentTransB, tGPU, tCPU, tTV, tAT, tGB);
  }

  stats.initialAllocAndCopyMs = (cuBQL::getCurrentTime() - tReconstructStart) * 1000.0;
}