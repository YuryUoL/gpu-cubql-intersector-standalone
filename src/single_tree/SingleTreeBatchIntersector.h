#ifndef BATCHED_CROSS_INTERSECTION_DOUBLE_H
#define BATCHED_CROSS_INTERSECTION_DOUBLE_H

#include <iostream>
#include <vector>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <tbb/concurrent_vector.h>
#include <cuBQL/bvh.h>
#include <loadOBJ.h>

#include "common/IntersectionTimeTracker.h"
#include "GPU_exact_predicates/TwoSumDoublePredicate.h"
#include "CPU/CgalDefinitions.h"
#include "GPU_exact_predicates/GPUInterval.h"

// Unified template launcher supporting any combination of vertex types VA and VB (e.g., double3 or vec3IntervalDouble)
template <class VA, class VB>
uint64_t executeSingleTreeBatchedTraversalWithPredicates(
    Mesh & meshAcpu, Mesh & meshBcpu,
    int batchMultiplier,
    int totalBatches,
    const thrust::device_vector<uint32_t>& d_outPairsA,
    const thrust::device_vector<uint32_t>& d_outPairsB,
    const thrust::device_vector<uint32_t>& d_reverseMapB,
    const thrust::device_vector<uint32_t>& d_markedNodeIndicesB, 
    const thrust::device_vector<uint32_t>& d_outOffsetsB,
    const thrust::device_vector<uint32_t>& d_outPrimsFlatB,
    const thrust::device_vector<uint32_t>& d_nodeDescendantCountsB, 
    uint32_t h_outMarkedCountB,
    const cuBQL::bvh3f& bvhA,
    const cuBQL::box3f* d_boxesA,
    const cuBQL::box3f* d_boxesB,
    const VA* d_vertsA,
    const uint3* d_indicesA,
    const VB* d_vertsB,
    const uint3* d_indicesB,
    int2*& outFinalExactPairs,       
    size_t& outFinalCount, 
    IntersectionTimeTracker& tracker, Point3 m_centerA, Point3 m_centerB, double3 m_rotA, double3 m_transA, double3 m_rotB, double3 m_transB,
    int exactPredicateComputeMode = 1,
    cudaStream_t stream = 0
);

#endif // BATCHED_CROSS_INTERSECTION_DOUBLE_H