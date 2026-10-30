#ifndef GPU_PREDICATE_DOUBLE_H
#define GPU_PREDICATE_DOUBLE_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include "PairStatus.h"
#include "GPUInterval.h"

// Templated pipeline wrapper supporting any mix of double3 and vec3IntervalDouble
template <class VA, class VB>
void evaluateAndCompactPairsDoubleFull(
    const int2* dCandidatePairs,
    int* dPairStatuses,
    const VA* dVertsA,
    const uint3* dIndicesA,
    const VB* dVertsB,
    const uint3* dIndicesB,
    int totalBatchPairs,
    cudaStream_t stream = 0
);

#endif // GPU_PREDICATE_DOUBLE_H