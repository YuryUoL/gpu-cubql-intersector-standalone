#ifndef GPU_PREDICATES_CHECK_DOUBLE_H
#define GPU_PREDICATES_CHECK_DOUBLE_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include "PairStatus.h"
#include "GPUInterval.h"

// Executes a double-single (two-float) precision evaluation on the candidate list
// accepting interval-valued vertex inputs.
void evaluateAndCompactPairsDouble(
    const int2* dYellowCandidatePairs,
    int* dPairStatuses,
    const vec3IntervalDouble* dVertsA,
    const uint3* dIndicesA,
    const vec3IntervalDouble* dVertsB,
    const uint3* dIndicesB,
    int numYellowPairs,
    cudaStream_t stream = 0
);

#endif // GPU_PREDICATES_CHECK_DOUBLE_H