#ifndef GPU_PREDICATE_DOUBLE_H
#define GPU_PREDICATE_DOUBLE_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include "PairStatus.h"
#include "GPUInterval.h"

// Executes a hardware FP64 double-precision interval evaluation on the candidate list
// accepting interval-valued vertex inputs.
void evaluateAndCompactPairsDoubleFull(
    const int2* dYellowCandidatePairs,
    int* dPairStatuses,
    const vec3IntervalDouble* dVertsA,
    const uint3* dIndicesA,
    const vec3IntervalDouble* dVertsB,
    const uint3* dIndicesB,
    int numYellowPairs,
    cudaStream_t stream = 0
);

#endif // GPU_PREDICATE_DOUBLE_H