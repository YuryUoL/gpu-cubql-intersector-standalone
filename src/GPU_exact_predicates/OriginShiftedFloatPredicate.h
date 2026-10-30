#ifndef GPU_PREDICATES_CHECK_DOUBLE_ASSISTED_H
#define GPU_PREDICATES_CHECK_DOUBLE_ASSISTED_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include "PairStatus.h"
#include "GPUInterval.h"

// VA / VB are each either double3 (exact points) or vec3IntervalDouble (boxes);
// mesh A and mesh B may use different types. Instantiated in the .cu for all 4 combinations.
template <class VA, class VB>
void evaluateAndCompactPairsDoubleAssisted(
    const int2* dCandidatePairs,
    int* dPairStatuses,
    const VA* dVertsA,
    const uint3* dIndicesA,
    const VB* dVertsB,
    const uint3* dIndicesB,
    int totalBatchPairs,
    cudaStream_t stream = 0
);

#endif // GPU_PREDICATES_CHECK_DOUBLE_ASSISTED_H
