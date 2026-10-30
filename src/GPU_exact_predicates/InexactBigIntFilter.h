#ifndef GPU_PREDICATES_CHECK_BIGINTEGER_H
#define GPU_PREDICATES_CHECK_BIGINTEGER_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include "PairStatus.h"
#include "GPUInterval.h"

// Templated inexact big-int filter pipeline supporting any mix of double3 and vec3IntervalDouble
template <class VA, class VB>
void evaluateGeometricPairsKernelBigInt(
    const int2* dYellowCandidatePairs,
    int* dPairStatuses,
    const VA* dVertsA,
    const uint3* dIndicesA,
    const VB* dVertsB,
    const uint3* dIndicesB,
    int numYellowPairs,
    cudaStream_t stream = 0
);

#endif // GPU_PREDICATES_CHECK_BIGINTEGER_H