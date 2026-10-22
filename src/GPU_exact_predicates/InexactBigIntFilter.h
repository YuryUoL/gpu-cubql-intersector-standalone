#ifndef GPU_PREDICATES_CHECK_BIGINTEGER_H
#define GPU_PREDICATES_CHECK_BIGINTEGER_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include "PairStatus.h"
#include "GPUInterval.h" // Include your interval definition

#ifdef __cplusplus
extern "C" {
#endif

void evaluateGeometricPairsKernelBigInt(
    const int2* dYellowCandidatePairs,
    int* dPairStatuses,
    const vec3IntervalDouble* dVertsA,
    const uint3* dIndicesA,
    const vec3IntervalDouble* dVertsB,
    const uint3* dIndicesB,
    int numYellowPairs,
    cudaStream_t stream = 0
);

#ifdef __cplusplus
}
#endif

#endif // GPU_PREDICATES_CHECK_BIGINTEGER_H