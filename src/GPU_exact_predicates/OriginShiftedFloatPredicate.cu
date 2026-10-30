#include "OriginShiftedFloatPredicate.h"
#include "OriginShiftedFloatPredicateCore.cuh"

// --------------------------------------------------------------------
// CORE INTERSECTION KERNEL (templated on vertex type of each mesh)
// --------------------------------------------------------------------
template <class VA, class VB>
__global__ void evaluateGeometricPairsKernelDoubleAssisted(
    int* outStatuses,
    const int2* candidatePairs,
    const VA* dVertsA, const uint3* dIndicesA,
    const VB* dVertsB, const uint3* dIndicesB,
    int numPairs)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numPairs) return;

    const int2 pair = candidatePairs[tid];
    const uint3 idxA = dIndicesA[pair.x];
    const uint3 idxB = dIndicesB[pair.y];

    const VA Aa = dVertsA[idxA.x], Ab = dVertsA[idxA.y], Ac = dVertsA[idxA.z];
    const VB Ba = dVertsB[idxB.x], Bb = dVertsB[idxB.y], Bc = dVertsB[idxB.z];

    outStatuses[tid] = (int)pred::classifyPair(Aa, Ab, Ac, Ba, Bb, Bc);
}

// --------------------------------------------------------------------
// EXPORTED PIPELINE WRAPPER STAGE
// --------------------------------------------------------------------
template <class VA, class VB>
void evaluateAndCompactPairsDoubleAssisted(
    const int2* dCandidatePairs, int* dPairStatuses,
    const VA* dVertsA, const uint3* dIndicesA,
    const VB* dVertsB, const uint3* dIndicesB,
    int totalBatchPairs, cudaStream_t stream)
{
    if (totalBatchPairs <= 0) return;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (totalBatchPairs + threadsPerBlock - 1) / threadsPerBlock;

    evaluateGeometricPairsKernelDoubleAssisted<VA, VB><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        dPairStatuses, dCandidatePairs, dVertsA, dIndicesA, dVertsB, dIndicesB, totalBatchPairs);
}

// Explicit instantiations: exact points (double3) and/or boxes (vec3IntervalDouble)
#define INSTANTIATE_PAIR_EVAL(VA, VB)                                                   \
    template void evaluateAndCompactPairsDoubleAssisted<VA, VB>(                        \
        const int2*, int*, const VA*, const uint3*, const VB*, const uint3*, int, cudaStream_t);

INSTANTIATE_PAIR_EVAL(vec3IntervalDouble, vec3IntervalDouble)
INSTANTIATE_PAIR_EVAL(double3,            double3)
INSTANTIATE_PAIR_EVAL(double3,            vec3IntervalDouble)
INSTANTIATE_PAIR_EVAL(vec3IntervalDouble, double3)
