#include "OriginShiftedFloatPredicate.h"
#include "GPUPredicatesCommon.h"

// --------------------------------------------------------------------
// ANCHOR-SHIFTED + RIGOROUS INTERVAL CLASSIFICATION FUNCTION
// --------------------------------------------------------------------
__device__ inline PairStatus classifyPair(
    const double3& Aa, const double3& Ab, const double3& Ac,
    const double3& Ba, const double3& Bb, const double3& Bc) 
{    
    // 1. Pick Vertex Aa as Local Anchor Origin
    const double3 anchor = Aa;

    // 2. Build Anchor Origin Interval (Zero-width point interval at origin)
    const vec3Interval A0 = {
        GPUInterval{0.0f, 0.0f},
        GPUInterval{0.0f, 0.0f},
        GPUInterval{0.0f, 0.0f}
    };

    // 3. Perform FP64 Shift and Rigorous FP32 Interval Conversion
    vec3Interval A1 = {
        make_provable_relative_interval(Ab.x, anchor.x),
        make_provable_relative_interval(Ab.y, anchor.y),
        make_provable_relative_interval(Ab.z, anchor.z)
    };
    vec3Interval A2 = {
        make_provable_relative_interval(Ac.x, anchor.x),
        make_provable_relative_interval(Ac.y, anchor.y),
        make_provable_relative_interval(Ac.z, anchor.z)
    };

    vec3Interval B0 = {
        make_provable_relative_interval(Ba.x, anchor.x),
        make_provable_relative_interval(Ba.y, anchor.y),
        make_provable_relative_interval(Ba.z, anchor.z)
    };
    vec3Interval B1 = {
        make_provable_relative_interval(Bb.x, anchor.x),
        make_provable_relative_interval(Bb.y, anchor.y),
        make_provable_relative_interval(Bb.z, anchor.z)
    };
    vec3Interval B2 = {
        make_provable_relative_interval(Bc.x, anchor.x),
        make_provable_relative_interval(Bc.y, anchor.y),
        make_provable_relative_interval(Bc.z, anchor.z)
    };

    // 4. Compute Local Maximum Coordinate Bound (Optional safety check for degeneracies)
    float mx1 = fmaxf(fabsf(A1.x.hi), fabsf(A1.y.hi));
    float mx2 = fmaxf(fabsf(A1.z.hi), fabsf(A2.x.hi));
    float mx3 = fmaxf(fabsf(A2.y.hi), fabsf(A2.z.hi));
    float mx4 = fmaxf(fabsf(B0.x.hi), fabsf(B0.y.hi));
    float mx5 = fmaxf(fabsf(B0.z.hi), fabsf(B1.x.hi));
    float mx6 = fmaxf(fabsf(B1.y.hi), fabsf(B1.z.hi));
    float mx7 = fmaxf(fabsf(B2.x.hi), fabsf(B2.y.hi));
    float mx8 = fabsf(B2.z.hi);

    float m12 = fmaxf(mx1, mx2);
    float m34 = fmaxf(mx3, mx4);
    float m56 = fmaxf(mx5, mx6);
    float m78 = fmaxf(mx7, mx8);

    float M_local = fmaxf(fmaxf(m12, m34), fmaxf(m56, m78));

    // 5. Early exit for extreme degeneracies
    if (M_local <= 1e-12f) {
        return PAIR_YELLOW;
    }

    // (Normalization step removed completely — zero extra rounding error introduced!)

    // 6. Geometrical Predicate Evaluation Passes (Using Raw Shifted Intervals)
    int ob0 = orient3d_interval(A0, A1, A2, B0);
    int ob1 = orient3d_interval(A0, A1, A2, B1);
    int ob2 = orient3d_interval(A0, A1, A2, B2);
    if (ob0 != 0 && ob1 != 0 && ob2 != 0) {
        if ((ob0 > 0 && ob1 > 0 && ob2 > 0) || (ob0 < 0 && ob1 < 0 && ob2 < 0)) return PAIR_NO;
    }

    int oa0 = orient3d_interval(B0, B1, B2, A0);
    int oa1 = orient3d_interval(B0, B1, B2, A1);
    int oa2 = orient3d_interval(B0, B1, B2, A2);
    if (oa0 != 0 && oa1 != 0 && oa2 != 0) {
        if ((oa0 > 0 && oa1 > 0 && oa2 > 0) || (oa0 < 0 && oa1 < 0 && oa2 < 0)) return PAIR_NO;
    }

    int r;
    r = edgeTriInterval(A0, A1, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTriInterval(A1, A2, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTriInterval(A2, A0, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;

    r = edgeTriInterval(B0, B1, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTriInterval(B1, B2, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTriInterval(B2, B0, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;

    return PAIR_NO;
}

// --------------------------------------------------------------------
// CORE INTERSECTION KERNEL
// --------------------------------------------------------------------
__global__ void evaluateGeometricPairsKernelDoubleAssisted(
    int* outStatuses, 
    const int2* candidatePairs, 
    const double3* dVertsA,
    const uint3* dIndicesA,
    const double3* dVertsB,
    const uint3* dIndicesB,
    int numPairs) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numPairs) return;

    int2 pair = candidatePairs[tid];
    
    uint3 idxA = dIndicesA[pair.x];
    uint3 idxB = dIndicesB[pair.y];

    double3 Aa = { __ldg(&dVertsA[idxA.x].x), __ldg(&dVertsA[idxA.x].y), __ldg(&dVertsA[idxA.x].z) };
    double3 Ab = { __ldg(&dVertsA[idxA.y].x), __ldg(&dVertsA[idxA.y].y), __ldg(&dVertsA[idxA.y].z) };
    double3 Ac = { __ldg(&dVertsA[idxA.z].x), __ldg(&dVertsA[idxA.z].y), __ldg(&dVertsA[idxA.z].z) };

    double3 Ba = { __ldg(&dVertsB[idxB.x].x), __ldg(&dVertsB[idxB.x].y), __ldg(&dVertsB[idxB.x].z) };
    double3 Bb = { __ldg(&dVertsB[idxB.y].x), __ldg(&dVertsB[idxB.y].y), __ldg(&dVertsB[idxB.y].z) };
    double3 Bc = { __ldg(&dVertsB[idxB.z].x), __ldg(&dVertsB[idxB.z].y), __ldg(&dVertsB[idxB.z].z) };

    PairStatus status = classifyPair(Aa, Ab, Ac, Ba, Bb, Bc);

    outStatuses[tid] = (int)status;
}

// --------------------------------------------------------------------
// EXPORTED PIPELINE WRAPPER STAGE
// --------------------------------------------------------------------
void evaluateAndCompactPairsDoubleAssisted(
    const int2* dCandidatePairs,
    int* dPairStatuses,
    const double3* dVertsA,
    const uint3* dIndicesA,
    const double3* dVertsB,
    const uint3* dIndicesB,
    int totalBatchPairs,
    cudaStream_t stream)
{
    if (totalBatchPairs <= 0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalBatchPairs + threadsPerBlock - 1) / threadsPerBlock;
    
    evaluateGeometricPairsKernelDoubleAssisted<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
         dPairStatuses, dCandidatePairs, 
         dVertsA, dIndicesA, 
         dVertsB, dIndicesB, 
         totalBatchPairs
    );
}