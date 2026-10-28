#include "OriginShiftedFloatPredicate.h"
#include "GPUPredicatesCommon.h"
#include "GPUInterval.h"

// --------------------------------------------------------------------
// RIGOROUS INTERVAL-TO-INTERVAL SHIFT & FP32 CONVERSION
// --------------------------------------------------------------------
__device__ inline GPUInterval<float> make_provable_interval_from_interval(
    GPUInterval<double> x_val, GPUInterval<double> anchor_val) 
{
    // 1. Rigorous interval subtraction in FP64 with directed rounding: [lo - hi, hi - lo]
    double rel_lo = __dsub_rd(x_val.lo, anchor_val.hi);
    double rel_hi = __dsub_ru(x_val.hi, anchor_val.lo);

    // 2. Rigorous conversion from FP64 to FP32 with outward directed rounding
    float f_lo = __double2float_rd(rel_lo);
    float f_hi = __double2float_ru(rel_hi);

    // 3. Combined relative error bound (FP64 subtraction + FP32 cast truncation)
    //constexpr float C_provable = 5.9604650e-8f; 
    constexpr float C_provable = 1.1920929e-7f;
    float err_lo = fabsf(f_lo) * C_provable;
    float err_hi = fabsf(f_hi) * C_provable;

    // Subnormal / underflow floor safeguard
    constexpr float min_normal_float = 1.17549435e-38f;
    err_lo = fmaxf(err_lo, min_normal_float * 2.0f);
    err_hi = fmaxf(err_hi, min_normal_float * 2.0f);

    return GPUInterval<float>{
        __fsub_rd(f_lo, err_lo),
        __fadd_ru(f_hi, err_hi)
    };
}

// --------------------------------------------------------------------
// ANCHOR-SHIFTED + RIGOROUS INTERVAL CLASSIFICATION FUNCTION
// --------------------------------------------------------------------
__device__ inline PairStatus classifyPair(
    const vec3IntervalDouble& Aa, const vec3IntervalDouble& Ab, const vec3IntervalDouble& Ac,
    const vec3IntervalDouble& Ba, const vec3IntervalDouble& Bb, const vec3IntervalDouble& Bc) 
{    
   // return PAIR_YELLOW;
    // 1. Pick Vertex Aa interval as Local Anchor Origin
    const vec3IntervalDouble anchor = Aa;

    // 2. Build Anchor Origin Interval (Zero-width point interval at origin)
    const vec3Interval<float> A0 = {
        GPUInterval<float>{0.0f, 0.0f},
        GPUInterval<float>{0.0f, 0.0f},
        GPUInterval<float>{0.0f, 0.0f}
    };

    // 3. Perform Rigorous Interval-to-Interval Shift and FP32 Interval Conversion
    vec3Interval<float> A1 = {
        make_provable_interval_from_interval(Ab.x, anchor.x),
        make_provable_interval_from_interval(Ab.y, anchor.y),
        make_provable_interval_from_interval(Ab.z, anchor.z)
    };
    vec3Interval<float> A2 = {
        make_provable_interval_from_interval(Ac.x, anchor.x),
        make_provable_interval_from_interval(Ac.y, anchor.y),
        make_provable_interval_from_interval(Ac.z, anchor.z)
    };

    vec3Interval<float> B0 = {
        make_provable_interval_from_interval(Ba.x, anchor.x),
        make_provable_interval_from_interval(Ba.y, anchor.y),
        make_provable_interval_from_interval(Ba.z, anchor.z)
    };
    vec3Interval<float> B1 = {
        make_provable_interval_from_interval(Bb.x, anchor.x),
        make_provable_interval_from_interval(Bb.y, anchor.y),
        make_provable_interval_from_interval(Bb.z, anchor.z)
    };
    vec3Interval<float> B2 = {
        make_provable_interval_from_interval(Bc.x, anchor.x),
        make_provable_interval_from_interval(Bc.y, anchor.y),
        make_provable_interval_from_interval(Bc.z, anchor.z)
    };

    // 4. Compute Local Maximum Coordinate Bound (Safety check for degeneracies)
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
    const vec3IntervalDouble* dVertsA,
    const uint3* dIndicesA,
    const vec3IntervalDouble* dVertsB,
    const uint3* dIndicesB,
    int numPairs) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numPairs) return;

    int2 pair = candidatePairs[tid];
    
    uint3 idxA = dIndicesA[pair.x];
    uint3 idxB = dIndicesB[pair.y];

    vec3IntervalDouble Aa = dVertsA[idxA.x];
    vec3IntervalDouble Ab = dVertsA[idxA.y];
    vec3IntervalDouble Ac = dVertsA[idxA.z];

    vec3IntervalDouble Ba = dVertsB[idxB.x];
    vec3IntervalDouble Bb = dVertsB[idxB.y];
    vec3IntervalDouble Bc = dVertsB[idxB.z];

    PairStatus status = classifyPair(Aa, Ab, Ac, Ba, Bb, Bc);

    outStatuses[tid] = (int)status;
}

// --------------------------------------------------------------------
// EXPORTED PIPELINE WRAPPER STAGE
// --------------------------------------------------------------------
void evaluateAndCompactPairsDoubleAssisted(
    const int2* dCandidatePairs,
    int* dPairStatuses,
    const vec3IntervalDouble* dVertsA,
    const uint3* dIndicesA,
    const vec3IntervalDouble* dVertsB,
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