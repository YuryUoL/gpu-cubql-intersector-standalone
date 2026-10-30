#include "DoublePredicate.h"
#include "GPUPredicatesCommon.h"
#include "GPUInterval.h"
#include <cmath>

// --------------------------------------------------------------------
// RIGOROUS FP64 INTERVAL-TO-INTERVAL CREATION & ARITHMETIC
// --------------------------------------------------------------------

// Rigorously shifts an input vertex interval relative to an anchor interval using FP64 directed rounding
__device__ inline GPUInterval<double> make_double_interval_from_interval(
    GPUInterval<double> x_val, GPUInterval<double> anchor_val) 
{
    return GPUInterval<double>{
        __dsub_rd(x_val.lo, anchor_val.hi), // Force-rounds down toward -inf
        __dsub_ru(x_val.hi, anchor_val.lo)  // Force-rounds up toward +inf
    };
}

__device__ inline GPUInterval<double> dinterval_sub(GPUInterval<double> a, GPUInterval<double> b) {
    return GPUInterval<double>{
        __dsub_rd(a.lo, b.hi),
        __dsub_ru(a.hi, b.lo)
    };
}

__device__ inline GPUInterval<double> dinterval_add(GPUInterval<double> a, GPUInterval<double> b) {
    return GPUInterval<double>{
        __dadd_rd(a.lo, b.lo),
        __dadd_ru(a.hi, b.hi)
    };
}

__device__ inline GPUInterval<double> dinterval_mul(GPUInterval<double> a, GPUInterval<double> b) {
    double lo1 = __dmul_rd(a.lo, b.lo); double lo2 = __dmul_rd(a.lo, b.hi);
    double lo3 = __dmul_rd(a.hi, b.lo); double lo4 = __dmul_rd(a.hi, b.hi);
    double hi1 = __dmul_ru(a.lo, b.lo); double hi2 = __dmul_ru(a.lo, b.hi);
    double hi3 = __dmul_ru(a.hi, b.lo); double hi4 = __dmul_ru(a.hi, b.hi);
    return GPUInterval<double>{
        fmin(fmin(lo1, lo2), fmin(lo3, lo4)),
        fmax(fmax(hi1, hi2), fmax(hi3, hi4))
    };
}

// --------------------------------------------------------------------
// RIGOROUS FP64 INTERVAL ORIENT3D & EDGETRI PREDICATES
// --------------------------------------------------------------------
__device__ inline int orient3d_double_interval(
    const vec3Interval<double>& p, const vec3Interval<double>& q, 
    const vec3Interval<double>& r, const vec3Interval<double>& s) 
{
    GPUInterval<double> pdx = dinterval_sub(p.x, s.x); 
    GPUInterval<double> pdy = dinterval_sub(p.y, s.y); 
    GPUInterval<double> pdz = dinterval_sub(p.z, s.z);

    GPUInterval<double> qdx = dinterval_sub(q.x, s.x); 
    GPUInterval<double> qdy = dinterval_sub(q.y, s.y); 
    GPUInterval<double> qdz = dinterval_sub(q.z, s.z);

    GPUInterval<double> rdx = dinterval_sub(r.x, s.x); 
    GPUInterval<double> rdy = dinterval_sub(r.y, s.y); 
    GPUInterval<double> rdz = dinterval_sub(r.z, s.z);

    GPUInterval<double> det = dinterval_add(
        dinterval_add(
            dinterval_mul(pdx, dinterval_sub(dinterval_mul(qdy, rdz), dinterval_mul(qdz, rdy))),
            dinterval_mul(pdy, dinterval_sub(dinterval_mul(qdz, rdx), dinterval_mul(qdx, rdz)))
        ),
        dinterval_mul(pdz, dinterval_sub(dinterval_mul(qdx, rdy), dinterval_mul(qdy, rdx)))
    );

    // Pure mathematical sign check on rigorous intervals
    if (det.lo > 0.0) return 1;
    if (det.hi < 0.0) return -1;
    return 0; // Determinant straddles zero -> Exact CPU Fallback (PAIR_YELLOW)
}

__device__ inline int edgeTri_double_interval(
    const vec3Interval<double> &a, const vec3Interval<double> &b,
    const vec3Interval<double> &p, const vec3Interval<double> &q, const vec3Interval<double> &r) 
{
    int s0 = orient3d_double_interval(p, q, r, a);
    int s1 = orient3d_double_interval(p, q, r, b);

    if (s0 == 0 || s1 == 0) return PAIR_YELLOW;
    if ((s0 > 0 && s1 > 0) || (s0 < 0 && s1 < 0)) return PAIR_NO;

    int e0 = orient3d_double_interval(a, b, p, q);
    int e1 = orient3d_double_interval(a, b, q, r);
    int e2 = orient3d_double_interval(a, b, r, p);

    if (e0 == 0 || e1 == 0 || e2 == 0) return PAIR_YELLOW;
    if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) return PAIR_GREEN;

    return PAIR_NO;
}

// --------------------------------------------------------------------
// CLASSIFICATION FUNCTION (Hardware FP64 Interval Arithmetic)
// --------------------------------------------------------------------
__device__ inline PairStatus classifyPairDoubleFull(
    const vec3IntervalDouble& Aa, const vec3IntervalDouble& Ab, const vec3IntervalDouble& Ac,
    const vec3IntervalDouble& Ba, const vec3IntervalDouble& Bb, const vec3IntervalDouble& Bc) 
{    
    // 1. Pick Vertex Aa interval as Local Anchor Origin
    const vec3IntervalDouble anchor = Aa;
    
    // 2. Build Anchor Origin Interval (Zero-width point interval at origin) and relative intervals
    const vec3Interval<double> A0 = {
        GPUInterval<double>{0.0, 0.0}, GPUInterval<double>{0.0, 0.0}, GPUInterval<double>{0.0, 0.0}
    };
    const vec3Interval<double> A1 = {
        make_double_interval_from_interval(Ab.x, anchor.x),
        make_double_interval_from_interval(Ab.y, anchor.y),
        make_double_interval_from_interval(Ab.z, anchor.z)
    };
    const vec3Interval<double> A2 = {
        make_double_interval_from_interval(Ac.x, anchor.x),
        make_double_interval_from_interval(Ac.y, anchor.y),
        make_double_interval_from_interval(Ac.z, anchor.z)
    };

    const vec3Interval<double> B0 = {
        make_double_interval_from_interval(Ba.x, anchor.x),
        make_double_interval_from_interval(Ba.y, anchor.y),
        make_double_interval_from_interval(Ba.z, anchor.z)
    };
    const vec3Interval<double> B1 = {
        make_double_interval_from_interval(Bb.x, anchor.x),
        make_double_interval_from_interval(Bb.y, anchor.y),
        make_double_interval_from_interval(Bb.z, anchor.z)
    };
    const vec3Interval<double> B2 = {
        make_double_interval_from_interval(Bc.x, anchor.x),
        make_double_interval_from_interval(Bc.y, anchor.y),
        make_double_interval_from_interval(Bc.z, anchor.z)
    };

    // 3. Early Rejection Checks using Interval Orientations
    int ob0 = orient3d_double_interval(A0, A1, A2, B0);
    int ob1 = orient3d_double_interval(A0, A1, A2, B1);
    int ob2 = orient3d_double_interval(A0, A1, A2, B2);
    if (ob0 != 0 && ob1 != 0 && ob2 != 0) {
        if ((ob0 > 0 && ob1 > 0 && ob2 > 0) || (ob0 < 0 && ob1 < 0 && ob2 < 0)) return PAIR_NO;
    }

    int oa0 = orient3d_double_interval(B0, B1, B2, A0);
    int oa1 = orient3d_double_interval(B0, B1, B2, A1);
    int oa2 = orient3d_double_interval(B0, B1, B2, A2);
    if (oa0 != 0 && oa1 != 0 && oa2 != 0) {
        if ((oa0 > 0 && oa1 > 0 && oa2 > 0) || (oa0 < 0 && oa1 < 0 && oa2 < 0)) return PAIR_NO;
    }

    // 4. Detailed Edge-Triangle Interval Predicates
    int r;
    r = edgeTri_double_interval(A0, A1, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTri_double_interval(A1, A2, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTri_double_interval(A2, A0, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;

    r = edgeTri_double_interval(B0, B1, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTri_double_interval(B1, B2, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTri_double_interval(B2, B0, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;

    return PAIR_NO;
}

// --------------------------------------------------------------------
// CORE INTERSECTION KERNEL & WRAPPER
// --------------------------------------------------------------------
__global__ void evaluateGeometricPairsKernelDoubleFull(
    int* dPairStatuses, 
    const int2* dYellowCandidatePairs, 
    const vec3IntervalDouble* dVertsA,
    const uint3* dIndicesA,
    const vec3IntervalDouble* dVertsB,
    const uint3* dIndicesB,
    int numYellowPairs) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numYellowPairs) return;

    int2 pair = dYellowCandidatePairs[tid];
    
    uint3 idxA = dIndicesA[pair.x];
    uint3 idxB = dIndicesB[pair.y];

    vec3IntervalDouble Aa = dVertsA[idxA.x];
    vec3IntervalDouble Ab = dVertsA[idxA.y];
    vec3IntervalDouble Ac = dVertsA[idxA.z];

    vec3IntervalDouble Ba = dVertsB[idxB.x];
    vec3IntervalDouble Bb = dVertsB[idxB.y];
    vec3IntervalDouble Bc = dVertsB[idxB.z];

    PairStatus status = classifyPairDoubleFull(Aa, Ab, Ac, Ba, Bb, Bc);

    dPairStatuses[tid] = (int)status; 
}

void evaluateAndCompactPairsDoubleFull(
    const int2* dYellowCandidatePairs,
    int* dPairStatuses,
    const vec3IntervalDouble* dVertsA,
    const uint3* dIndicesA,
    const vec3IntervalDouble* dVertsB,
    const uint3* dIndicesB,
    int numYellowPairs,
    cudaStream_t stream)
{
    if (numYellowPairs <= 0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (numYellowPairs + threadsPerBlock - 1) / threadsPerBlock;
    
    evaluateGeometricPairsKernelDoubleFull<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
         dPairStatuses, dYellowCandidatePairs, 
         dVertsA, dIndicesA, 
         dVertsB, dIndicesB, 
         numYellowPairs
    );
}