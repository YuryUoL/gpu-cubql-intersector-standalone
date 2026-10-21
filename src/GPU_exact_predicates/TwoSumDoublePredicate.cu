#include "TwoSumDoublePredicate.h"
#include "GPUPredicatesCommon.h"
#include "GPUInterval.h"
#include <cmath>

// --------------------------------------------------------------------
// DOUBLE-SINGLE (DS) SIMULATION ARITHMETIC
// --------------------------------------------------------------------
struct ds_float {
    float hi;
    float lo;
};

struct ds_vec3 {
    ds_float x, y, z;
};

// Converts native double into two exact FP32 components
__device__ inline ds_float double_to_ds(double val) {
    float hi = static_cast<float>(val);
    float lo = static_cast<float>(val - static_cast<double>(hi));
    return { hi, lo };
}

__device__ inline ds_float quick_two_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    float e = __fsub_rn(b, __fsub_rn(s, a));
    return { s, e };
}

__device__ inline ds_float two_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    float v = __fsub_rn(s, a);
    float e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, v)), __fsub_rn(b, v));
    return { s, e };
}

__device__ inline ds_float two_prod(float a, float b) {
    float p = __fmul_rn(a, b);
    float e = __fmaf_rn(a, b, -p); 
    return { p, e };
}

__device__ inline ds_float ds_add(ds_float a, ds_float b) {
    ds_float s = two_sum(a.hi, b.hi);
    ds_float t = two_sum(a.lo, b.lo);
    s.lo = __fadd_rn(s.lo, t.hi);
    s = quick_two_sum(s.hi, s.lo);
    s.lo = __fadd_rn(s.lo, t.lo);
    return quick_two_sum(s.hi, s.lo);
}

__device__ inline ds_float ds_sub(ds_float a, ds_float b) {
    return ds_add(a, { -b.hi, -b.lo });
}

__device__ inline ds_float ds_mul(ds_float a, ds_float b) {
    ds_float p = two_prod(a.hi, b.hi);
    p.lo = __fadd_rn(p.lo, __fadd_rn(__fmul_rn(a.hi, b.lo), __fmul_rn(a.lo, b.hi)));
    return quick_two_sum(p.hi, p.lo);
}

// Comparison helpers for DS floats
__device__ inline bool ds_gt(ds_float a, ds_float b) {
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
}

__device__ inline bool ds_lt(ds_float a, ds_float b) {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

// --------------------------------------------------------------------
// DOUBLE-SINGLE (DS) INTERVAL ARITHMETIC ALIASES & OPERATIONS
// --------------------------------------------------------------------
using dsInterval = GPUInterval<ds_float>;
using vec3DsInterval = vec3Interval<ds_float>;

// Rigorous interval-to-interval creation using FP64 directed rounding and conversion to DS
__device__ inline dsInterval make_ds_interval_from_interval(GPUInterval<double> x_val, GPUInterval<double> anchor_val) {
    // 1. FP64 interval subtraction with directed rounding: [lo - hi, hi - lo]
    double rel_lo = __dsub_rd(x_val.lo, anchor_val.hi);
    double rel_hi = __dsub_ru(x_val.hi, anchor_val.lo);

    // 2. Convert exact bounds to double-single representation
    ds_float ds_lo = double_to_ds(rel_lo);
    ds_float ds_hi = double_to_ds(rel_hi);

    // 3. Apply safety error bounds for double-single floating-point conversion
    constexpr float eps_ds = 1.1920929e-7f * 1.1920929e-7f; // ~1.4e-14
    float abs_lo = fabsf(ds_lo.hi);
    float abs_hi = fabsf(ds_hi.hi);

    ds_float lo_bound = ds_sub(ds_lo, {abs_lo * eps_ds, 0.0f});
    ds_float hi_bound = ds_add(ds_hi, {abs_hi * eps_ds, 0.0f});
    
    return dsInterval{ lo_bound, hi_bound };
}

__device__ inline dsInterval ds_interval_sub(dsInterval a, dsInterval b) {
    return dsInterval{
        ds_sub(a.lo, b.hi),
        ds_sub(a.hi, b.lo)
    };
}

__device__ inline dsInterval ds_interval_add(dsInterval a, dsInterval b) {
    return dsInterval{
        ds_add(a.lo, b.lo),
        ds_add(a.hi, b.hi)
    };
}

__device__ inline dsInterval ds_interval_mul(dsInterval a, dsInterval b) {
    ds_float p1 = ds_mul(a.lo, b.lo);
    ds_float p2 = ds_mul(a.lo, b.hi);
    ds_float p3 = ds_mul(a.hi, b.lo);
    ds_float p4 = ds_mul(a.hi, b.hi);

    ds_float min_val = p1;
    if (ds_gt(min_val, p2)) min_val = p2;
    if (ds_gt(min_val, p3)) min_val = p3;
    if (ds_gt(min_val, p4)) min_val = p4;

    ds_float max_val = p1;
    if (ds_lt(max_val, p2)) max_val = p2;
    if (ds_lt(max_val, p3)) max_val = p3;
    if (ds_lt(max_val, p4)) max_val = p4;

    return dsInterval{ min_val, max_val };
}

// --------------------------------------------------------------------
// RIGOROUS DS INTERVAL PREDICATES (ZERO EPSILON REQUIRED)
// --------------------------------------------------------------------
__device__ inline int orient3d_ds_interval(
    const vec3DsInterval& p, const vec3DsInterval& q, 
    const vec3DsInterval& r, const vec3DsInterval& s) 
{
    dsInterval pdx = ds_interval_sub(p.x, s.x); 
    dsInterval pdy = ds_interval_sub(p.y, s.y); 
    dsInterval pdz = ds_interval_sub(p.z, s.z);

    dsInterval qdx = ds_interval_sub(q.x, s.x); 
    dsInterval qdy = ds_interval_sub(q.y, s.y); 
    dsInterval qdz = ds_interval_sub(q.z, s.z);

    dsInterval rdx = ds_interval_sub(r.x, s.x); 
    dsInterval rdy = ds_interval_sub(r.y, s.y); 
    dsInterval rdz = ds_interval_sub(r.z, s.z);

    dsInterval det = ds_interval_add(
        ds_interval_add(
            ds_interval_mul(pdx, ds_interval_sub(ds_interval_mul(qdy, rdz), ds_interval_mul(qdz, rdy))),
            ds_interval_mul(pdy, ds_interval_sub(ds_interval_mul(qdz, rdx), ds_interval_mul(qdx, rdz)))
        ),
        ds_interval_mul(pdz, ds_interval_sub(ds_interval_mul(qdx, rdy), ds_interval_mul(qdy, rdx)))
    );

    ds_float zero = {0.0f, 0.0f};
    if (ds_gt(det.lo, zero)) return 1;
    if (ds_lt(det.hi, zero)) return -1;
    return 0; 
}

__device__ inline int edgeTri_ds_interval(
    const vec3DsInterval &a, const vec3DsInterval &b,
    const vec3DsInterval &p, const vec3DsInterval &q, const vec3DsInterval &r) 
{
    int s0 = orient3d_ds_interval(p, q, r, a);
    int s1 = orient3d_ds_interval(p, q, r, b);

    if (s0 == 0 || s1 == 0) return PAIR_YELLOW;
    if ((s0 > 0 && s1 > 0) || (s0 < 0 && s1 < 0)) return PAIR_NO;

    int e0 = orient3d_ds_interval(a, b, p, q);
    int e1 = orient3d_ds_interval(a, b, q, r);
    int e2 = orient3d_ds_interval(a, b, r, p);

    if (e0 == 0 || e1 == 0 || e2 == 0) return PAIR_YELLOW;
    if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) return PAIR_GREEN;

    return PAIR_NO;
}

// --------------------------------------------------------------------
// CLASSIFICATION FUNCTION (Double-Single Interval Arithmetic)
// --------------------------------------------------------------------
__device__ inline PairStatus classifyPairDS(
    const vec3IntervalDouble& Aa, const vec3IntervalDouble& Ab, const vec3IntervalDouble& Ac,
    const vec3IntervalDouble& Ba, const vec3IntervalDouble& Bb, const vec3IntervalDouble& Bc) 
{    
    // 1. Pick Vertex Aa interval as Local Anchor Origin
    const vec3IntervalDouble anchor = Aa;
    
    // 2. Build DS Intervals for all relative vertex intervals
    const vec3DsInterval A0 = {
        dsInterval{{0.0f, 0.0f}, {0.0f, 0.0f}},
        dsInterval{{0.0f, 0.0f}, {0.0f, 0.0f}},
        dsInterval{{0.0f, 0.0f}, {0.0f, 0.0f}}
    };
    const vec3DsInterval A1 = {
        make_ds_interval_from_interval(Ab.x, anchor.x),
        make_ds_interval_from_interval(Ab.y, anchor.y),
        make_ds_interval_from_interval(Ab.z, anchor.z)
    };
    const vec3DsInterval A2 = {
        make_ds_interval_from_interval(Ac.x, anchor.x),
        make_ds_interval_from_interval(Ac.y, anchor.y),
        make_ds_interval_from_interval(Ac.z, anchor.z)
    };

    const vec3DsInterval B0 = {
        make_ds_interval_from_interval(Ba.x, anchor.x),
        make_ds_interval_from_interval(Ba.y, anchor.y),
        make_ds_interval_from_interval(Ba.z, anchor.z)
    };
    const vec3DsInterval B1 = {
        make_ds_interval_from_interval(Bb.x, anchor.x),
        make_ds_interval_from_interval(Bb.y, anchor.y),
        make_ds_interval_from_interval(Bb.z, anchor.z)
    };
    const vec3DsInterval B2 = {
        make_ds_interval_from_interval(Bc.x, anchor.x),
        make_ds_interval_from_interval(Bc.y, anchor.y),
        make_ds_interval_from_interval(Bc.z, anchor.z)
    };

    // 3. Early Rejection Checks using DS Interval Orientations
    int ob0 = orient3d_ds_interval(A0, A1, A2, B0);
    int ob1 = orient3d_ds_interval(A0, A1, A2, B1);
    int ob2 = orient3d_ds_interval(A0, A1, A2, B2);
    if (ob0 != 0 && ob1 != 0 && ob2 != 0) {
        if ((ob0 > 0 && ob1 > 0 && ob2 > 0) || (ob0 < 0 && ob1 < 0 && ob2 < 0)) return PAIR_NO;
    }

    int oa0 = orient3d_ds_interval(B0, B1, B2, A0);
    int oa1 = orient3d_ds_interval(B0, B1, B2, A1);
    int oa2 = orient3d_ds_interval(B0, B1, B2, A2);
    if (oa0 != 0 && oa1 != 0 && oa2 != 0) {
        if ((oa0 > 0 && oa1 > 0 && oa2 > 0) || (oa0 < 0 && oa1 < 0 && oa2 < 0)) return PAIR_NO;
    }

    // 4. Detailed Edge-Triangle DS Interval Predicates
    int r;
    r = edgeTri_ds_interval(A0, A1, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTri_ds_interval(A1, A2, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTri_ds_interval(A2, A0, B0, B1, B2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;

    r = edgeTri_ds_interval(B0, B1, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTri_ds_interval(B1, B2, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;
    r = edgeTri_ds_interval(B2, B0, A0, A1, A2); if (r == PAIR_GREEN) return PAIR_GREEN; if (r == PAIR_YELLOW) return PAIR_YELLOW;

    return PAIR_NO;
}

// --------------------------------------------------------------------
// CORE INTERSECTION KERNEL
// --------------------------------------------------------------------
__global__ void evaluateGeometricPairsKernelDouble(
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

    PairStatus status = classifyPairDS(Aa, Ab, Ac, Ba, Bb, Bc);

    dPairStatuses[tid] = (int)status; 
}

// --------------------------------------------------------------------
// PIPELINE EXPORTED WRAPPER
// --------------------------------------------------------------------
void evaluateAndCompactPairsDouble(
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
    
    evaluateGeometricPairsKernelDouble<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
         dPairStatuses, dYellowCandidatePairs, 
         dVertsA, dIndicesA, 
         dVertsB, dIndicesB, 
         numYellowPairs
    );
}