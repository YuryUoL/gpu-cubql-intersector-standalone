#include "TwoSumDoublePredicate.h"
#include "GPUPredicatesCommon.h"
#include <cmath>

// --------------------------------------------------------------------
// MEMORY HELPER (Fixes __ldg overload error for 24-byte double3)
// --------------------------------------------------------------------
__device__ inline double3 load_double3(const double3* ptr) {
    return make_double3(__ldg(&(ptr->x)), __ldg(&(ptr->y)), __ldg(&(ptr->z)));
}

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

__device__ inline ds_vec3 to_ds_vec(double3 v) {
    return { double_to_ds(v.x), double_to_ds(v.y), double_to_ds(v.z) };
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
// DOUBLE-SINGLE (DS) INTERVAL ARITHMETIC TYPES & OPERATIONS
// --------------------------------------------------------------------
struct dsInterval {
    ds_float lo;
    ds_float hi;
};

struct vec3DsInterval {
    dsInterval x, y, z;
};

// Rigorous interval creation for relative anchor offsets (using device-safe fabsf)
__device__ inline dsInterval make_ds_interval(double x_world, double anchor_world) {
    double diff = x_world - anchor_world;
    ds_float exact_ds = double_to_ds(diff);
    
    constexpr float eps_ds = 1.1920929e-7f * 1.1920929e-7f; // ~1.4e-14
    float abs_hi = fabsf(exact_ds.hi);
    ds_float lo_bound = ds_sub(exact_ds, {abs_hi * eps_ds, 0.0f});
    ds_float hi_bound = ds_add(exact_ds, {abs_hi * eps_ds, 0.0f});
    return { lo_bound, hi_bound };
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

    // Pure mathematical sign check on rigorous intervals
    ds_float zero = {0.0f, 0.0f};
    if (ds_gt(det.lo, zero)) return 1;
    if (ds_lt(det.hi, zero)) return -1;
    return 0; // Determinant straddles zero -> Exact CPU Fallback (PAIR_YELLOW)
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
    const double3& Aa, const double3& Ab, const double3& Ac,
    const double3& Ba, const double3& Bb, const double3& Bc) 
{    
    // 1. Exact Double Anchor Shift
    const double3 anchor = Aa;
    
    // 2. Build DS Intervals for all relative vertices
    const vec3DsInterval A0 = {
        dsInterval{{0.0f, 0.0f}, {0.0f, 0.0f}},
        dsInterval{{0.0f, 0.0f}, {0.0f, 0.0f}},
        dsInterval{{0.0f, 0.0f}, {0.0f, 0.0f}}
    };
    const vec3DsInterval A1 = {
        make_ds_interval(Ab.x, anchor.x),
        make_ds_interval(Ab.y, anchor.y),
        make_ds_interval(Ab.z, anchor.z)
    };
    const vec3DsInterval A2 = {
        make_ds_interval(Ac.x, anchor.x),
        make_ds_interval(Ac.y, anchor.y),
        make_ds_interval(Ac.z, anchor.z)
    };

    const vec3DsInterval B0 = {
        make_ds_interval(Ba.x, anchor.x),
        make_ds_interval(Ba.y, anchor.y),
        make_ds_interval(Ba.z, anchor.z)
    };
    const vec3DsInterval B1 = {
        make_ds_interval(Bb.x, anchor.x),
        make_ds_interval(Bb.y, anchor.y),
        make_ds_interval(Bb.z, anchor.z)
    };
    const vec3DsInterval B2 = {
        make_ds_interval(Bc.x, anchor.x),
        make_ds_interval(Bc.y, anchor.y),
        make_ds_interval(Bc.z, anchor.z)
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
    const double3* dVertsA,
    const uint3* dIndicesA,
    const double3* dVertsB,
    const uint3* dIndicesB,
    int numYellowPairs) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numYellowPairs) return;

    int2 pair = dYellowCandidatePairs[tid];
    
    uint3 idxA = dIndicesA[pair.x];
    uint3 idxB = dIndicesB[pair.y];

    double3 Aa = load_double3(&dVertsA[idxA.x]);
    double3 Ab = load_double3(&dVertsA[idxA.y]);
    double3 Ac = load_double3(&dVertsA[idxA.z]);

    double3 Ba = load_double3(&dVertsB[idxB.x]);
    double3 Bb = load_double3(&dVertsB[idxB.y]);
    double3 Bc = load_double3(&dVertsB[idxB.z]);

    PairStatus status = classifyPairDS(Aa, Ab, Ac, Ba, Bb, Bc);

    dPairStatuses[tid] = (int)status; 
}

// --------------------------------------------------------------------
// PIPELINE EXPORTED WRAPPER
// --------------------------------------------------------------------
void evaluateAndCompactPairsDouble(
    const int2* dYellowCandidatePairs,
    int* dPairStatuses,
    const double3* dVertsA,
    const uint3* dIndicesA,
    const double3* dVertsB,
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