#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "PairStatus.h"
#include "cuBQL/bvh.h"
#include "samples/common/loadOBJ.h"

#ifndef CUBQL_CUDA_CALL
#define CUBQL_CUDA_CALL(X) { cudaError_t err = cuda ## X; if(err != cudaSuccess) { printf("CUDA Error at %s:%d\n", __FILE__, __LINE__); } }
#endif

// --------------------------------------------------------------------
// INTERVAL ARITHMETIC & GEOMETRIC TYPES
// --------------------------------------------------------------------
struct GPUInterval {
    float lo;
    float hi;
};

struct vec3Interval {
    GPUInterval x, y, z;
};

// Rigorously constructs a provable FP32 interval from an exact double relative shift.
// Accounts for FP64 subtraction error + FP32 cast truncation via hardware directed rounding.
__device__ inline GPUInterval make_provable_relative_interval(double x_world, double anchor_world) {
    double diff_64 = x_world - anchor_world;
    float f = static_cast<float>(diff_64);

    // Provable combined relative error bound: 2^-24 + 2^-53 + 2^-77
    constexpr float C_provable = 5.9604650e-8f; 
    float err = fabsf(f) * C_provable;

    // Subnormal / underflow floor safeguard
    constexpr float min_normal_float = 1.17549435e-38f;
    err = fmaxf(err, min_normal_float * 2.0f);

    return GPUInterval{
        __fsub_rd(f, err),
        __fadd_ru(f, err)
    };
}

__device__ inline GPUInterval interval_sub(GPUInterval a, GPUInterval b) { 
    return { __fsub_rd(a.lo, b.hi), __fsub_ru(a.hi, b.lo) }; 
}

__device__ inline GPUInterval interval_add(GPUInterval a, GPUInterval b) { 
    return { __fadd_rd(a.lo, b.lo), __fadd_ru(a.hi, b.hi) }; 
}

__device__ inline GPUInterval interval_mul(GPUInterval a, GPUInterval b) {
    float lo1 = __fmul_rd(a.lo, b.lo); float lo2 = __fmul_rd(a.lo, b.hi);
    float lo3 = __fmul_rd(a.hi, b.lo); float lo4 = __fmul_rd(a.hi, b.hi);
    float hi1 = __fmul_ru(a.lo, b.lo); float hi2 = __fmul_ru(a.lo, b.hi);
    float hi3 = __fmul_ru(a.hi, b.lo); float hi4 = __fmul_ru(a.hi, b.hi);
    return { fminf(fminf(lo1, lo2), fminf(lo3, lo4)), fmaxf(fmaxf(hi1, hi2), fmaxf(hi3, hi4)) };
}

__device__ inline GPUInterval interval_scale(GPUInterval a, float s) {
    float p1 = __fmul_rd(a.lo, s);
    float p2 = __fmul_rd(a.hi, s);
    float p3 = __fmul_ru(a.lo, s);
    float p4 = __fmul_ru(a.hi, s);
    return GPUInterval{ fminf(p1, p2), fmaxf(p3, p4) };
}

__device__ inline vec3Interval scale_vec3_interval(const vec3Interval& v, float scale) {
    return vec3Interval{
        interval_scale(v.x, scale),
        interval_scale(v.y, scale),
        interval_scale(v.z, scale)
    };
}

// --------------------------------------------------------------------
// INTERVAL-BASED PREDICATES (ZERO EPSILON REQUIRED)
// --------------------------------------------------------------------
__device__ inline int orient3d_interval(
    const vec3Interval& p, const vec3Interval& q, 
    const vec3Interval& r, const vec3Interval& s) 
{
    GPUInterval pdx = interval_sub(p.x, s.x); 
    GPUInterval pdy = interval_sub(p.y, s.y); 
    GPUInterval pdz = interval_sub(p.z, s.z);

    GPUInterval qdx = interval_sub(q.x, s.x); 
    GPUInterval qdy = interval_sub(q.y, s.y); 
    GPUInterval qdz = interval_sub(q.z, s.z);

    GPUInterval rdx = interval_sub(r.x, s.x); 
    GPUInterval rdy = interval_sub(r.y, s.y); 
    GPUInterval rdz = interval_sub(r.z, s.z);

    GPUInterval det = interval_add(
        interval_add(
            interval_mul(pdx, interval_sub(interval_mul(qdy, rdz), interval_mul(qdz, rdy))),
            interval_mul(pdy, interval_sub(interval_mul(qdz, rdx), interval_mul(qdx, rdz)))
        ),
        interval_mul(pdz, interval_sub(interval_mul(qdx, rdy), interval_mul(qdy, rdx)))
    );

    if (det.lo > 0.0f) return 1;
    if (det.hi < 0.0f) return -1;
    return 0; // Determinant straddles zero -> ambiguous/fallback required
}

__device__ inline int edgeTriInterval(
    const vec3Interval &a, const vec3Interval &b,
    const vec3Interval &p, const vec3Interval &q, const vec3Interval &r) 
{
    int s0 = orient3d_interval(p, q, r, a);
    int s1 = orient3d_interval(p, q, r, b);

    if (s0 == 0 || s1 == 0) return PAIR_YELLOW;
    if ((s0 > 0 && s1 > 0) || (s0 < 0 && s1 < 0)) return PAIR_NO;

    int e0 = orient3d_interval(a, b, p, q);
    int e1 = orient3d_interval(a, b, q, r);
    int e2 = orient3d_interval(a, b, r, p);

    if (e0 == 0 || e1 == 0 || e2 == 0) return PAIR_YELLOW;
    if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) return PAIR_GREEN;

    return PAIR_NO;
}