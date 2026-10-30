#pragma once
// Templated core of the origin-shifted FP32 interval filter.
// Vertex types supported: double3 (exact points) and vec3IntervalDouble (boxes).
// Everything after the shift runs on FP32 intervals, so there is ONE predicate core.
//
// SOUNDNESS REQUIREMENTS
//  * Compile WITHOUT -ftz=true / --use_fast_math (directed rounding must be IEEE for subnormals).
//  * Coordinates must be finite and |coordinate differences| must not overflow FP32.

#include <cuda_runtime.h>
#include "PairStatus.h"
#include "GPUInterval.h"
#include "GPUPredicatesCommon.h"   // interval_add/sub/mul, orient3d_interval

namespace pred {

using vec3F = vec3Interval<float>;

// ---------------------------------------------------------------------------
// Vertex adapters: any supported vertex type -> FP64 interval box
// ---------------------------------------------------------------------------
__device__ inline vec3IntervalDouble toInterval(const vec3IntervalDouble& v) { return v; }
__device__ inline vec3IntervalDouble toInterval(const double3& p) {
    return vec3IntervalDouble{ {p.x, p.x}, {p.y, p.y}, {p.z, p.z} };
}

// ---------------------------------------------------------------------------
// Rigorous shift (x - anchor) into an FP32 interval.
// Directed rounding in both steps already gives a guaranteed enclosure, so NO
// extra relative inflation is needed (the old C_provable term only widened it).
// ---------------------------------------------------------------------------
__device__ inline GPUInterval<float> shiftToFloat(const GPUInterval<double>& x,
                                                  const GPUInterval<double>& a)
{
    return GPUInterval<float>{
        __double2float_rd(__dsub_rd(x.lo, a.hi)),
        __double2float_ru(__dsub_ru(x.hi, a.lo))
    };
}

__device__ inline vec3F shiftToFloat(const vec3IntervalDouble& v, const vec3IntervalDouble& a) {
    return vec3F{ shiftToFloat(v.x, a.x), shiftToFloat(v.y, a.y), shiftToFloat(v.z, a.z) };
}

// ---------------------------------------------------------------------------
// Small FP32 interval vector helpers (all outward rounded via GPUPredicatesCommon.h)
// ---------------------------------------------------------------------------
__device__ inline vec3F vsub(const vec3F& a, const vec3F& b) {
    return vec3F{ interval_sub(a.x, b.x), interval_sub(a.y, b.y), interval_sub(a.z, b.z) };
}
__device__ inline vec3F vcross(const vec3F& a, const vec3F& b) {
    return vec3F{
        interval_sub(interval_mul(a.y, b.z), interval_mul(a.z, b.y)),
        interval_sub(interval_mul(a.z, b.x), interval_mul(a.x, b.z)),
        interval_sub(interval_mul(a.x, b.y), interval_mul(a.y, b.x))
    };
}
__device__ inline GPUInterval<float> vdot(const vec3F& a, const vec3F& b) {
    return interval_add(interval_add(interval_mul(a.x, b.x), interval_mul(a.y, b.y)),
                        interval_mul(a.z, b.z));
}
// +1 / -1 only if the interval excludes zero, else 0 ("maybe")
__device__ inline int isign(const GPUInterval<float>& v) {
    if (v.lo > 0.0f) return 1;
    if (v.hi < 0.0f) return -1;
    return 0;
}

// ---------------------------------------------------------------------------
// Edge (a,b) vs triangle (p,q,r), with the endpoint-vs-plane signs already known.
// ---------------------------------------------------------------------------
__device__ inline PairStatus edgeTriWithSigns(
    const vec3F& a, const vec3F& b, int sa, int sb,
    const vec3F& p, const vec3F& q, const vec3F& r)
{
    if (sa == 0 || sb == 0) return PAIR_YELLOW;
    if (sa == sb)           return PAIR_NO;       // both strictly on the same side

    int e0 = orient3d_interval(a, b, p, q);
    int e1 = orient3d_interval(a, b, q, r);
    int e2 = orient3d_interval(a, b, r, p);

    if (e0 == 0 || e1 == 0 || e2 == 0) return PAIR_YELLOW;
    if ((e0 > 0 && e1 > 0 && e2 > 0) || (e0 < 0 && e1 < 0 && e2 < 0)) return PAIR_GREEN;
    return PAIR_NO;
}

// ---------------------------------------------------------------------------
// Predicate core on anchor-relative FP32 intervals. A0 is exactly the origin.
//   orient3d(p,q,r,s) = (p - s) . ((q-p) x (r-p))
// so each triangle's normal is computed once and reused for 3 vertex tests.
// ---------------------------------------------------------------------------
__device__ inline PairStatus classifyLocal(
    const vec3F& A1, const vec3F& A2,
    const vec3F& B0, const vec3F& B1, const vec3F& B2)
{
    const GPUInterval<float> Z{0.0f, 0.0f};
    const vec3F A0{Z, Z, Z};

    // --- B vertices vs plane A: orient3d(A0,A1,A2,X) = (A0 - X).nA = -(X.nA), nA = A1 x A2
    const vec3F nA = vcross(A1, A2);
    const int sB0 = -isign(vdot(B0, nA));
    const int sB1 = -isign(vdot(B1, nA));
    const int sB2 = -isign(vdot(B2, nA));
    if (sB0 != 0 && sB0 == sB1 && sB1 == sB2) return PAIR_NO;

    // --- A vertices vs plane B: orient3d(B0,B1,B2,X) = (B0 - X).nB
    const vec3F nB = vcross(vsub(B1, B0), vsub(B2, B0));
    const int sA0 = isign(vdot(B0, nB));              // X = A0 = origin
    const int sA1 = isign(vdot(vsub(B0, A1), nB));
    const int sA2 = isign(vdot(vsub(B0, A2), nB));
    if (sA0 != 0 && sA0 == sA1 && sA1 == sA2) return PAIR_NO;

    PairStatus r;
    r = edgeTriWithSigns(A0, A1, sA0, sA1, B0, B1, B2); if (r != PAIR_NO) return r;
    r = edgeTriWithSigns(A1, A2, sA1, sA2, B0, B1, B2); if (r != PAIR_NO) return r;
    r = edgeTriWithSigns(A2, A0, sA2, sA0, B0, B1, B2); if (r != PAIR_NO) return r;

    r = edgeTriWithSigns(B0, B1, sB0, sB1, A0, A1, A2); if (r != PAIR_NO) return r;
    r = edgeTriWithSigns(B1, B2, sB1, sB2, A0, A1, A2); if (r != PAIR_NO) return r;
    r = edgeTriWithSigns(B2, B0, sB2, sB0, A0, A1, A2); if (r != PAIR_NO) return r;

    return PAIR_NO;
}

// ---------------------------------------------------------------------------
// Front end: any mix of vertex types for mesh A and mesh B.
// ---------------------------------------------------------------------------
template <class VA, class VB>
__device__ inline PairStatus classifyPair(
    const VA& Aa, const VA& Ab, const VA& Ac,
    const VB& Ba, const VB& Bb, const VB& Bc)
{
    const vec3IntervalDouble anchor = toInterval(Aa);
    const vec3F A1 = shiftToFloat(toInterval(Ab), anchor);
    const vec3F A2 = shiftToFloat(toInterval(Ac), anchor);
    const vec3F B0 = shiftToFloat(toInterval(Ba), anchor);
    const vec3F B1 = shiftToFloat(toInterval(Bb), anchor);
    const vec3F B2 = shiftToFloat(toInterval(Bc), anchor);
    return classifyLocal(A1, A2, B0, B1, B2);
}

} // namespace pred
