#include "DoublePredicate.h"
#include "GPUPredicatesCommon.h"
#include <cmath>

namespace pred_double {

using vec3D = vec3Interval<double>;

// --------------------------------------------------------------------
// FP64 Interval Arithmetic Helpers (Directed Rounding)
// --------------------------------------------------------------------
__device__ inline GPUInterval<double> dinterval_sub(GPUInterval<double> a, GPUInterval<double> b) {
    return GPUInterval<double>{ __dsub_rd(a.lo, b.hi), __dsub_ru(a.hi, b.lo) };
}

__device__ inline GPUInterval<double> dinterval_add(GPUInterval<double> a, GPUInterval<double> b) {
    return GPUInterval<double>{ __dadd_rd(a.lo, b.lo), __dadd_ru(a.hi, b.hi) };
}

__device__ inline GPUInterval<double> dinterval_mul(GPUInterval<double> a, GPUInterval<double> b) {
    double lo1 = __dmul_rd(a.lo, b.lo); double lo2 = __dmul_rd(a.lo, b.hi);
    double lo3 = __dmul_rd(a.hi, b.lo); double lo4 = __dmul_rd(a.hi, b.hi);
    double hi1 = __dmul_ru(a.lo, b.lo); double hi2 = __dmul_ru(a.lo, b.hi);
    double hi3 = __dmul_ru(a.hi, b.lo); double hi4 = __dmul_ru(a.hi, b.hi);
    return GPUInterval<double>{ fmin(fmin(lo1, lo2), fmin(lo3, lo4)), fmax(fmax(hi1, hi2), fmax(hi3, hi4)) };
}

// --------------------------------------------------------------------
// Vertex Adapters & Shifting (Exact FP64 to FP64 Intervals)
// --------------------------------------------------------------------
__device__ inline vec3IntervalDouble toInterval(const vec3IntervalDouble& v) { return v; }
__device__ inline vec3IntervalDouble toInterval(const double3& p) {
    return vec3IntervalDouble{ {p.x, p.x}, {p.y, p.y}, {p.z, p.z} };
}

__device__ inline GPUInterval<double> shiftToDouble(const GPUInterval<double>& x,
                                                   const GPUInterval<double>& a)
{
    return GPUInterval<double>{
        __dsub_rd(x.lo, a.hi),
        __dsub_ru(x.hi, a.lo)
    };
}

__device__ inline vec3D shiftToDouble(const vec3IntervalDouble& v, const vec3IntervalDouble& a) {
    return vec3D{ shiftToDouble(v.x, a.x), shiftToDouble(v.y, a.y), shiftToDouble(v.z, a.z) };
}

// --------------------------------------------------------------------
// FP64 Interval Vector Operations
// --------------------------------------------------------------------
__device__ inline vec3D vsubD(const vec3D& a, const vec3D& b) {
    return vec3D{ dinterval_sub(a.x, b.x), dinterval_sub(a.y, b.y), dinterval_sub(a.z, b.z) };
}
__device__ inline vec3D vcrossD(const vec3D& a, const vec3D& b) {
    return vec3D{
        dinterval_sub(dinterval_mul(a.y, b.z), dinterval_mul(a.z, b.y)),
        dinterval_sub(dinterval_mul(a.z, b.x), dinterval_mul(a.x, b.z)),
        dinterval_sub(dinterval_mul(a.x, b.y), dinterval_mul(a.y, b.x))
    };
}
__device__ inline GPUInterval<double> vdotD(const vec3D& a, const vec3D& b) {
    return dinterval_add(dinterval_add(dinterval_mul(a.x, b.x), dinterval_mul(a.y, b.y)),
                         dinterval_mul(a.z, b.z));
}
__device__ inline int isignD(const GPUInterval<double>& v) {
    if (v.lo > 0.0) return 1;
    if (v.hi < 0.0) return -1;
    return 0;
}

// --------------------------------------------------------------------
// Rigorous FP64 orient3d & edgeTri using interval helpers
// --------------------------------------------------------------------
__device__ inline int orient3d_double_interval(
    const vec3D& p, const vec3D& q, const vec3D& r, const vec3D& s) 
{
    vec3D pdx = vsubD(p, s);
    vec3D qdx = vsubD(q, s);
    vec3D rdx = vsubD(r, s);

    GPUInterval<double> det = dinterval_add(
        dinterval_add(
            dinterval_mul(pdx.x, dinterval_sub(dinterval_mul(qdx.y, rdx.z), dinterval_mul(qdx.z, rdx.y))),
            dinterval_mul(pdx.y, dinterval_sub(dinterval_mul(qdx.z, rdx.x), dinterval_mul(qdx.x, rdx.z)))
        ),
        dinterval_mul(pdx.z, dinterval_sub(dinterval_mul(qdx.x, rdx.y), dinterval_mul(qdx.y, rdx.x)))
    );

    if (det.lo > 0.0) return 1;
    if (det.hi < 0.0) return -1;
    return 0; 
}

__device__ inline PairStatus edgeTriWithSignsD(
    const vec3D& a, const vec3D& b, int sa, int sb,
    const vec3D& p, const vec3D& q, const vec3D& r)
{
    if (sa == 0 || sb == 0) return PAIR_YELLOW;
    if (sa == sb)           return PAIR_NO;

    int e0 = orient3d_double_interval(a, b, p, q);
    int e1 = orient3d_double_interval(a, b, q, r);
    int e2 = orient3d_double_interval(a, b, r, p);

    if (e0 == 0 || e1 == 0 || e2 == 0) return PAIR_YELLOW;
    if ((e0 > 0 && e1 > 0 && e2 > 0) || (e0 < 0 && e1 < 0 && e2 < 0)) return PAIR_GREEN;
    return PAIR_NO;
}

// --------------------------------------------------------------------
// Optimized Core Classification (Anchor-Relative FP64 Intervals)
// --------------------------------------------------------------------
__device__ inline PairStatus classifyLocalD(
    const vec3D& A1, const vec3D& A2,
    const vec3D& B0, const vec3D& B1, const vec3D& B2)
{
    const GPUInterval<double> Z{0.0, 0.0};
    const vec3D A0{Z, Z, Z};

    // --- B vertices vs plane A (cached cross-product normal nA)
    const vec3D nA = vcrossD(A1, A2);
    const int sB0 = -isignD(vdotD(B0, nA));
    const int sB1 = -isignD(vdotD(B1, nA));
    const int sB2 = -isignD(vdotD(B2, nA));
    if (sB0 != 0 && sB0 == sB1 && sB1 == sB2) return PAIR_NO;

    // --- A vertices vs plane B (cached cross-product normal nB)
    const vec3D nB = vcrossD(vsubD(B1, B0), vsubD(B2, B0));
    const int sA0 = isignD(vdotD(B0, nB));
    const int sA1 = isignD(vdotD(vsubD(B0, A1), nB));
    const int sA2 = isignD(vdotD(vsubD(B0, A2), nB));
    if (sA0 != 0 && sA0 == sA1 && sA1 == sA2) return PAIR_NO;

    PairStatus r;
    r = edgeTriWithSignsD(A0, A1, sA0, sA1, B0, B1, B2); if (r != PAIR_NO) return r;
    r = edgeTriWithSignsD(A1, A2, sA1, sA2, B0, B1, B2); if (r != PAIR_NO) return r;
    r = edgeTriWithSignsD(A2, A0, sA2, sA0, B0, B1, B2); if (r != PAIR_NO) return r;

    r = edgeTriWithSignsD(B0, B1, sB0, sB1, A0, A1, A2); if (r != PAIR_NO) return r;
    r = edgeTriWithSignsD(B1, B2, sB1, sB2, A0, A1, A2); if (r != PAIR_NO) return r;
    r = edgeTriWithSignsD(B2, B0, sB2, sB0, A0, A1, A2); if (r != PAIR_NO) return r;

    return PAIR_NO;
}

template <class VA, class VB>
__device__ inline PairStatus classifyPair(
    const VA& Aa, const VA& Ab, const VA& Ac,
    const VB& Ba, const VB& Bb, const VB& Bc)
{
    const vec3IntervalDouble anchor = toInterval(Aa);
    const vec3D A1 = shiftToDouble(toInterval(Ab), anchor);
    const vec3D A2 = shiftToDouble(toInterval(Ac), anchor);
    const vec3D B0 = shiftToDouble(toInterval(Ba), anchor);
    const vec3D B1 = shiftToDouble(toInterval(Bb), anchor);
    const vec3D B2 = shiftToDouble(toInterval(Bc), anchor);
    return classifyLocalD(A1, A2, B0, B1, B2);
}

} // namespace pred_double

// --------------------------------------------------------------------
// Kernel & Exported Wrapper Implementation
// --------------------------------------------------------------------
template <class VA, class VB>
__global__ void evaluateGeometricPairsKernelDoubleFull(
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

    outStatuses[tid] = (int)pred_double::classifyPair(Aa, Ab, Ac, Ba, Bb, Bc);
}

template <class VA, class VB>
void evaluateAndCompactPairsDoubleFull(
    const int2* dCandidatePairs, int* dPairStatuses,
    const VA* dVertsA, const uint3* dIndicesA,
    const VB* dVertsB, const uint3* dIndicesB,
    int totalBatchPairs, cudaStream_t stream)
{
    if (totalBatchPairs <= 0) return;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (totalBatchPairs + threadsPerBlock - 1) / threadsPerBlock;

    evaluateGeometricPairsKernelDoubleFull<VA, VB><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        dPairStatuses, dCandidatePairs, dVertsA, dIndicesA, dVertsB, dIndicesB, totalBatchPairs);
}

// Explicit instantiations for all 4 mesh vertex type combinations
#define INSTANTIATE_DOUBLE_PAIR_EVAL(VA, VB)                                            \
    template void evaluateAndCompactPairsDoubleFull<VA, VB>(                            \
        const int2*, int*, const VA*, const uint3*, const VB*, const uint3*, int, cudaStream_t);

INSTANTIATE_DOUBLE_PAIR_EVAL(vec3IntervalDouble, vec3IntervalDouble)
INSTANTIATE_DOUBLE_PAIR_EVAL(double3,            double3)
INSTANTIATE_DOUBLE_PAIR_EVAL(double3,            vec3IntervalDouble)
INSTANTIATE_DOUBLE_PAIR_EVAL(vec3IntervalDouble, double3)