#include "DoublePredicate.h"
#include "GPUPredicatesCommon.h"
#include <cmath>

// --------------------------------------------------------------------
// FP64 INTERVAL TYPES & DIRECTED ROUNDING INTRINSICS
// --------------------------------------------------------------------
struct DoubleInterval {
    double lo;
    double hi;
};

// Rigorous interval creation from world coordinates using FP64 directed rounding
__device__ inline DoubleInterval make_double_interval(double x_world, double anchor_world) {
    return DoubleInterval{
        __dsub_rd(x_world, anchor_world), // Force-rounds down toward -inf
        __dsub_ru(x_world, anchor_world)  // Force-rounds up toward +inf
    };
}

__device__ inline DoubleInterval dinterval_sub(DoubleInterval a, DoubleInterval b) {
    return DoubleInterval{
        __dsub_rd(a.lo, b.hi),
        __dsub_ru(a.hi, b.lo)
    };
}

__device__ inline DoubleInterval dinterval_add(DoubleInterval a, DoubleInterval b) {
    return DoubleInterval{
        __dadd_rd(a.lo, b.lo),
        __dadd_ru(a.hi, b.hi)
    };
}

__device__ inline DoubleInterval dinterval_mul(DoubleInterval a, DoubleInterval b) {
    double lo1 = __dmul_rd(a.lo, b.lo); double lo2 = __dmul_rd(a.lo, b.hi);
    double lo3 = __dmul_rd(a.hi, b.lo); double lo4 = __dmul_rd(a.hi, b.hi);
    double hi1 = __dmul_ru(a.lo, b.lo); double hi2 = __dmul_ru(a.lo, b.hi);
    double hi3 = __dmul_ru(a.hi, b.lo); double hi4 = __dmul_ru(a.hi, b.hi);
    return DoubleInterval{
        fmin(fmin(lo1, lo2), fmin(lo3, lo4)),
        fmax(fmax(hi1, hi2), fmax(hi3, hi4))
    };
}

struct vec3DoubleInterval {
    DoubleInterval x, y, z;
};

// --------------------------------------------------------------------
// RIGOROUS FP64 INTERVAL ORIENT3D & EDGETRI PREDICATES
// --------------------------------------------------------------------
__device__ inline int orient3d_double_interval(
    const vec3DoubleInterval& p, const vec3DoubleInterval& q, 
    const vec3DoubleInterval& r, const vec3DoubleInterval& s) 
{
    DoubleInterval pdx = dinterval_sub(p.x, s.x); 
    DoubleInterval pdy = dinterval_sub(p.y, s.y); 
    DoubleInterval pdz = dinterval_sub(p.z, s.z);

    DoubleInterval qdx = dinterval_sub(q.x, s.x); 
    DoubleInterval qdy = dinterval_sub(q.y, s.y); 
    DoubleInterval qdz = dinterval_sub(q.z, s.z);

    DoubleInterval rdx = dinterval_sub(r.x, s.x); 
    DoubleInterval rdy = dinterval_sub(r.y, s.y); 
    DoubleInterval rdz = dinterval_sub(r.z, s.z);

    DoubleInterval det = dinterval_add(
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
    const vec3DoubleInterval &a, const vec3DoubleInterval &b,
    const vec3DoubleInterval &p, const vec3DoubleInterval &q, const vec3DoubleInterval &r) 
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
// MEMORY HELPER
// --------------------------------------------------------------------
__device__ inline double3 load_double3(const double3* ptr) {
    return make_double3(__ldg(&(ptr->x)), __ldg(&(ptr->y)), __ldg(&(ptr->z)));
}

// --------------------------------------------------------------------
// CLASSIFICATION FUNCTION (Hardware FP64 Interval Arithmetic)
// --------------------------------------------------------------------
__device__ inline PairStatus classifyPairDoubleFull(
    const double3& Aa, const double3& Ab, const double3& Ac,
    const double3& Ba, const double3& Bb, const double3& Bc) 
{    
    // 1. Exact Double Anchor Shift with Rigorous Interval Construction
    const double3 anchor = Aa;
    
    const vec3DoubleInterval A0 = {
        DoubleInterval{0.0, 0.0}, DoubleInterval{0.0, 0.0}, DoubleInterval{0.0, 0.0}
    };
    const vec3DoubleInterval A1 = {
        make_double_interval(Ab.x, anchor.x),
        make_double_interval(Ab.y, anchor.y),
        make_double_interval(Ab.z, anchor.z)
    };
    const vec3DoubleInterval A2 = {
        make_double_interval(Ac.x, anchor.x),
        make_double_interval(Ac.y, anchor.y),
        make_double_interval(Ac.z, anchor.z)
    };

    const vec3DoubleInterval B0 = {
        make_double_interval(Ba.x, anchor.x),
        make_double_interval(Ba.y, anchor.y),
        make_double_interval(Ba.z, anchor.z)
    };
    const vec3DoubleInterval B1 = {
        make_double_interval(Bb.x, anchor.x),
        make_double_interval(Bb.y, anchor.y),
        make_double_interval(Bb.z, anchor.z)
    };
    const vec3DoubleInterval B2 = {
        make_double_interval(Bc.x, anchor.x),
        make_double_interval(Bc.y, anchor.y),
        make_double_interval(Bc.z, anchor.z)
    };

    // 2. Early Rejection Checks using Interval Orientations
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

    // 3. Detailed Edge-Triangle Interval Predicates
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

    PairStatus status = classifyPairDoubleFull(Aa, Ab, Ac, Ba, Bb, Bc);

    dPairStatuses[tid] = (int)status; 
}

void evaluateAndCompactPairsDoubleFull(
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
    
    evaluateGeometricPairsKernelDoubleFull<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
         dPairStatuses, dYellowCandidatePairs, 
         dVertsA, dIndicesA, 
         dVertsB, dIndicesB, 
         numYellowPairs
    );
}