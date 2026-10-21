#pragma once

#include <cuda_runtime.h>

// Generic interval structure template
template <typename T>
struct GPUInterval {
    T lo;
    T hi;
};

// Generic 3D vector of intervals
template <typename T>
struct vec3Interval {
    GPUInterval<T> x, y, z;
};

// Explicit type aliases for Float (FP32) and Double (FP64)
using GPUIntervalFloat  = GPUInterval<float>;
using GPUIntervalDouble = GPUInterval<double>;

using vec3IntervalFloat  = vec3Interval<float>;
using vec3IntervalDouble = vec3Interval<double>;