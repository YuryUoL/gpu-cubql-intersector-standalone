#ifndef CGAL_PARALLEL_LAUNCHER_H
#define CGAL_PARALLEL_LAUNCHER_H

#include "ApplicationState.h"

// Returns elapsed time in milliseconds and output triangle pairs
double computeCGALParallel(const Mesh& meshA, 
                          const Mesh& meshB, 
                          std::vector<std::pair<size_t, size_t>>& outPairs);

bool isCGALParallelSupported();

#endif // CGAL_PARALLEL_LAUNCHER_H