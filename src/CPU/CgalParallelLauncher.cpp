#include "CgalParallelLauncher.h"
#include <chrono>
#include <iostream>


#ifdef ENABLE_CGAL_PARALLEL
// Include the experimental CGAL headers only when compiled with the flag
//#include <CGAL/AABB_meshes_intersections.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <AABB_meshes_intersections.h>
namespace PMP = CGAL::Polygon_mesh_processing;
#endif

bool isCGALParallelSupported() {
#ifdef ENABLE_CGAL_PARALLEL
    return true;
#else
    return false;
#endif
}

double computeCGALParallel(const Mesh& meshA, 
                          const Mesh& meshB, 
                          std::vector<std::pair<size_t, size_t>>& outPairs) 
{
    outPairs.clear();
#ifdef ENABLE_CGAL_PARALLEL
    auto tStart = std::chrono::high_resolution_clock::now();

    using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
    std::vector<std::pair<face_descriptor, face_descriptor>> cgalIntersectedTris;

    PMP::AABB_meshes_intersections(meshA, meshB, std::back_inserter(cgalIntersectedTris),
                                   CGAL::parameters::concurrency_tag(CGAL::Parallel_tag()));

    auto tEnd = std::chrono::high_resolution_clock::now();

    outPairs.reserve(cgalIntersectedTris.size());
    for(const auto& pair : cgalIntersectedTris) {
        outPairs.emplace_back(static_cast<size_t>(pair.first), static_cast<size_t>(pair.second));
    }

    return std::chrono::duration<double, std::milli>(tEnd - tStart).count();
#else
    std::cerr << "Error: CGAL Parallel is not compiled into this build!\n";
    return 0.0;
#endif
}