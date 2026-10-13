#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/intersections.h>
#include <boost/graph/graph_traits.hpp>

// Toggle via preprocessor definition
#ifdef USE_EXACT_KERNEL
  typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
#else
  typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
#endif

typedef CGAL::Surface_mesh<Kernel::Point_3> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;
typedef Kernel::Point_3 Point3;
typedef Kernel::Triangle_3 Triangle3;