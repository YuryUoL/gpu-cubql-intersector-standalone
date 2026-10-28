#pragma once
// RobustOffLoader.h
// Classic-style (CGAL) OFF loader that does NOT fail on non-manifold / degenerate meshes and
// guarantees:   CGAL face i  <->  outIndices[i]   and   CGAL vertex i  <->  outVerts[i].
// hIndices/hVerts are always derived from the finished CGAL mesh, never from the file rows.

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <CGAL/IO/OFF.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

#include <vector_types.h>  // double3, uint3

namespace RobustIO {

enum class NonManifoldPolicy {
  SplitVertices,  // keep every face; duplicate vertices at pinch points / non-manifold edges (default)
  SkipFaces       // keep original vertices; drop faces CGAL's add_face rejects (explicitly, with report)
};

struct OffLoadReport {
  size_t soupPoints = 0;
  size_t soupPolygons = 0;               // polygons in file
  size_t extraTrianglesFromFan = 0;      // added by fan-triangulating quads/ngons
  std::vector<uint32_t> droppedDegenerate;  // OFF rows with repeated vertex index
  std::vector<uint32_t> droppedDuplicate;   // OFF rows duplicating an earlier face
  std::vector<uint32_t> droppedByAddFace;   // OFF rows add_face rejected (SkipFaces only)
  size_t meshVertices = 0;
  size_t meshFaces = 0;
  size_t verticesAddedBySplitting = 0;
  std::vector<uint32_t> faceToRow;       // CGAL face index -> OFF row it came from

  void print(const std::string& tag) const {
    std::cout << "[" << tag << "] file: " << soupPoints << " verts, " << soupPolygons << " polygons"
              << " | mesh: " << meshVertices << " verts, " << meshFaces << " faces\n";
    if (extraTrianglesFromFan) std::cout << "  fan-triangulated extra triangles : " << extraTrianglesFromFan << "\n";
    auto line = [](const char* n, const std::vector<uint32_t>& v) {
      if (v.empty()) return;
      std::cout << "  " << n << ": " << v.size() << "  (first rows:";
      for (size_t i = 0; i < v.size() && i < 8; ++i) std::cout << " " << v[i];
      std::cout << ")\n";
    };
    line("dropped degenerate (repeated vertex)", droppedDegenerate);
    line("dropped duplicate faces             ", droppedDuplicate);
    line("dropped by add_face                 ", droppedByAddFace);
    if (verticesAddedBySplitting) std::cout << "  vertices added by splitting       : " << verticesAddedBySplitting << "\n";
  }
};

/// Rebuild GPU-side arrays from the mesh so indices are consistent with CGAL by construction.
template <typename Mesh>
inline bool deriveBuffersFromMesh(const Mesh& mesh, std::vector<double3>& outVerts, std::vector<uint3>& outIndices) {
  if (mesh.has_garbage()) {
    std::cerr << "[RobustIO] mesh has garbage (removed elements); indices would not be dense.\n";
    return false;
  }
  
  const size_t nv = mesh.number_of_vertices();
  const size_t nf = mesh.number_of_faces();
  
  // 1. Pre-allocate output buffers sequentially (CRITICAL for thread safety)
  outVerts.resize(nv);
  outIndices.resize(nf);
  
  const auto& pts = mesh.points();

  // 2. Parallelize vertex buffer extraction
  tbb::parallel_for(tbb::blocked_range<size_t>(0, nv), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const auto& p = pts[typename Mesh::Vertex_index(static_cast<typename Mesh::size_type>(i))];
      outVerts[i] = make_double3(CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z()));
    }
  });

  // 3. Parallelize face buffer extraction using integer indices
  tbb::parallel_for(tbb::blocked_range<size_t>(0, nf), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      typename Mesh::Face_index f(static_cast<typename Mesh::size_type>(i));
      auto h = mesh.halfedge(f);
      outIndices[i] = make_uint3(
          static_cast<uint32_t>(mesh.source(h)), 
          static_cast<uint32_t>(mesh.target(h)),
          static_cast<uint32_t>(mesh.target(mesh.next(h)))
      );
    }
  });

  return true;
}

/// Cheap sanity check: every face's vertex triple equals outIndices[face]. Returns #mismatches.
template <typename Mesh>
inline size_t countIndexMismatches(const Mesh& mesh, const std::vector<uint3>& idx) {
  if (idx.size() != mesh.number_of_faces()) return std::max(idx.size(), (size_t)mesh.number_of_faces());
  size_t bad = 0;
  for (auto f : mesh.faces()) {
    auto h = mesh.halfedge(f);
    uint3 g = idx[f.idx()];
    if ((uint32_t)mesh.source(h) != g.x || (uint32_t)mesh.target(h) != g.y ||
        (uint32_t)mesh.target(mesh.next(h)) != g.z)
      ++bad;
  }
  return bad;
}

template <typename Mesh, typename Point3>
inline bool loadOffRobust(const std::string& path,
                          Mesh& mesh,
                          std::vector<double3>& outVerts,
                          std::vector<uint3>& outIndices,
                          OffLoadReport& rep,
                          NonManifoldPolicy policy = NonManifoldPolicy::SplitVertices) {
  namespace PMP = CGAL::Polygon_mesh_processing;
  rep = OffLoadReport();

  std::ifstream in(path);
  if (!in.is_open()) {
    std::cerr << "[RobustIO] cannot open " << path << "\n";
    return false;
  }

  // 1. Read as polygon SOUP: no manifold requirement, so this cannot fail on topology.
  std::vector<Point3> points;
  std::vector<std::vector<std::size_t>> polys;
  if (!CGAL::IO::read_OFF(in, points, polys)) {
    std::cerr << "[RobustIO] OFF syntax/index error in " << path << " (not a topology problem)\n";
    return false;
  }
  rep.soupPoints = points.size();
  rep.soupPolygons = polys.size();

  // 2. Fan-triangulate, drop topologically degenerate and duplicate faces (explicitly, with row ids).
  std::vector<std::vector<std::size_t>> tris;
  std::vector<uint32_t> rows;
  tris.reserve(polys.size());
  rows.reserve(polys.size());
  for (size_t r = 0; r < polys.size(); ++r) {
    const auto& p = polys[r];
    if (p.size() < 3) { rep.droppedDegenerate.push_back((uint32_t)r); continue; }
    for (size_t k = 1; k + 1 < p.size(); ++k) {
      if (p[0] == p[k] || p[k] == p[k + 1] || p[0] == p[k + 1]) { rep.droppedDegenerate.push_back((uint32_t)r); continue; }
      if (k > 1) ++rep.extraTrianglesFromFan;
      tris.push_back({p[0], p[k], p[k + 1]});
      rows.push_back((uint32_t)r);
    }
  }
  {
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> keys(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
      std::array<size_t, 3> s{tris[i][0], tris[i][1], tris[i][2]};
      std::sort(s.begin(), s.end());
      keys[i] = {s[0], s[1], s[2], i};
    }
    std::sort(keys.begin(), keys.end());
    std::vector<char> drop(tris.size(), 0);
    for (size_t i = 1; i < keys.size(); ++i)
      if (std::get<0>(keys[i]) == std::get<0>(keys[i - 1]) && std::get<1>(keys[i]) == std::get<1>(keys[i - 1]) &&
          std::get<2>(keys[i]) == std::get<2>(keys[i - 1]))
        drop[std::get<3>(keys[i])] = 1;
    size_t w = 0;
    for (size_t i = 0; i < tris.size(); ++i) {
      if (drop[i]) { rep.droppedDuplicate.push_back(rows[i]); continue; }
      if (w != i) { tris[w] = std::move(tris[i]); rows[w] = rows[i]; }
      ++w;
    }
    tris.resize(w);
    rows.resize(w);
  }

  // 3. Build the CGAL mesh
  mesh.clear();
  if (policy == NonManifoldPolicy::SplitVertices) {
    // Orients consistently and duplicates points where the soup is non-manifold. Keeps polygon order.
    PMP::orient_polygon_soup(points, tris);
    PMP::polygon_soup_to_polygon_mesh(points, tris, mesh);
    rep.verticesAddedBySplitting = points.size() > rep.soupPoints ? points.size() - rep.soupPoints : 0;
    if (mesh.number_of_faces() == tris.size()) {
      rep.faceToRow = rows;
    } else {
      std::cerr << "[RobustIO] soup->mesh produced " << mesh.number_of_faces() << " of " << tris.size()
                << " faces; row mapping unavailable, use SkipFaces if you need it.\n";
      rep.faceToRow.clear();
    }
  } else {
    mesh.reserve(points.size(), tris.size() * 3, tris.size());
    std::vector<typename Mesh::Vertex_index> vh(points.size());
    for (size_t i = 0; i < points.size(); ++i) vh[i] = mesh.add_vertex(points[i]);
    for (size_t i = 0; i < tris.size(); ++i) {
      auto f = mesh.add_face(vh[tris[i][0]], vh[tris[i][1]], vh[tris[i][2]]);
      if (f == Mesh::null_face()) f = mesh.add_face(vh[tris[i][0]], vh[tris[i][2]], vh[tris[i][1]]);  // flipped
      if (f == Mesh::null_face()) { rep.droppedByAddFace.push_back(rows[i]); continue; }
      rep.faceToRow.push_back(rows[i]);  // faces are appended densely, so this index == f.idx()
    }
  }

  // 4. Derive GPU arrays from the final mesh => consistent by construction.
  if (!deriveBuffersFromMesh(mesh, outVerts, outIndices)) return false;
  rep.meshVertices = mesh.number_of_vertices();
  rep.meshFaces = mesh.number_of_faces();
  return true;
}

}  // namespace RobustIO