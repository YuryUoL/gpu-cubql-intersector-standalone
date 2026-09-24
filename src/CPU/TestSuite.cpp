#include "TestSuite.h"
#include "CPU/RotationTools.h"
#include "CPU/PolyscopeBridge.h"
#include "GPUIntersector/StandaloneBVHPipeline.h"
#include "CPU/CgalParallelLauncher.h"

#include "GPUIntersector/KernelBVHController.h"
#include "ApplicationState.h"

#include <CGAL/Polygon_mesh_processing/intersection.h>


#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <random>

#include <CGAL/Polygon_mesh_processing/measure.h>

TestSuite::TestSuite(ApplicationState& appState) : app_(appState) {}

std::set<std::pair<int, int>> TestSuite::convertToCanonicalSet(const std::vector<int2>& pairs) {
    std::set<std::pair<int, int>> canonicalSet;
    for (const auto& p : pairs) {
        canonicalSet.insert({p.x, p.y});
    }
    return canonicalSet;
}

double TestSuite::computeTotalMeshArea(const Mesh& mesh) {
    return CGAL::Polygon_mesh_processing::area(mesh);
}

double TestSuite::computeIntersectedSurfaceArea(
    const Mesh& meshA, 
    const Mesh& meshB, 
    const std::set<std::pair<int, int>>& intersectedPairs,
    double totalAreaA,
    double totalAreaB) 
{
    if (intersectedPairs.empty() || (totalAreaA + totalAreaB) <= 0.0) {
        return 0.0;
    }

    std::unordered_set<int> intersectedFacesA;
    std::unordered_set<int> intersectedFacesB;

    for (const auto& pair : intersectedPairs) {
        intersectedFacesA.insert(pair.first);
        intersectedFacesB.insert(pair.second);
    }

    double areaA_int = 0.0;
    auto facesA = faces(meshA);
    for (int faceIdx : intersectedFacesA) {
        auto faceIter = std::next(facesA.first, faceIdx);
        areaA_int += CGAL::Polygon_mesh_processing::face_area(*faceIter, meshA);
    }

    double areaB_int = 0.0;
    auto facesB = faces(meshB);
    for (int faceIdx : intersectedFacesB) {
        auto faceIter = std::next(facesB.first, faceIdx);
        areaB_int += CGAL::Polygon_mesh_processing::face_area(*faceIter, meshB);
    }

    return (areaA_int + areaB_int) / (totalAreaA + totalAreaB);
}

std::vector<int2> TestSuite::runCGALClassicGroundTruth(
    const double3& rotA, const double3& transA,
    const double3& rotB, const double3& transB,
    double& outTimeMs) 
{
    Mesh meshA_transformed, meshB_transformed;
    Point3 centerA(app_.normCenterA.x, app_.normCenterA.y, app_.normCenterA.z);
    Point3 centerB(app_.normCenterB.x, app_.normCenterB.y, app_.normCenterB.z);

    transformCgalMesh(app_.meshA, meshA_transformed, centerA, rotA, transA);
    transformCgalMesh(app_.meshB, meshB_transformed, centerB, rotB, transB);

    using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
    std::vector<std::pair<face_descriptor, face_descriptor>> cgal_intersected_tris;

    auto tStart = std::chrono::high_resolution_clock::now();
    CGAL::Polygon_mesh_processing::internal::compute_face_face_intersection(
        meshA_transformed, meshB_transformed,
        std::back_inserter(cgal_intersected_tris),
        CGAL::parameters::all_default(),
        CGAL::parameters::all_default()
    );
    auto tEnd = std::chrono::high_resolution_clock::now();
    outTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    std::vector<int2> results;
    results.reserve(cgal_intersected_tris.size());
    for (const auto& pair : cgal_intersected_tris) {
        results.push_back(make_int2(static_cast<int>(pair.first), static_cast<int>(pair.second)));
    }
    return results;
}

std::vector<int2> TestSuite::runCGALParallel(
    const double3& rotA, const double3& transA,
    const double3& rotB, const double3& transB,
    double& outTimeMs) 
{
    std::vector<int2> results;
    if (!isCGALParallelSupported()) {
        outTimeMs = 0.0;
        return results;
    }

    Mesh meshA_transformed, meshB_transformed;
    Point3 centerA(app_.normCenterA.x, app_.normCenterA.y, app_.normCenterA.z);
    Point3 centerB(app_.normCenterB.x, app_.normCenterB.y, app_.normCenterB.z);

    transformCgalMesh(app_.meshA, meshA_transformed, centerA, rotA, transA);
    transformCgalMesh(app_.meshB, meshB_transformed, centerB, rotB, transB);

    std::vector<std::pair<size_t, size_t>> intersectedTris;
    outTimeMs = computeCGALParallel(meshA_transformed, meshB_transformed, intersectedTris);

    results.reserve(intersectedTris.size());
    for (const auto& pair : intersectedTris) {
        results.push_back(make_int2(static_cast<int>(pair.first), static_cast<int>(pair.second)));
    }
    return results;
}

std::vector<int2> TestSuite::runMainGPUPipeline(
    const double3& rotA, const double3& transA,
    const double3& rotB, const double3& transB,
    const TestConfig& config,
    double& outTimeMs,
    size_t& outCandidates) 
{
    if (!app_.controller.isGPUAllocated()) {
        app_.controller.reconstructGPU(app_.stats);
    }

    float tGPU, tCPU, tTV, tAT, tGB;
    app_.controller.setTransformBoth(rotA, transA, rotB, transB, tGPU, tCPU, tTV, tAT, tGB);

    int2* outPairs = nullptr;
    size_t outCount = 0;

    auto tStart = std::chrono::high_resolution_clock::now();
    app_.controller.runIntersectionPipeline(
        config.batchMultiplier, config.dualTreeSteps, 0, outPairs, outCount, 
        app_.stats, config.enableGpuPrecision
    );
    auto tEnd = std::chrono::high_resolution_clock::now();
    outTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    outCandidates = app_.stats.finalAabbCandidatePairs;

    std::vector<int2> results;
    if (outPairs && outCount > 0) {
        results.assign(outPairs, outPairs + outCount);
        std::free(outPairs);
    }
    return results;
}

std::vector<int2> TestSuite::runStandalonePipeline(
    const double3& rotA, const double3& transA,
    const double3& rotB, const double3& transB,
    const TestConfig& config,
    double& outTimeMs,
    size_t& outCandidates) 
{
    Point3 centerA(app_.normCenterA.x, app_.normCenterA.y, app_.normCenterA.z);
    Point3 centerB(app_.normCenterB.x, app_.normCenterB.y, app_.normCenterB.z);

    ExecutionStats testStats;
    int2* outPairs = nullptr;
    size_t outCount = 0;

    auto tStart = std::chrono::high_resolution_clock::now();
    runGridForestIntersectionPipeline(
        app_.meshA, app_.meshB, app_.hVertsA.data(), static_cast<int>(app_.hVertsA.size()), 
        app_.hIndicesA.data(), static_cast<int>(app_.hIndicesA.size()), config.queryDescentLevel, 
        app_.hVertsB.data(), static_cast<int>(app_.hVertsB.size()), app_.hIndicesB.data(), 
        static_cast<int>(app_.hIndicesB.size()), config.referenceDescentLevel, config.batchMultiplier, 
        config.dualTreeSteps, config.enableGpuPrecision, config.leafThreshold, testStats, 
        outPairs, outCount, centerA, centerB, rotA, transA, rotB, transB
    );
    auto tEnd = std::chrono::high_resolution_clock::now();
    outTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    outCandidates = testStats.finalAabbCandidatePairs;

    std::vector<int2> results;
    if (outPairs && outCount > 0) {
        results.assign(outPairs, outPairs + outCount);
        std::free(outPairs);
    }
    return results;
}

VerificationResult TestSuite::evaluateStep(
    int stepIdx,
    const double3& rotA, const double3& transA,
    const double3& rotB, const double3& transB,
    const TestConfig& config) 
{
    VerificationResult res;
    res.stepIndex = stepIdx;
    res.rotA = rotA; res.transA = transA;
    res.rotB = rotB; res.transB = transB;

    size_t facesA = num_faces(app_.meshA);
    size_t facesB = num_faces(app_.meshB);
    res.totalPossiblePairs = facesA * facesB;

    // 1. CGAL Classic (Ground Truth)
    auto gtPairs = runCGALClassicGroundTruth(rotA, transA, rotB, transB, res.cgalClassicTimeMs);
    auto gtSet = convertToCanonicalSet(gtPairs);
    res.groundTruthCount = gtSet.size();

    // Calculate Surface Overlap Ratio Omega
    double totalAreaA = computeTotalMeshArea(app_.meshA);
    double totalAreaB = computeTotalMeshArea(app_.meshB);
    res.surfaceOverlapRatio = computeIntersectedSurfaceArea(app_.meshA, app_.meshB, gtSet, totalAreaA, totalAreaB);

    // 2. CGAL Parallel
    if (config.testCgalParallel) {
        auto cgalPPairs = runCGALParallel(rotA, transA, rotB, transB, res.cgalParallelTimeMs);
        auto cgalPSet = convertToCanonicalSet(cgalPPairs);
        res.cgalParallelCount = cgalPSet.size();
        res.cgalParallelExactMatch = (gtSet == cgalPSet);
    }

    // 3. Main GPU Pipeline (Hybrid)
    if (config.testMainGpuPipeline) {
        auto mainPairs = runMainGPUPipeline(rotA, transA, rotB, transB, config, res.gpuMainTimeMs, res.candidatePairs);
        auto mainSet = convertToCanonicalSet(mainPairs);
        res.gpuMainCount = mainSet.size();
        res.gpuMainExactMatch = (gtSet == mainSet);

        for (const auto& pair : mainSet) {
            if (gtSet.find(pair) == gtSet.end()) res.mainFalsePositives++;
        }
        for (const auto& pair : gtSet) {
            if (mainSet.find(pair) == mainSet.end()) res.mainFalseNegatives++;
        }
    }

    // 4. Standalone Pipeline
    if (config.testStandalonePipeline) {
        size_t standaloneCandidates = 0;
        auto standalonePairs = runStandalonePipeline(rotA, transA, rotB, transB, config, res.standaloneTimeMs, standaloneCandidates);
        auto standaloneSet = convertToCanonicalSet(standalonePairs);
        res.standaloneCount = standaloneSet.size();
        res.standaloneExactMatch = (gtSet == standaloneSet);

        if (!config.testMainGpuPipeline) {
            res.candidatePairs = standaloneCandidates;
        }
    }

    // Compute Candidate Yield (N_cand / (M_A * M_B))
    if (res.totalPossiblePairs > 0) {
        res.candidateYield = static_cast<double>(res.candidatePairs) / static_cast<double>(res.totalPossiblePairs);
    }

    return res;
}

void TestSuite::exportToCSV(const std::string& filepath, const std::vector<VerificationResult>& results) {
    std::ofstream csv(filepath);
    if (!csv.is_open()) {
        std::cerr << "Error: Could not open CSV output file: " << filepath << "\n";
        return;
    }

    // CSV Header
    csv << "step,intersections,candidates,total_possible_pairs,candidate_yield,surface_overlap_omega,"
        << "hybrid_ms,hybrid_standalone_ms,cgal_classic_ms,cgal_parallel_ms,"
        << "hybrid_match,standalone_match,cgal_parallel_match\n";

    csv << std::setprecision(10);
    for (const auto& res : results) {
        csv << res.stepIndex << ","
            << res.groundTruthCount << ","
            << res.candidatePairs << ","
            << res.totalPossiblePairs << ","
            << res.candidateYield << ","
            << res.surfaceOverlapRatio << ","
            << res.gpuMainTimeMs << ","
            << res.standaloneTimeMs << ","
            << res.cgalClassicTimeMs << ","
            << res.cgalParallelTimeMs << ","
            << (res.gpuMainExactMatch ? 1 : 0) << ","
            << (res.standaloneExactMatch ? 1 : 0) << ","
            << (res.cgalParallelExactMatch ? 1 : 0) << "\n";
    }

    std::cout << "[TestSuite] Benchmark table saved to: " << filepath << "\n";
}

void TestSuite::runSuite(const TestConfig& config) {
    if (!app_.isLoaded) {
        std::cout << "Error: No meshes loaded. Run 'load' before invoking TestSuite.\n";
        return;
    }

    float3 baseRotA_f{0.0f, 0.0f, 0.0f}, baseTransA_f{0.0f, 0.0f, 0.0f};
    float3 baseRotB_f{0.0f, 0.0f, 0.0f}, baseTransB_f{0.0f, 0.0f, 0.0f};

    if (!PolyscopeBridge::getCurrentTransforms(baseRotA_f, baseTransA_f, baseRotB_f, baseTransB_f)) {
        std::cout << "[TestSuite] Warning: Could not fetch Polyscope transforms. Falling back to origin.\n";
    }

    double3 baseRotA   = make_double3(baseRotA_f.x, baseRotA_f.y, baseRotA_f.z);
    double3 baseTransA = make_double3(baseTransA_f.x, baseTransA_f.y, baseTransA_f.z);
    double3 baseRotB   = make_double3(baseRotB_f.x, baseRotB_f.y, baseRotB_f.z);
    double3 baseTransB = make_double3(baseTransB_f.x, baseTransB_f.y, baseTransB_f.z);

    std::cout << "\n========================================================================================\n";
    std::cout << "                   STARTING AUTOMATED PIPELINE BENCHMARK SWEEP                          \n";
    std::cout << "========================================================================================\n";
    std::cout << "  Steps: " << config.numSteps 
              << " | Trans Perturb: [-" << config.maxTranslation << ", +" << config.maxTranslation << "]"
              << " | Rot Perturb: [-" << config.maxRotationDeg << "°, +" << config.maxRotationDeg << "°]"
              << " | Seed: " << config.seed << "\n\n";

    std::vector<VerificationResult> allResults;
    allResults.reserve(config.numSteps);

    std::mt19937 rng(config.seed);
    const double maxRotRad = config.maxRotationDeg * (M_PI / 180.0);

    std::uniform_real_distribution<double> rotDist(-maxRotRad, maxRotRad);
    std::uniform_real_distribution<double> transDist(-config.maxTranslation, config.maxTranslation);

    for (int i = 0; i < config.numSteps; ++i) {
        double3 rotA   = baseRotA;
        double3 transA = baseTransA;

        double3 rotB = make_double3(
            baseRotB.x + rotDist(rng),
            baseRotB.y + rotDist(rng),
            baseRotB.z + rotDist(rng)
        );
        double3 transB = make_double3(
            baseTransB.x + transDist(rng),
            baseTransB.y + transDist(rng),
            baseTransB.z + transDist(rng)
        );

        VerificationResult res = evaluateStep(i, rotA, transA, rotB, transB, config);
        allResults.push_back(res);

        std::cout << "test " << i << ": "
          << "Tris=" << std::setw(4) << res.groundTruthCount << " | "
          << "Omega=" << std::fixed << std::setprecision(2) << (res.surfaceOverlapRatio * 100.0) << "% | "
          << "N_cand=" << std::setw(6) << res.candidatePairs << " | "
          << "Yield=" << std::scientific << std::setprecision(2) << res.candidateYield << " | "
          << std::fixed << std::setprecision(2)
          << "Hybrid=" << res.gpuMainTimeMs << "ms | "
          << "Standalone=" << res.standaloneTimeMs << "ms | "
          << "CGAL_Classic=" << res.cgalClassicTimeMs << "ms | "
          << "CGAL_Parallel=" << res.cgalParallelTimeMs << "ms\n";
    }

    exportToCSV(config.csvOutputPath, allResults);
    std::cout << "========================================================================================\n\n";
}