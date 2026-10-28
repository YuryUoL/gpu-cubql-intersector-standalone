#include "TestSuite.h"
#include "CPU/RotationTools.h"
#include "CPU/PolyscopeBridge.h"
#include "GPUIntersector/StandaloneBVHPipeline.h"
#include "CPU/CgalParallelLauncher.h"

#include "GPUIntersector/KernelBVHController.h"
#include "ApplicationState.h"

#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <random>

#include "RobustOffLoader.h"

TestSuite::TestSuite(ApplicationState &appState) : app_(appState) {}

std::set<std::pair<int, int>> TestSuite::convertToCanonicalSet(const std::vector<int2> &pairs)
{
    std::set<std::pair<int, int>> canonicalSet;
    for (const auto &p : pairs)
    {
        canonicalSet.insert({p.x, p.y});
    }
    return canonicalSet;
}

double TestSuite::computeTotalMeshArea(const Mesh &mesh)
{
    return CGAL::to_double(CGAL::Polygon_mesh_processing::area(mesh));
}

double TestSuite::computeIntersectedSurfaceArea(
    const Mesh &meshA,
    const Mesh &meshB,
    const std::set<std::pair<int, int>> &intersectedPairs,
    double totalAreaA,
    double totalAreaB)
{
    if (intersectedPairs.empty() || (totalAreaA + totalAreaB) <= 0.0)
    {
        return 0.0;
    }

    std::unordered_set<int> intersectedFacesA;
    std::unordered_set<int> intersectedFacesB;

    for (const auto &pair : intersectedPairs)
    {
        intersectedFacesA.insert(pair.first);
        intersectedFacesB.insert(pair.second);
    }

    double areaA_int = 0.0;
    auto facesA = faces(meshA);
    for (int faceIdx : intersectedFacesA)
    {
        auto faceIter = std::next(facesA.first, faceIdx);
        areaA_int += CGAL::to_double(CGAL::Polygon_mesh_processing::face_area(*faceIter, meshA));
    }

    double areaB_int = 0.0;
    auto facesB = faces(meshB);
    for (int faceIdx : intersectedFacesB)
    {
        auto faceIter = std::next(facesB.first, faceIdx);
        areaB_int += CGAL::to_double(CGAL::Polygon_mesh_processing::face_area(*faceIter, meshB));
    }

    return (areaA_int + areaB_int) / (totalAreaA + totalAreaB);
}

std::vector<int2> TestSuite::runCGALClassicGroundTruth(
    const Mesh &meshA_transformed, const Mesh &meshB_transformed,
    double &outTimeMs)
{
    using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
    std::vector<std::pair<face_descriptor, face_descriptor>> cgal_intersected_tris;

    auto tStart = std::chrono::high_resolution_clock::now();
    CGAL::Polygon_mesh_processing::internal::compute_face_face_intersection(
        meshA_transformed, meshB_transformed,
        std::back_inserter(cgal_intersected_tris),
        CGAL::parameters::all_default(),
        CGAL::parameters::all_default());
    auto tEnd = std::chrono::high_resolution_clock::now();
    outTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    std::vector<int2> results;
    results.reserve(cgal_intersected_tris.size());
    for (const auto &pair : cgal_intersected_tris)
    {
        results.push_back(make_int2(static_cast<int>(pair.first), static_cast<int>(pair.second)));
    }
    return results;
}

std::vector<int2> TestSuite::runCGALParallel(
    const Mesh &meshA_transformed, const Mesh &meshB_transformed,
    double &outTimeMs)
{
    std::vector<int2> results;
    if (!isCGALParallelSupported())
    {
        outTimeMs = 0.0;
        return results;
    }

    std::vector<std::pair<size_t, size_t>> intersectedTris;
    outTimeMs = computeCGALParallel(meshA_transformed, meshB_transformed, intersectedTris);

    results.reserve(intersectedTris.size());
    for (const auto &pair : intersectedTris)
    {
        results.push_back(make_int2(static_cast<int>(pair.first), static_cast<int>(pair.second)));
    }
    return results;
}

std::vector<int2> TestSuite::runMainGPUPipeline(
    const double3 &rotA, const double3 &transA,
    const double3 &rotB, const double3 &transB,
    const TestConfig &config,
    double &outTimeMs,
    size_t &outCandidates,
    ExecutionStats &outStats)
{
    if (!app_.controller.isGPUAllocated())
    {
        app_.controller.reconstructGPU(app_.stats);
    }

    auto tStart = std::chrono::high_resolution_clock::now();
    float tGPU, tCPU, tTV, tAT, tGB;
    app_.controller.setTransformBoth(rotA, transA, rotB, transB, tGPU, tCPU, tTV, tAT, tGB);

    int2 *outPairs = nullptr;
    size_t outCount = 0;

    app_.controller.runIntersectionPipeline(
        config.batchMultiplier, config.dualTreeSteps, 0, outPairs, outCount,
        app_.stats, config.enableGpuPrecision);
    auto tEnd = std::chrono::high_resolution_clock::now();
    outTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    outCandidates = app_.stats.finalAabbCandidatePairs;
    outStats = app_.stats;

    std::vector<int2> results;
    if (outPairs && outCount > 0)
    {
        results.assign(outPairs, outPairs + outCount);
        std::free(outPairs);
    }
    return results;
}

void extractVertexCoordinatesParallelLocal(const Mesh &mesh, std::vector<double3> &outVerts)
{
    size_t numVerts = num_vertices(mesh);
    outVerts.resize(numVerts);
    auto coords = mesh.points();

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numVerts), [&](const tbb::blocked_range<size_t> &r)
                      {
    for(size_t i = r.begin(); i != r.end(); ++i) {
      typename Mesh::Vertex_index v(static_cast<Mesh::size_type>(i));
      const auto& p = coords[v];
      outVerts[i] = make_double3(
          CGAL::to_double(p.x()), 
          CGAL::to_double(p.y()), 
          CGAL::to_double(p.z())
      );
    } });
}
std::vector<int2> TestSuite::runStandalonePipeline(
    Mesh &meshA_transformed, // Removed const here
    Mesh &meshB_transformed, // Removed const here
    const TestConfig &config,
    double &outTimeMs,
    size_t &outCandidates,
    ExecutionStats &outStats)
{
    Point3 centerA(app_.normCenterA.x, app_.normCenterA.y, app_.normCenterA.z);
    Point3 centerB(app_.normCenterB.x, app_.normCenterB.y, app_.normCenterB.z);

    // Extract vertex coordinates from the pre-transformed meshes
    std::vector<double3> hVertsA_transformed, hVertsB_transformed;
    std::vector<uint3> hIndicesA_transformed, hIndicesB_transformed;

    if (!RobustIO::deriveBuffersFromMesh(meshA_transformed, hVertsA_transformed, hIndicesA_transformed) ||
        !RobustIO::deriveBuffersFromMesh(meshB_transformed, hVertsB_transformed, hIndicesB_transformed))
    {
        std::cout << "Error: transformed mesh has removed elements (garbage); cannot build dense GPU buffers.\n";
       // return;
    }

   

    ExecutionStats testStats;
    int2 *outPairs = nullptr;
    size_t outCount = 0;

    auto tStart = std::chrono::high_resolution_clock::now();

    runGridForestIntersectionPipeline(
      meshA_transformed, meshB_transformed,
      hVertsA_transformed.data(), static_cast<int>(hVertsA_transformed.size()),
      hIndicesA_transformed.data(), static_cast<int>(hIndicesA_transformed.size()), config.queryDescentLevel,
      hVertsB_transformed.data(), static_cast<int>(hVertsB_transformed.size()),
      hIndicesB_transformed.data(), static_cast<int>(hIndicesB_transformed.size()), config.referenceDescentLevel,
      config.batchMultiplier, config.dualTreeSteps, config.enableGpuPrecision, config.leafThreshold, testStats,
        outPairs, outCount, centerA, centerB, 
        make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0)); // RotB, TransB as 0s

    auto tEnd = std::chrono::high_resolution_clock::now();
    outTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    outCandidates = testStats.finalAabbCandidatePairs;
    outStats = testStats;

    std::vector<int2> results;
    if (outPairs && outCount > 0)
    {
        results.assign(outPairs, outPairs + outCount);
        std::free(outPairs);
    }
    return results;
}
VerificationResult TestSuite::evaluateStep(
    int stepIdx,
    const double3 &rotA, const double3 &transA,
    const double3 &rotB, const double3 &transB,
    const TestConfig &config)
{
    VerificationResult res;
    res.stepIndex = stepIdx;
    res.rotA = rotA;
    res.transA = transA;
    res.rotB = rotB;
    res.transB = transB;

    size_t facesA = num_faces(app_.meshA);
    size_t facesB = num_faces(app_.meshB);
    res.totalPossiblePairs = facesA * facesB;

    // 1. Transform CGAL meshes ONCE for all CPU-based pipelines in this step
    Mesh meshA_transformed, meshB_transformed;
    Point3 centerA(app_.normCenterA.x, app_.normCenterA.y, app_.normCenterA.z);
    Point3 centerB(app_.normCenterB.x, app_.normCenterB.y, app_.normCenterB.z);

    transformCgalMesh(app_.meshA, meshA_transformed, centerA, rotA, transA);
    transformCgalMesh(app_.meshB, meshB_transformed, centerB, rotB, transB);

    // 2. CGAL Classic (Ground Truth)
    auto gtPairs = runCGALClassicGroundTruth(meshA_transformed, meshB_transformed, res.cgalClassicTimeMs);
    auto gtSet = convertToCanonicalSet(gtPairs);
    res.groundTruthCount = gtSet.size();

    // Calculate Surface Overlap Ratio Omega
    double totalAreaA = computeTotalMeshArea(app_.meshA);
    double totalAreaB = computeTotalMeshArea(app_.meshB);
    res.surfaceOverlapRatio = computeIntersectedSurfaceArea(app_.meshA, app_.meshB, gtSet, totalAreaA, totalAreaB);

    // 3. CGAL Parallel
    if (config.testCgalParallel)
    {
        auto cgalPPairs = runCGALParallel(meshA_transformed, meshB_transformed, res.cgalParallelTimeMs);
        auto cgalPSet = convertToCanonicalSet(cgalPPairs);
        res.cgalParallelCount = cgalPSet.size();
        res.cgalParallelExactMatch = (gtSet == cgalPSet);
    }

    // 4. Main GPU Pipeline (Standard Compute - uses its own controller transform mechanism)
    if (config.testMainGpuPipeline)
    {
        auto mainPairs = runMainGPUPipeline(rotA, transA, rotB, transB, config, res.gpuMainTimeMs, res.candidatePairs, res.mainStats);
        auto mainSet = convertToCanonicalSet(mainPairs);
        res.gpuMainCount = mainSet.size();
        res.gpuMainExactMatch = (gtSet == mainSet);

        for (const auto &pair : mainSet)
        {
            if (gtSet.find(pair) == gtSet.end())
                res.mainFalsePositives++;
        }
        for (const auto &pair : gtSet)
        {
            if (mainSet.find(pair) == mainSet.end())
                res.mainFalseNegatives++;
        }
    }

    // 5. Standalone Pipeline
    if (config.testStandalonePipeline)
    {
        size_t standaloneCandidates = 0;
        auto standalonePairs = runStandalonePipeline(meshA_transformed, meshB_transformed, config, res.standaloneTimeMs, standaloneCandidates, res.standaloneStats);
        auto standaloneSet = convertToCanonicalSet(standalonePairs);
        res.standaloneCount = standaloneSet.size();
        res.standaloneExactMatch = (gtSet == standaloneSet);

        if (!config.testMainGpuPipeline)
        {
            res.candidatePairs = standaloneCandidates;
        }
    }

    // Compute Candidate Yield (N_cand / (M_A * M_B))
    if (res.totalPossiblePairs > 0)
    {
        res.candidateYield = static_cast<double>(res.candidatePairs) / static_cast<double>(res.totalPossiblePairs);
    }

    return res;
}
// 1. Summary CSV (Overall Benchmarks)
void TestSuite::exportToCSV(const std::string &filepath, const std::vector<VerificationResult> &results)
{
    std::ofstream csv(filepath);
    if (!csv.is_open())
    {
        std::cerr << "Error: Could not open CSV output file: " << filepath << "\n";
        return;
    }

    csv << "step,intersections,candidates,total_possible_pairs,candidate_yield,surface_overlap_omega,"
        << "hybrid_ms,hybrid_standalone_ms,cgal_classic_ms,cgal_parallel_ms,"
        << "hybrid_match,standalone_match,cgal_parallel_match\n";

    csv << std::setprecision(10);
    for (const auto &res : results)
    {
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

    std::cout << "[TestSuite] Summary benchmark table saved to: " << filepath << "\n";
}

// 2. Standard Compute Breakdown CSV
void TestSuite::exportStandardComputeCSV(const std::string &filepath, const std::vector<VerificationResult> &results)
{
    std::ofstream csv(filepath);
    if (!csv.is_open())
    {
        std::cerr << "Error: Could not open Standard Compute CSV file: " << filepath << "\n";
        return;
    }

    csv << "step,gpu_total_ms,cross_check_engine_ms,dual_tree_step_ms,parallel_dfs_descent_ms,"
        << "preallocate_ms,assembly_phase_ms,aabb_candidate_exec_ms,gpu_float_predicates_ms,"
        << "gpu_double_predicates_ms,thrust_compaction_d2h_ms,cpu_narrow_phase_ms,"
        << "loop_workspace_cleanup_ms,explicit_cleanup_sync_ms,"
        << "confirmed_green_pairs,confirmed_yellow_pairs,confirmed_orange_pairs\n";

    csv << std::setprecision(10);
    for (const auto &res : results)
    {
        const auto &s = res.mainStats;
        csv << res.stepIndex << ","
            << s.GPUTotalTime << ","
            << s.gpuCrossCheckEngineMs << ","
            << s.dualTreeStepMs << ","
            << s.parallelDfsDescentBMs << ","
            << s.loopTracker.preallocateTimeMs << ","
            << s.loopTracker.assemblyPhaseMs << ","
            << s.loopTracker.executionPhaseMs << ","
            << s.loopTracker.fineEvaluationPhaseMs << ","
            << s.loopTracker.gpuDoublePredicatesMs << ","
            << s.loopTracker.DownloadAndClean << ","
            << s.loopTracker.CPUPredicates << ","
            << s.loopTracker.cleanupTimeMs << ","
            << s.finalCleanupSyncMs << ","
            << s.loopTracker.confirmedGreenPairs << ","
            << s.loopTracker.confirmedYellowPairs << ","
            << s.loopTracker.confirmedOrangePairs << "\n";
    }

    std::cout << "[TestSuite] Standard Compute breakdown saved to: " << filepath << "\n";
}

// 3. Standalone Compute Breakdown CSV
void TestSuite::exportStandaloneComputeCSV(const std::string &filepath, const std::vector<VerificationResult> &results)
{
    std::ofstream csv(filepath);
    if (!csv.is_open())
    {
        std::cerr << "Error: Could not open Standalone Compute CSV file: " << filepath << "\n";
        return;
    }

    csv << "step,gpu_total_ms,raw_alloc_copy_ms,thrust_init_overhead_ms,build_refit_mesh_a_ms,build_refit_mesh_b_ms,"
        << "cross_check_engine_ms,dual_tree_step_ms,parallel_dfs_descent_ms,preallocate_ms,"
        << "assembly_phase_ms,aabb_candidate_exec_ms,gpu_float_predicates_ms,gpu_double_predicates_ms,"
        << "thrust_compaction_d2h_ms,cpu_narrow_phase_ms,loop_workspace_cleanup_ms,explicit_cleanup_sync_ms,"
        << "confirmed_green_pairs,confirmed_yellow_pairs,confirmed_orange_pairs\n";

    csv << std::setprecision(10);
    for (const auto &res : results)
    {
        const auto &s = res.standaloneStats;
        csv << res.stepIndex << ","
            << s.GPUTotalTime << ","
            << s.initialAllocAndCopyMs << ","
            << s.thrustInitOverheadMs << ","
            << s.buildRefitMeshAMs << ","
            << s.buildRefitMeshBMs << ","
            << s.gpuCrossCheckEngineMs << ","
            << s.dualTreeStepMs << ","
            << s.parallelDfsDescentBMs << ","
            << s.loopTracker.preallocateTimeMs << ","
            << s.loopTracker.assemblyPhaseMs << ","
            << s.loopTracker.executionPhaseMs << ","
            << s.loopTracker.fineEvaluationPhaseMs << ","
            << s.loopTracker.gpuDoublePredicatesMs << ","
            << s.loopTracker.DownloadAndClean << ","
            << s.loopTracker.CPUPredicates << ","
            << s.loopTracker.cleanupTimeMs << ","
            << s.finalCleanupSyncMs << ","
            << s.loopTracker.confirmedGreenPairs << ","
            << s.loopTracker.confirmedYellowPairs << ","
            << s.loopTracker.confirmedOrangePairs << "\n";
    }

    std::cout << "[TestSuite] Standalone Compute breakdown saved to: " << filepath << "\n";
}

static std::string deriveFilename(const std::string &basePath, const std::string &suffix)
{
    size_t dotPos = basePath.find_last_of('.');
    if (dotPos == std::string::npos)
    {
        return basePath + suffix + ".csv";
    }
    return basePath.substr(0, dotPos) + suffix + ".csv";
}

void TestSuite::runSuite(const TestConfig &config)
{
    if (!app_.isLoaded)
    {
        std::cout << "Error: No meshes loaded. Run 'load' before invoking TestSuite.\n";
        return;
    }

    float3 baseRotA_f{0.0f, 0.0f, 0.0f}, baseTransA_f{0.0f, 0.0f, 0.0f};
    float3 baseRotB_f{0.0f, 0.0f, 0.0f}, baseTransB_f{0.0f, 0.0f, 0.0f};

    if (!PolyscopeBridge::getCurrentTransforms(baseRotA_f, baseTransA_f, baseRotB_f, baseTransB_f))
    {
        std::cout << "[TestSuite] Warning: Could not fetch Polyscope transforms. Falling back to origin.\n";
    }

    double3 baseRotA = make_double3(baseRotA_f.x, baseRotA_f.y, baseRotA_f.z);
    double3 baseTransA = make_double3(baseTransA_f.x, baseTransA_f.y, baseTransA_f.z);
    double3 baseRotB = make_double3(baseRotB_f.x, baseRotB_f.y, baseRotB_f.z);
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

    // FIX: Use maxRotationDeg directly since baseRot and input configs are in DEGREES
    std::uniform_real_distribution<double> rotDist(-config.maxRotationDeg, config.maxRotationDeg);
    std::uniform_real_distribution<double> transDist(-config.maxTranslation, config.maxTranslation);

    for (int i = 0; i < config.numSteps; ++i)
    {
        double3 rotA = baseRotA;
        double3 transA = baseTransA;

        double3 rotB = make_double3(
            baseRotB.x + rotDist(rng),
            baseRotB.y + rotDist(rng),
            baseRotB.z + rotDist(rng));
        double3 transB = make_double3(
            baseTransB.x + transDist(rng),
            baseTransB.y + transDist(rng),
            baseTransB.z + transDist(rng));

        VerificationResult res = evaluateStep(i, rotA, transA, rotB, transB, config);
        allResults.push_back(res);

        // Check if all active methods coincided on the exact intersection count
        bool countsCoincided = true;
        size_t baselineCount = res.groundTruthCount;
        if (config.testCgalParallel && res.cgalParallelCount != baselineCount)
            countsCoincided = false;
        if (config.testMainGpuPipeline && res.gpuMainCount != baselineCount)
            countsCoincided = false;
        if (config.testStandalonePipeline && res.standaloneCount != baselineCount)
            countsCoincided = false;

        // Print step header and full-precision transformation details (already in degrees)
        std::cout << "test " << std::setw(3) << i << ":\n";

        // Save current stream flags
        std::ios state(nullptr);
        state.copyfmt(std::cout);

        std::cout << std::defaultfloat << std::setprecision(16);
        std::cout << "  [Transform A] Rot(°)=(" << rotA.x << ", " << rotA.y << ", " << rotA.z << ") | "
                  << "Trans=(" << transA.x << ", " << transA.y << ", " << transA.z << ")\n";
        std::cout << "  [Transform B] Rot(°)=(" << rotB.x << ", " << rotB.y << ", " << rotB.z << ") | "
                  << "Trans=(" << transB.x << ", " << transB.y << ", " << transB.z << ")\n";

        // Restore stream formatting for results & metrics
        std::cout.copyfmt(state);

        // Print intersection counts and performance stats
        std::cout << "  [Results]     CGAL_Classic=" << std::setw(4) << res.groundTruthCount;

        if (config.testCgalParallel)
            std::cout << " | CGAL_Parallel=" << std::setw(4) << res.cgalParallelCount;
        if (config.testMainGpuPipeline)
            std::cout << " | Hybrid=" << std::setw(4) << res.gpuMainCount;
        if (config.testStandalonePipeline)
            std::cout << " | Standalone=" << std::setw(4) << res.standaloneCount;

        std::cout << " | " << (countsCoincided ? "[ALL COINCIDED]" : "[DISTINCTIONS FOUND]") << "\n"
                  << "  [Metrics]     Omega=" << std::fixed << std::setprecision(2) << (res.surfaceOverlapRatio * 100.0) << "% | "
                  << "Hybrid_t=" << res.gpuMainTimeMs << "ms | "
                  << "Standalone_t=" << res.standaloneTimeMs << "ms\n\n";
    }

    // Compute overall suite consensus summary
    int totalSteps = static_cast<int>(allResults.size());
    int fullyCoincidedSteps = 0;
    for (const auto &res : allResults)
    {
        bool coincided = true;
        size_t gt = res.groundTruthCount;
        if (config.testCgalParallel && res.cgalParallelCount != gt)
            coincided = false;
        if (config.testMainGpuPipeline && res.gpuMainCount != gt)
            coincided = false;
        if (config.testStandalonePipeline && res.standaloneCount != gt)
            coincided = false;
        if (coincided)
            fullyCoincidedSteps++;
    }

    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "                         INTERSECTION CONSENSUS SUMMARY                                 \n";
    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "  Total Steps Run          : " << totalSteps << "\n";
    std::cout << "  Steps with Full Consensus: " << fullyCoincidedSteps << " / " << totalSteps
              << " (" << std::fixed << std::setprecision(1) << (100.0 * fullyCoincidedSteps / totalSteps) << "%)\n";
    if (fullyCoincidedSteps == totalSteps)
    {
        std::cout << "  Status                   : ALL methods coincided on intersection counts across every step.\n";
    }
    else
    {
        std::cout << "  Status                   : DISTINCTIONS found in intersection counts across methods in "
                  << (totalSteps - fullyCoincidedSteps) << " step(s).\n";
    }
    std::cout << "----------------------------------------------------------------------------------------\n\n";

    // Export all 3 CSV files
    std::string summaryCsv = config.csvOutputPath;
    std::string standardCsv = deriveFilename(config.csvOutputPath, "_standard_breakdown");
    std::string standaloneCsv = deriveFilename(config.csvOutputPath, "_standalone_breakdown");

    exportToCSV(summaryCsv, allResults);
    exportStandardComputeCSV(standardCsv, allResults);
    exportStandaloneComputeCSV(standaloneCsv, allResults);

    std::cout << "========================================================================================\n\n";
}