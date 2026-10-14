#pragma once

#include "ApplicationState.h"
#include "common/ExecutionStats.h"
#include <string>
#include <vector>
#include <set>

struct TestConfig {
    int numSteps = 10;
    double maxTranslation = 0.5;
    double maxRotationDeg = 15.0;
    int enableGpuPrecision = 1;
    int dualTreeSteps = 0;
    int queryDescentLevel = 12;
    int referenceDescentLevel = 12;
    int batchMultiplier = 1;
    int leafThreshold = 4;
    unsigned int seed = 1337;
    std::string csvOutputPath = "benchmark_results.csv";

    bool testMainGpuPipeline = true;
    bool testStandalonePipeline = true;
    bool testCgalParallel = true;
};

struct VerificationResult {
    int stepIndex = 0;
    double3 rotA, transA, rotB, transB;
    size_t totalPossiblePairs = 0;
    size_t groundTruthCount = 0;
    size_t candidatePairs = 0;
    double candidateYield = 0.0;
    double surfaceOverlapRatio = 0.0;

    double cgalClassicTimeMs = 0.0;
    double cgalParallelTimeMs = 0.0;
    size_t cgalParallelCount = 0;
    bool cgalParallelExactMatch = false;

    double gpuMainTimeMs = 0.0;
    size_t gpuMainCount = 0;
    bool gpuMainExactMatch = false;
    size_t mainFalsePositives = 0;
    size_t mainFalseNegatives = 0;
    ExecutionStats mainStats; // Standard compute stats breakdown

    double standaloneTimeMs = 0.0;
    size_t standaloneCount = 0;
    bool standaloneExactMatch = false;
    ExecutionStats standaloneStats; // Standalone compute stats breakdown
};

class TestSuite {
public:
    explicit TestSuite(ApplicationState& appState);

    void runSuite(const TestConfig& config);

    void exportToCSV(const std::string& filepath, const std::vector<VerificationResult>& results);
    void exportStandardComputeCSV(const std::string& filepath, const std::vector<VerificationResult>& results);
    void exportStandaloneComputeCSV(const std::string& filepath, const std::vector<VerificationResult>& results);

private:
    ApplicationState& app_;

    VerificationResult evaluateStep(
        int stepIdx,
        const double3& rotA, const double3& transA,
        const double3& rotB, const double3& transB,
        const TestConfig& config);

    std::vector<int2> runCGALClassicGroundTruth(
        const Mesh& meshA_transformed,
        const Mesh& meshB_transformed,
        double& outTimeMs);

    std::vector<int2> runCGALParallel(
        const Mesh& meshA_transformed,
        const Mesh& meshB_transformed,
        double& outTimeMs);

    std::vector<int2> runMainGPUPipeline(
        const double3& rotA, const double3& transA,
        const double3& rotB, const double3& transB,
        const TestConfig& config,
        double& outTimeMs,
        size_t& outCandidates,
        ExecutionStats& outStats);

    std::vector<int2> runStandalonePipeline(
        Mesh& meshA_transformed, // Removed const
        Mesh& meshB_transformed, // Removed const
        const TestConfig& config,
        double& outTimeMs,
        size_t& outCandidates,
        ExecutionStats& outStats);

    std::set<std::pair<int, int>> convertToCanonicalSet(const std::vector<int2>& pairs);
    double computeTotalMeshArea(const Mesh& mesh);
    double computeIntersectedSurfaceArea(
        const Mesh& meshA, const Mesh& meshB, 
        const std::set<std::pair<int, int>>& intersectedPairs,
        double totalAreaA, double totalAreaB);
};