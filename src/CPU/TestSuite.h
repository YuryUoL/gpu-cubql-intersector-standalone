#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include <vector>
#include <set>
#include <string>
#include <cuda_runtime.h>

#include "CPU/CgalDefinitions.h"

class KernelBVHController;
struct ApplicationState;

struct TestConfig {
    int numSteps = 8;
    double maxTranslation = 1.0;
    double maxRotationDeg = 180.0;
    unsigned int seed = 1337;
    bool testMainGpuPipeline = true;
    bool testStandalonePipeline = true;
    bool testCgalParallel = true;
    int enableGpuPrecision = 1;
    int queryDescentLevel = 12;
    int referenceDescentLevel = 12;
    int batchMultiplier = 2147483647;
    int dualTreeSteps = 8;
    int leafThreshold = 20;
    std::string csvOutputPath = "benchmark_results.csv";
};

struct VerificationResult {
    int stepIndex = 0;
    double3 rotA{0,0,0}, transA{0,0,0};
    double3 rotB{0,0,0}, transB{0,0,0};

    size_t groundTruthCount = 0;
    size_t gpuMainCount = 0;
    size_t standaloneCount = 0;
    size_t cgalParallelCount = 0;

    bool gpuMainExactMatch = false;
    bool standaloneExactMatch = false;
    bool cgalParallelExactMatch = false;

    size_t mainFalsePositives = 0;
    size_t mainFalseNegatives = 0;
    size_t standaloneFalsePositives = 0;
    size_t standaloneFalseNegatives = 0;

    // Execution Times (ms)
    double cgalClassicTimeMs = 0.0;
    double cgalParallelTimeMs = 0.0;
    double gpuMainTimeMs = 0.0;
    double standaloneTimeMs = 0.0;

    // Candidate Yield & Overlap Metrics
    size_t totalPossiblePairs = 0; // M_A * M_B
    size_t candidatePairs = 0;     // N_cand
    double candidateYield = 0.0;   // N_cand / (M_A * M_B)
    double surfaceOverlapRatio = 0.0; // Omega = (S_A_int + S_B_int) / (S_A + S_B)
};

class TestSuite {
public:
    explicit TestSuite(ApplicationState& appState);

    void runSuite(const TestConfig& config);

    VerificationResult evaluateStep(
        int stepIdx,
        const double3& rotA, const double3& transA,
        const double3& rotB, const double3& transB,
        const TestConfig& config
    );

private:
    ApplicationState& app_;


    double computeTotalMeshArea(const Mesh& mesh);
    
    double computeIntersectedSurfaceArea(
        const Mesh& meshA, 
        const Mesh& meshB, 
        const std::set<std::pair<int, int>>& intersectedPairs,
        double totalAreaA,
        double totalAreaB
    );
    
    std::set<std::pair<int, int>> convertToCanonicalSet(const std::vector<int2>& pairs);

    std::vector<int2> runCGALClassicGroundTruth(
        const double3& rotA, const double3& transA,
        const double3& rotB, const double3& transB,
        double& outTimeMs
    );

    std::vector<int2> runCGALParallel(
        const double3& rotA, const double3& transA,
        const double3& rotB, const double3& transB,
        double& outTimeMs
    );

    std::vector<int2> runMainGPUPipeline(
        const double3& rotA, const double3& transA,
        const double3& rotB, const double3& transB,
        const TestConfig& config,
        double& outTimeMs,
        size_t& outCandidates
    );

    std::vector<int2> runStandalonePipeline(
        const double3& rotA, const double3& transA,
        const double3& rotB, const double3& transB,
        const TestConfig& config,
        double& outTimeMs,
        size_t& outCandidates
    );

    void exportToCSV(const std::string& filepath, const std::vector<VerificationResult>& results);
};

#endif // TEST_SUITE_H