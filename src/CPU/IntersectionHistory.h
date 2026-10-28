#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>
#include <ctime>
#include <vector_types.h>

// Structure holding single intersecting pair face IDs, vertex indices, and full-precision coordinates
struct IntersectingTriangleRecord {
    int faceA;
    int faceB;
    
    // Vertex Indices from Mesh A and Mesh B
    int vA0_idx, vA1_idx, vA2_idx;
    int vB0_idx, vB1_idx, vB2_idx;

    // Transformed coordinates for Triangle A (v0, v1, v2)
    double3 a_v0, a_v1, a_v2;
    // Transformed coordinates for Triangle B (v0, v1, v2)
    double3 b_v0, b_v1, b_v2;

    // Standard ordering comparator for std::set operations
    bool operator<(const IntersectingTriangleRecord& other) const {
        if (faceA != other.faceA) return faceA < other.faceA;
        return faceB < other.faceB;
    }
};

struct HistoryEntry {
    size_t id;
    std::string method;         // "standaloneBVH", "hybrid", "cgalparallel", "cgalclassic"
    std::string timestamp;      // YYYY-MM-DD HH:MM:SS
    double executionTimeMs;
    size_t intersectionCount;
    std::vector<IntersectingTriangleRecord> records;
};

class IntersectionHistory {
private:
    static constexpr size_t MAX_HISTORY = 5;
    std::vector<HistoryEntry> m_entries;
    size_t m_nextId = 0;

    static std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
#if defined(_WIN32)
        localtime_s(&tm_buf, &now_time);
#else
        localtime_r(&now_time, &tm_buf);
#endif
        std::ostringstream ss;
        ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

public:
    void recordRun(const std::string& method, double elapsedMs, 
                   const std::vector<IntersectingTriangleRecord>& records) {
        HistoryEntry entry;
        entry.id = m_nextId++;
        entry.method = method;
        entry.timestamp = getCurrentTimestamp();
        entry.executionTimeMs = elapsedMs;
        entry.intersectionCount = records.size();
        entry.records = records;

        if (m_entries.size() >= MAX_HISTORY) {
            m_entries.erase(m_entries.begin()); // Maintain last 5 runs
        }
        m_entries.push_back(entry);

        std::cout << "[History] Recorded Entry " << entry.id << " (" << method 
                  << ") with " << records.size() << " intersections.\n";
    }

    void printOverview() const {
        std::cout << "\n========================================================================================\n";
        std::cout << "                                  INTERSECTION HISTORY (LAST 5)                         \n";
        std::cout << "========================================================================================\n";
        std::cout << std::left << std::setw(8)  << "Entry ID" 
                  << std::setw(18) << "Method" 
                  << std::setw(24) << "Timestamp" 
                  << std::setw(16) << "Time (ms)" 
                  << std::setw(16) << "Intersections" << "\n";
        std::cout << "----------------------------------------------------------------------------------------\n";
        
        if (m_entries.empty()) {
            std::cout << "  [No history entries recorded yet]\n";
        } else {
            for (const auto& e : m_entries) {
                std::cout << std::left << std::setw(8)  << e.id
                          << std::setw(18) << e.method
                          << std::setw(24) << e.timestamp
                          << std::setw(16) << std::fixed << std::setprecision(2) << e.executionTimeMs
                          << std::setw(16) << e.intersectionCount << "\n";
            }
        }
        std::cout << "========================================================================================\n\n";
    }

    const HistoryEntry* getEntryById(size_t id) const {
        for (const auto& e : m_entries) {
            if (e.id == id) return &e;
        }
        return nullptr;
    }

    // NEW METHOD: Exports a full run/entry to CSV including vertex indices and full-precision coordinates
    bool exportRunToCsv(size_t id, const std::string& csvFilePath) const {
        const HistoryEntry* entry = getEntryById(id);
        if (!entry) {
            std::cerr << "Error: Entry ID " << id << " not found in history ring buffer.\n";
            return false;
        }

        std::ofstream csv(csvFilePath);
        if (!csv.is_open()) {
            std::cerr << "Error: Could not open file for writing CSV: " << csvFilePath << "\n";
            return false;
        }

        // Full precision formatting (17 digits)
        csv << std::setprecision(17);

        // Header
        csv << "FaceA,FaceB,"
            << "VA0_Idx,VA1_Idx,VA2_Idx,"
            << "VB0_Idx,VB1_Idx,VB2_Idx,"
            << "A_V0_X,A_V0_Y,A_V0_Z,A_V1_X,A_V1_Y,A_V1_Z,A_V2_X,A_V2_Y,A_V2_Z,"
            << "B_V0_X,B_V0_Y,B_V0_Z,B_V1_X,B_V1_Y,B_V1_Z,B_V2_X,B_V2_Y,B_V2_Z\n";

        for (const auto& r : entry->records) {
            csv << r.faceA << "," << r.faceB << ","
                << r.vA0_idx << "," << r.vA1_idx << "," << r.vA2_idx << ","
                << r.vB0_idx << "," << r.vB1_idx << "," << r.vB2_idx << ","
                << r.a_v0.x << "," << r.a_v0.y << "," << r.a_v0.z << ","
                << r.a_v1.x << "," << r.a_v1.y << "," << r.a_v1.z << ","
                << r.a_v2.x << "," << r.a_v2.y << "," << r.a_v2.z << ","
                << r.b_v0.x << "," << r.b_v0.y << "," << r.b_v0.z << ","
                << r.b_v1.x << "," << r.b_v1.y << "," << r.b_v1.z << ","
                << r.b_v2.x << "," << r.b_v2.y << "," << r.b_v2.z << "\n";
        }

        csv.close();

        std::cout << "\n=======================================================\n";
        std::cout << "               EXPORT RUN TO CSV SUCCESS               \n";
        std::cout << "=======================================================\n";
        std::cout << "  Entry ID         : " << entry->id << " (" << entry->method << ")\n";
        std::cout << "  Total Records    : " << entry->records.size() << "\n";
        std::cout << "  Output File      : " << csvFilePath << "\n";
        std::cout << "=======================================================\n\n";

        return true;
    }

    bool exportDiff(size_t id1, size_t id2, const std::string& filter, const std::string& csvFilePath) const {
        const HistoryEntry* e1 = getEntryById(id1);
        const HistoryEntry* e2 = getEntryById(id2);

        if (!e1 || !e2) {
            std::cerr << "Error: One or both entry IDs (" << id1 << ", " << id2 << ") not found in history ring buffer.\n";
            return false;
        }

        std::set<IntersectingTriangleRecord> set1(e1->records.begin(), e1->records.end());
        std::set<IntersectingTriangleRecord> set2(e2->records.begin(), e2->records.end());

        std::vector<IntersectingTriangleRecord> in1Not2, in2Not1, inBoth;

        for (const auto& rec : set1) {
            if (set2.count(rec)) {
                inBoth.push_back(rec);
            } else {
                in1Not2.push_back(rec);
            }
        }
        for (const auto& rec : set2) {
            if (!set1.count(rec)) {
                in2Not1.push_back(rec);
            }
        }

        std::ofstream csv(csvFilePath);
        if (!csv.is_open()) {
            std::cerr << "Error: Could not open file for writing CSV diff: " << csvFilePath << "\n";
            return false;
        }

        csv << std::setprecision(17);
        csv << "DiffCategory,FaceA,FaceB,"
            << "VA0_Idx,VA1_Idx,VA2_Idx,"
            << "VB0_Idx,VB1_Idx,VB2_Idx,"
            << "A_V0_X,A_V0_Y,A_V0_Z,A_V1_X,A_V1_Y,A_V1_Z,A_V2_X,A_V2_Y,A_V2_Z,"
            << "B_V0_X,B_V0_Y,B_V0_Z,B_V1_X,B_V1_Y,B_V1_Z,B_V2_X,B_V2_Y,B_V2_Z\n";

        auto writeRow = [&](const std::string& category, const IntersectingTriangleRecord& r) {
            csv << category << "," << r.faceA << "," << r.faceB << ","
                << r.vA0_idx << "," << r.vA1_idx << "," << r.vA2_idx << ","
                << r.vB0_idx << "," << r.vB1_idx << "," << r.vB2_idx << ","
                << r.a_v0.x << "," << r.a_v0.y << "," << r.a_v0.z << ","
                << r.a_v1.x << "," << r.a_v1.y << "," << r.a_v1.z << ","
                << r.a_v2.x << "," << r.a_v2.y << "," << r.a_v2.z << ","
                << r.b_v0.x << "," << r.b_v0.y << "," << r.b_v0.z << ","
                << r.b_v1.x << "," << r.b_v1.y << "," << r.b_v1.z << ","
                << r.b_v2.x << "," << r.b_v2.y << "," << r.b_v2.z << "\n";
        };

        std::string f = filter;
        std::transform(f.begin(), f.end(), f.begin(), ::tolower);

        bool writeOnly1 = (f == "all" || f == "only1" || f == "1" || f == "exclusive1");
        bool writeOnly2 = (f == "all" || f == "only2" || f == "2" || f == "exclusive2");
        bool writeBoth  = (f == "all" || f == "both");

        if (writeOnly1) {
            for (const auto& r : in1Not2) writeRow("ONLY_IN_" + std::to_string(id1), r);
        }
        if (writeOnly2) {
            for (const auto& r : in2Not1) writeRow("ONLY_IN_" + std::to_string(id2), r);
        }
        if (writeBoth) {
            for (const auto& r : inBoth) writeRow("IN_BOTH", r);
        }

        csv.close();

        std::cout << "\n=======================================================\n";
        std::cout << "               INTERSECTION DIFF SUMMARY               \n";
        std::cout << "=======================================================\n";
        std::cout << "  Comparing Entry " << id1 << " (" << e1->method << ") vs Entry " << id2 << " (" << e2->method << ")\n";
        std::cout << "  In Entry " << id1 << " ONLY : " << in1Not2.size() << "\n";
        std::cout << "  In Entry " << id2 << " ONLY : " << in2Not1.size() << "\n";
        std::cout << "  In BOTH Entries : " << inBoth.size() << "\n";
        std::cout << "  Export Filter   : " << filter << "\n";
        std::cout << "  Exported CSV to : " << csvFilePath << "\n";
        std::cout << "=======================================================\n\n";

        return true;
    }
};