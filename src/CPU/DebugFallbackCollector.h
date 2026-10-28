#pragma once
#include <vector>
#include <vector_types.h>
#include <mutex>

namespace DebugFallbackCollector {
    inline std::vector<int2> g_capturedFallbackPairs;
    inline std::mutex g_collectorMutex;

    inline void clear() {
        std::lock_guard<std::mutex> lock(g_collectorMutex);
        g_capturedFallbackPairs.clear();
    }

    inline void addPairs(const int2* pairs, size_t count) {
        if (count == 0 || !pairs) return;
        std::lock_guard<std::mutex> lock(g_collectorMutex);
        g_capturedFallbackPairs.insert(g_capturedFallbackPairs.end(), pairs, pairs + count);
    }

    inline std::vector<int2> getCaptured() {
        std::lock_guard<std::mutex> lock(g_collectorMutex);
        return g_capturedFallbackPairs;
    }
}