#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace ane_lm {

class ModelWeights;

// Pre-converted weight cache file format:
//
// Offset 0:  [8 bytes]   magic "ANECACHE"
// Offset 8:  [4 bytes]   version (1)
// Offset 12: [4 bytes]   num_entries
// Offset 16: [8 bytes]   data_section_offset (page-aligned)
// Offset 24: [entry_size * num_entries] tensor directory
//   Each entry:
//     [4 bytes]  name_len
//     [252 bytes] name (null-padded)
//     [4 bytes]  dtype (0=f16, 1=f32)
//     [8 bytes]  data_offset (relative to data_section_offset)
//     [8 bytes]  num_elements
// After directory: padding to page boundary
// data_section_offset: concatenated weight data (f16 or f32)
//
// The data section is page-aligned so the entire region can be
// wrapped by Metal's newBufferWithBytesNoCopy for zero-copy GPU access.

constexpr size_t CACHE_ENTRY_SIZE = 256 + 4 + 8 + 8;  // 276 bytes
constexpr uint32_t CACHE_VERSION = 1;

enum class CacheDtype : uint32_t { F16 = 0, F32 = 1 };

struct CacheEntry {
    char name[256];
    CacheDtype dtype;
    size_t data_offset;   // relative to data section start
    int64_t num_elements;
};

// Weight cache for zero-copy mmap loading.
// Creates an f16 cache file from safetensors on first run.
// On subsequent runs, mmap the cache and use pointers directly.
class WeightCache {
public:
    ~WeightCache();

    // Build cache from model weights. Converts all tensors to f16.
    // Writes to <model_dir>/weights.f16cache
    static bool build(const std::string& model_dir, ModelWeights* weights);

    // Open existing cache (mmap). Returns nullptr if cache doesn't exist or is invalid.
    static WeightCache* open(const std::string& model_dir);

    // Get a direct f16 pointer into the mmap'd cache. Returns nullptr if not found.
    const uint16_t* get_f16(const char* name) const;

    // Get the base pointer of the data section (page-aligned, suitable for Metal nocopy).
    void* data_base() const { return data_base_; }

    // Size of the data section in bytes.
    size_t data_size() const { return data_size_; }

    // Total mmap size
    size_t total_size() const { return mmap_size_; }

    int num_entries() const { return (int)entries_.size(); }

    // Cache file path for a model directory.
    static std::string cache_path(const std::string& model_dir);

private:
    int fd_ = -1;
    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    void* data_base_ = nullptr;
    size_t data_size_ = 0;

    std::unordered_map<std::string, CacheEntry> entries_;
};

} // namespace ane_lm
