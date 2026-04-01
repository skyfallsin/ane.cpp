#include "weight_cache.h"
#include "model_loader.h"
#include <ane_lm/common.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace ane_lm {

static const char CACHE_MAGIC[8] = {'A','N','E','C','A','C','H','E'};

std::string WeightCache::cache_path(const std::string& model_dir) {
    return model_dir + "/weights.f16cache";
}

WeightCache::~WeightCache() {
    if (mmap_base_) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool WeightCache::build(const std::string& model_dir, ModelWeights* weights) {
    std::string path = cache_path(model_dir);
    Timer timer;

    // Collect all tensors from all shards
    struct TensorInfo {
        std::string name;
        int64_t numel;
    };
    std::vector<TensorInfo> tensors;

    // Iterate over all shards to collect tensor names
    for (int s = 0; s < weights->shard_count(); s++) {
        // We need to enumerate tensors — go through the shard files
    }

    // Use a different approach: try to find all known weight patterns
    // Instead, let's iterate the internal tensor_owner_ map indirectly by
    // checking every tensor in every shard
    // Actually, the simplest approach: enumerate from the shard files directly
    // ModelWeights exposes files() but not shard iteration... 
    // Let's open each file and enumerate tensors
    for (const auto& file : weights->files()) {
        auto* sf = SafeTensors::open(file);
        if (!sf) continue;
        for (int i = 0; i < sf->n_tensors(); i++) {
            const SFTensor& t = sf->tensor(i);
            if (t.dtype == SFDtype::BF16 || t.dtype == SFDtype::F16) {
                tensors.push_back({t.name, SafeTensors::numel(&t)});
            }
        }
        delete sf;
    }

    if (tensors.empty()) {
        fprintf(stderr, "[cache] No BF16/F16 tensors found\n");
        return false;
    }

    // Sort by name for deterministic output
    std::sort(tensors.begin(), tensors.end(),
              [](const TensorInfo& a, const TensorInfo& b) { return a.name < b.name; });

    // Calculate layout
    size_t page_size = getpagesize();
    uint32_t num_entries = (uint32_t)tensors.size();
    size_t header_size = 24;  // magic + version + num_entries + data_offset
    size_t directory_size = num_entries * CACHE_ENTRY_SIZE;
    size_t data_section_offset = (header_size + directory_size + page_size - 1) & ~(page_size - 1);

    // Calculate data offsets for each tensor (f16 = 2 bytes per element)
    size_t data_cursor = 0;
    std::vector<size_t> data_offsets(num_entries);
    for (uint32_t i = 0; i < num_entries; i++) {
        // Align each tensor to 16 bytes for SIMD access
        data_offsets[i] = (data_cursor + 15) & ~(size_t)15;
        data_cursor = data_offsets[i] + tensors[i].numel * 2;  // f16 = 2 bytes
    }
    // Pad total data to page boundary
    size_t data_size = (data_cursor + page_size - 1) & ~(page_size - 1);
    size_t total_size = data_section_offset + data_size;

    LOG("[cache] Building f16 cache: %u tensors, %.1f MB data, %.1f MB total\n",
        num_entries, data_size / 1e6, total_size / 1e6);

    // Create the file
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        fprintf(stderr, "[cache] Cannot create %s: %s\n", path.c_str(), strerror(errno));
        return false;
    }

    // Extend file to total size
    if (ftruncate(fd, (off_t)total_size) < 0) {
        fprintf(stderr, "[cache] ftruncate failed: %s\n", strerror(errno));
        ::close(fd);
        return false;
    }

    // mmap for writing
    void* base = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "[cache] mmap for write failed: %s\n", strerror(errno));
        ::close(fd);
        return false;
    }

    // Write header
    uint8_t* p = (uint8_t*)base;
    memcpy(p, CACHE_MAGIC, 8);
    *(uint32_t*)(p + 8) = CACHE_VERSION;
    *(uint32_t*)(p + 12) = num_entries;
    *(uint64_t*)(p + 16) = (uint64_t)data_section_offset;

    // Write directory entries
    uint8_t* dir = p + 24;
    for (uint32_t i = 0; i < num_entries; i++) {
        uint8_t* entry = dir + i * CACHE_ENTRY_SIZE;
        uint32_t name_len = (uint32_t)tensors[i].name.size();
        *(uint32_t*)entry = name_len;
        memset(entry + 4, 0, 252);
        memcpy(entry + 4, tensors[i].name.c_str(), std::min(name_len, (uint32_t)251));

        *(uint32_t*)(entry + 256) = (uint32_t)CacheDtype::F16;
        *(uint64_t*)(entry + 260) = (uint64_t)data_offsets[i];
        *(int64_t*)(entry + 268) = tensors[i].numel;
    }

    // Convert and write tensor data
    uint8_t* data_section = p + data_section_offset;
    int converted = 0;
    for (uint32_t i = 0; i < num_entries; i++) {
        uint16_t* dst = (uint16_t*)(data_section + data_offsets[i]);
        if (!weights->convert_bf16_to_f16_into(dst, tensors[i].name.c_str(), tensors[i].numel)) {
            fprintf(stderr, "[cache] Failed to convert %s\n", tensors[i].name.c_str());
            munmap(base, total_size);
            ::close(fd);
            unlink(path.c_str());
            return false;
        }
        converted++;
        if (converted % 50 == 0 || converted == (int)num_entries) {
            fprintf(stderr, "  Converting tensors: %d/%u\r", converted, num_entries);
        }
    }
    fprintf(stderr, "\n");

    // Sync to disk
    msync(base, total_size, MS_SYNC);
    munmap(base, total_size);
    ::close(fd);

    double elapsed = timer.elapsed_ms();
    LOG("[cache] Cache built: %s (%.1f MB, %.1f sec)\n",
        path.c_str(), total_size / 1e6, elapsed / 1000.0);

    return true;
}

WeightCache* WeightCache::open(const std::string& model_dir) {
    std::string path = cache_path(model_dir);

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    if (file_size < 24) {
        ::close(fd);
        return nullptr;
    }

    void* base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        return nullptr;
    }

    // Validate header
    uint8_t* p = (uint8_t*)base;
    if (memcmp(p, CACHE_MAGIC, 8) != 0) {
        munmap(base, file_size);
        ::close(fd);
        return nullptr;
    }

    uint32_t version = *(uint32_t*)(p + 8);
    if (version != CACHE_VERSION) {
        fprintf(stderr, "[cache] Version mismatch: got %u, expected %u\n", version, CACHE_VERSION);
        munmap(base, file_size);
        ::close(fd);
        return nullptr;
    }

    uint32_t num_entries = *(uint32_t*)(p + 12);
    uint64_t data_section_offset = *(uint64_t*)(p + 16);

    if (data_section_offset >= file_size) {
        munmap(base, file_size);
        ::close(fd);
        return nullptr;
    }

    auto* cache = new WeightCache();
    cache->fd_ = fd;
    cache->mmap_base_ = base;
    cache->mmap_size_ = file_size;
    cache->data_base_ = p + data_section_offset;
    cache->data_size_ = file_size - data_section_offset;

    // Parse directory
    uint8_t* dir = p + 24;
    for (uint32_t i = 0; i < num_entries; i++) {
        uint8_t* entry = dir + i * CACHE_ENTRY_SIZE;
        uint32_t name_len = *(uint32_t*)entry;
        if (name_len > 251) name_len = 251;

        CacheEntry ce;
        memset(ce.name, 0, sizeof(ce.name));
        memcpy(ce.name, entry + 4, name_len);
        ce.name[name_len] = '\0';

        ce.dtype = (CacheDtype)*(uint32_t*)(entry + 256);
        ce.data_offset = (size_t)*(uint64_t*)(entry + 260);
        ce.num_elements = *(int64_t*)(entry + 268);

        cache->entries_[ce.name] = ce;
    }

    // Advise the kernel for sequential access on the data section
    madvise(cache->data_base_, cache->data_size_, MADV_SEQUENTIAL);

    LOG("[cache] Opened f16 cache: %u tensors, %.1f MB, data at offset %llu\n",
        num_entries, file_size / 1e6, data_section_offset);

    return cache;
}

const uint16_t* WeightCache::get_f16(const char* name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) return nullptr;
    return (const uint16_t*)((uint8_t*)data_base_ + it->second.data_offset);
}

} // namespace ane_lm
