#include "safetensors.h"
#include <ane_lm/common.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace ane_lm {

static SFDtype parse_dtype(const char* s, int len) {
    if (len == 4 && memcmp(s, "BF16", 4) == 0) return SFDtype::BF16;
    if (len == 3 && memcmp(s, "F16", 3) == 0) return SFDtype::F16;
    if (len == 3 && memcmp(s, "F32", 3) == 0) return SFDtype::F32;
    if (len == 3 && memcmp(s, "F64", 3) == 0) return SFDtype::F64;
    if (len == 3 && memcmp(s, "I32", 3) == 0) return SFDtype::I32;
    if (len == 3 && memcmp(s, "I64", 3) == 0) return SFDtype::I64;
    if (len == 2 && memcmp(s, "U8", 2) == 0) return SFDtype::U8;
    return SFDtype::Unknown;
}

int SafeTensors::dtype_size(SFDtype d) {
    switch (d) {
        case SFDtype::BF16: case SFDtype::F16: return 2;
        case SFDtype::F32: case SFDtype::I32: return 4;
        case SFDtype::F64: case SFDtype::I64: return 8;
        case SFDtype::U8: return 1;
        default: return 0;
    }
}

// JSON helpers
static int64_t json_find(const char* json, int64_t len, int64_t pos, char c) {
    bool in_string = false;
    for (int64_t i = pos; i < len; i++) {
        if (json[i] == '"' && (i == 0 || json[i-1] != '\\')) in_string = !in_string;
        if (!in_string && json[i] == c) return i;
    }
    return -1;
}

static int json_string(const char* json, int64_t pos, char* out, int max_len) {
    if (json[pos] != '"') return -1;
    int len = 0;
    for (int64_t i = pos + 1; json[i] != '"' || json[i-1] == '\\'; i++) {
        if (len < max_len - 1) out[len++] = json[i];
    }
    out[len] = '\0';
    return len;
}

static int64_t json_string_end(const char* json, int64_t pos) {
    for (int64_t i = pos + 1; ; i++) {
        if (json[i] == '"' && json[i-1] != '\\') return i + 1;
    }
}

bool SafeTensors::parse_header(const char* json, int64_t json_len) {
    n_tensors_ = 0;
    int64_t pos = 0;

    pos = json_find(json, json_len, pos, '{');
    if (pos < 0) return false;
    pos++;

    while (pos < json_len && n_tensors_ < SF_MAX_TENSORS) {
        int64_t key_start = -1;
        for (int64_t i = pos; i < json_len; i++) {
            if (json[i] == '"') { key_start = i; break; }
            if (json[i] == '}') return true;
        }
        if (key_start < 0) break;

        char key[SF_MAX_NAME];
        int key_len = json_string(json, key_start, key, SF_MAX_NAME);
        if (key_len < 0) break;
        pos = json_string_end(json, key_start);

        if (strcmp(key, "__metadata__") == 0) {
            pos = json_find(json, json_len, pos, ':');
            if (pos < 0) break;
            pos++;
            int depth = 0;
            bool in_str = false;
            for (int64_t i = pos; i < json_len; i++) {
                if (json[i] == '"' && (i == 0 || json[i-1] != '\\')) in_str = !in_str;
                if (!in_str) {
                    if (json[i] == '{') depth++;
                    if (json[i] == '}') { depth--; if (depth == 0) { pos = i + 1; break; } }
                }
            }
            if (pos < json_len && json[pos] == ',') pos++;
            continue;
        }

        SFTensor* t = &tensors_[n_tensors_];
        strncpy(t->name, key, SF_MAX_NAME - 1);
        t->name[SF_MAX_NAME - 1] = '\0';

        pos = json_find(json, json_len, pos, ':');
        if (pos < 0) break;
        pos++;
        pos = json_find(json, json_len, pos, '{');
        if (pos < 0) break;
        int64_t obj_start = pos;
        pos++;

        t->ndims = 0;
        t->dtype = SFDtype::Unknown;
        t->data_offset = 0;
        t->data_size = 0;

        int depth = 1;
        int64_t obj_end = obj_start + 1;
        bool in_str = false;
        for (int64_t i = obj_end; i < json_len && depth > 0; i++) {
            if (json[i] == '"' && (i == 0 || json[i-1] != '\\')) in_str = !in_str;
            if (!in_str) {
                if (json[i] == '{') depth++;
                if (json[i] == '}') { depth--; if (depth == 0) obj_end = i; }
            }
        }

        int64_t p = obj_start + 1;
        while (p < obj_end) {
            int64_t ks = -1;
            for (int64_t i = p; i < obj_end; i++) {
                if (json[i] == '"') { ks = i; break; }
            }
            if (ks < 0) break;

            char prop[64];
            json_string(json, ks, prop, 64);
            p = json_string_end(json, ks);
            p = json_find(json, obj_end + 1, p, ':');
            if (p < 0) break;
            p++;

            while (p < obj_end && (json[p] == ' ' || json[p] == '\n' || json[p] == '\t' || json[p] == '\r')) p++;

            if (strcmp(prop, "dtype") == 0) {
                if (json[p] == '"') {
                    char dtype_str[16];
                    int dl = json_string(json, p, dtype_str, 16);
                    t->dtype = parse_dtype(dtype_str, dl);
                    p = json_string_end(json, p);
                }
            } else if (strcmp(prop, "shape") == 0) {
                p = json_find(json, obj_end + 1, p, '[');
                if (p < 0) break;
                p++;
                t->ndims = 0;
                while (p < obj_end) {
                    while (p < obj_end && (json[p] == ' ' || json[p] == ',')) p++;
                    if (json[p] == ']') { p++; break; }
                    t->shape[t->ndims++] = strtoll(json + p, nullptr, 10);
                    while (p < obj_end && json[p] != ',' && json[p] != ']') p++;
                }
            } else if (strcmp(prop, "data_offsets") == 0) {
                p = json_find(json, obj_end + 1, p, '[');
                if (p < 0) break;
                p++;
                while (p < obj_end && json[p] == ' ') p++;
                int64_t start = strtoll(json + p, nullptr, 10);
                p = json_find(json, obj_end + 1, p, ',');
                if (p < 0) break;
                p++;
                while (p < obj_end && json[p] == ' ') p++;
                int64_t end = strtoll(json + p, nullptr, 10);
                t->data_offset = (size_t)start;
                t->data_size = (size_t)(end - start);
                while (p < obj_end && json[p] != ']') p++;
                if (p < obj_end) p++;
            }

            while (p < obj_end && (json[p] == ' ' || json[p] == ',' || json[p] == '\n' || json[p] == '\t' || json[p] == '\r')) p++;
        }

        pos = obj_end + 1;
        n_tensors_++;

        while (pos < json_len && (json[pos] == ' ' || json[pos] == ',' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r')) pos++;
    }

    return true;
}

SafeTensors::~SafeTensors() {
    close();
}

void SafeTensors::close() {
    if (mmap_base_) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

SafeTensors* SafeTensors::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        return nullptr;
    }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    void* base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        fprintf(stderr, "mmap failed\n");
        return nullptr;
    }

    uint64_t header_size = *(uint64_t*)base;
    if (header_size > file_size - 8) {
        munmap(base, file_size);
        ::close(fd);
        fprintf(stderr, "Invalid safetensors header size: %llu\n", header_size);
        return nullptr;
    }

    auto* sf = new SafeTensors();
    sf->fd_ = fd;
    sf->mmap_base_ = base;
    sf->mmap_size_ = file_size;
    sf->header_size_ = header_size;
    sf->data_base_ = (const uint8_t*)base + 8 + header_size;

    const char* json = (const char*)base + 8;
    if (!sf->parse_header(json, (int64_t)header_size)) {
        delete sf;
        fprintf(stderr, "Failed to parse safetensors header\n");
        return nullptr;
    }

    LOG("Loaded safetensors: %d tensors, %.1f MB\n", sf->n_tensors_, file_size / 1e6);
    return sf;
}

const SFTensor* SafeTensors::find(const char* name) const {
    for (int i = 0; i < n_tensors_; i++) {
        if (strcmp(tensors_[i].name, name) == 0) {
            return &tensors_[i];
        }
    }
    return nullptr;
}

const void* SafeTensors::data(const SFTensor* t) const {
    return data_base_ + t->data_offset;
}

int64_t SafeTensors::numel(const SFTensor* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndims; i++) n *= t->shape[i];
    return n;
}

float* SafeTensors::load_bf16_to_f32(const char* name, int64_t expected_numel) const {
    const SFTensor* t = find(name);
    if (!t) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return nullptr;
    }
    int64_t n = numel(t);
    if (expected_numel > 0 && n != expected_numel) {
        fprintf(stderr, "Shape mismatch for %s: expected %lld, got %lld\n", name, expected_numel, n);
        return nullptr;
    }
    float* out = (float*)malloc(n * sizeof(float));
    const uint16_t* bf16 = (const uint16_t*)data(t);
    bf16_to_f32_vec(out, bf16, (int)n);
    return out;
}

uint16_t* SafeTensors::load_bf16_to_f16(const char* name, int64_t expected_numel) const {
    const SFTensor* t = find(name);
    if (!t) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return nullptr;
    }
    int64_t n = numel(t);
    if (expected_numel > 0 && n != expected_numel) {
        fprintf(stderr, "Shape mismatch for %s: expected %lld, got %lld\n", name, expected_numel, n);
        return nullptr;
    }
    uint16_t* out = (uint16_t*)malloc(n * sizeof(uint16_t));
    const uint16_t* bf16 = (const uint16_t*)data(t);
    bf16_to_f16_vec(out, bf16, (int)n);
    return out;
}

float* SafeTensors::load_f32_direct(const char* name, int64_t expected_numel) const {
    const SFTensor* t = find(name);
    if (!t) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return nullptr;
    }
    int64_t n = numel(t);
    if (expected_numel > 0 && n != expected_numel) {
        fprintf(stderr, "Shape mismatch for %s: expected %lld, got %lld\n", name, expected_numel, n);
        return nullptr;
    }
    float* out = (float*)malloc(n * sizeof(float));
    memcpy(out, data(t), n * sizeof(float));
    return out;
}

// Qwen3.5 RMSNorm uses (1+w) pattern
float* SafeTensors::load_norm_weight(const char* name, int64_t expected_numel) const {
    float* w = load_bf16_to_f32(name, expected_numel);
    if (w) {
        for (int64_t i = 0; i < expected_numel; i++) w[i] += 1.0f;
    }
    return w;
}

const uint16_t* SafeTensors::get_bf16_ptr(const char* name) const {
    const SFTensor* t = find(name);
    if (!t) return nullptr;
    return (const uint16_t*)data(t);
}

bool SafeTensors::convert_bf16_to_f16_into(uint16_t* dst, const char* name, int64_t expected_numel) const {
    const SFTensor* t = find(name);
    if (!t) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return false;
    }
    int64_t n = numel(t);
    if (expected_numel > 0 && n != expected_numel) {
        fprintf(stderr, "Shape mismatch for %s: expected %lld, got %lld\n", name, expected_numel, n);
        return false;
    }
    const uint16_t* bf16 = (const uint16_t*)data(t);
    bf16_to_f16_vec(dst, bf16, (int)n);
    return true;
}

bool SafeTensors::convert_bf16_to_f32_into(float* dst, const char* name, int64_t expected_numel) const {
    const SFTensor* t = find(name);
    if (!t) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return false;
    }
    int64_t n = numel(t);
    if (expected_numel > 0 && n != expected_numel) {
        fprintf(stderr, "Shape mismatch for %s: expected %lld, got %lld\n", name, expected_numel, n);
        return false;
    }
    const uint16_t* bf16 = (const uint16_t*)data(t);
    bf16_to_f32_vec(dst, bf16, (int)n);
    return true;
}

// Build ANE weight blob: 128-byte header + FP16 data
// Same format as ane_bridge_build_weight_blob / ane_runtime build_weight_blob
static bool write_ane_blob(const std::string& path, const uint16_t* bf16_data, int64_t num_elements) {
    size_t wsize = (size_t)num_elements * 2;  // FP16 = 2 bytes
    size_t total = 128 + wsize;               // header + data
    uint8_t* buf = (uint8_t*)calloc(total, 1);
    if (!buf) return false;

    // Global header (64 bytes)
    buf[0] = 0x01;
    buf[4] = 0x02;

    // Chunk header (64 bytes at offset 64)
    uint8_t* chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;  // magic
    chunk[4] = 0x01;                                                        // version
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;                              // data size
    *(uint32_t*)(chunk + 16) = 128;                                         // data offset

    // Convert BF16 → FP16 at offset 128
    uint16_t* fp16 = (uint16_t*)(buf + 128);
    bf16_to_f16_vec(fp16, bf16_data, (int)num_elements);

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { free(buf); return false; }
    fwrite(buf, 1, total, fp);
    fclose(fp);
    free(buf);
    return true;
}

int SafeTensors::write_ane_blobs(const SafeTensors& src, const std::string& output_dir) {
    int n = src.n_tensors();
    int written = 0;

    for (int i = 0; i < n; i++) {
        const SFTensor& t = src.tensor(i);
        if (t.dtype != SFDtype::BF16) continue;

        int64_t elem_count = numel(&t);
        const uint16_t* bf16_data = (const uint16_t*)src.data(&t);

        // tensor name → filename: dots to slashes for directory structure
        // e.g. "model.language_model.layers.0.mlp.gate_proj.weight"
        //    → "model/language_model/layers/0/mlp/gate_proj/weight.bin"
        std::string rel_path = t.name;
        for (char& c : rel_path) {
            if (c == '.') c = '/';
        }
        std::string full_path = output_dir + "/" + rel_path + ".bin";

        // Create parent directories
        std::string dir = full_path.substr(0, full_path.rfind('/'));
        std::string tmp;
        for (size_t j = 0; j < dir.size(); j++) {
            tmp += dir[j];
            if (dir[j] == '/' || j == dir.size() - 1)
                mkdir(tmp.c_str(), 0755);
        }

        if (!write_ane_blob(full_path, bf16_data, elem_count)) {
            fprintf(stderr, "Failed to write %s\n", full_path.c_str());
            return -1;
        }
        written++;
    }

    fprintf(stderr, "Wrote %d ANE blobs to %s\n", written, output_dir.c_str());
    return written;
}

} // namespace ane_lm
