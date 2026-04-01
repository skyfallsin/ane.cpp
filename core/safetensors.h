#pragma once

#include <ane_lm/common.h>
#include <cstdint>
#include <string>

namespace ane_lm {

constexpr int SF_MAX_TENSORS = 1024;
constexpr int SF_MAX_NAME = 256;
constexpr int SF_MAX_DIMS = 8;

enum class SFDtype { BF16 = 0, F16, F32, F64, I32, I64, U8, Unknown };

struct SFTensor {
    char name[SF_MAX_NAME];
    SFDtype dtype;
    int64_t shape[SF_MAX_DIMS];
    int ndims;
    size_t data_offset;
    size_t data_size;
};

class SafeTensors {
public:
    ~SafeTensors();

    static SafeTensors* open(const std::string& path);
    void close();

    const SFTensor* find(const char* name) const;
    const void* data(const SFTensor* t) const;
    static int64_t numel(const SFTensor* t);
    static int dtype_size(SFDtype d);

    int n_tensors() const { return n_tensors_; }
    const SFTensor& tensor(int i) const { return tensors_[i]; }

    // Write BF16 tensors as ANE blobs (128-byte header + FP16 data) to output_dir.
    // Returns number of blobs written, or -1 on error.
    static int write_ane_blobs(const SafeTensors& src, const std::string& output_dir);

    // Load helpers (allocate + convert)
    float* load_bf16_to_f32(const char* name, int64_t expected_numel = -1) const;
    uint16_t* load_bf16_to_f16(const char* name, int64_t expected_numel = -1) const;
    float* load_f32_direct(const char* name, int64_t expected_numel = -1) const;
    float* load_norm_weight(const char* name, int64_t expected_numel) const;
    const uint16_t* get_bf16_ptr(const char* name) const;

    // Direct-to-buffer conversion helpers (no temp alloc)
    bool convert_bf16_to_f16_into(uint16_t* dst, const char* name, int64_t expected_numel = -1) const;
    bool convert_bf16_to_f32_into(float* dst, const char* name, int64_t expected_numel = -1) const;

private:
    int fd_ = -1;
    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    uint64_t header_size_ = 0;
    const uint8_t* data_base_ = nullptr;
    SFTensor tensors_[SF_MAX_TENSORS];
    int n_tensors_ = 0;

    bool parse_header(const char* json, int64_t json_len);
};

} // namespace ane_lm
