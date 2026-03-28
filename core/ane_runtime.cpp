// ane_runtime.cpp — ANE compile/load/eval wrapper + MIL kernel generation (pure C++)
// Uses _ANEInMemoryModel via private AppleNeuralEngine.framework through ObjC runtime C API
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <dlfcn.h>
#include <IOSurface/IOSurface.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <ftw.h>
#include "ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

namespace ane_lm {

// ============ ObjC runtime helpers ============

static inline SEL    sel(const char* n) { return sel_registerName(n); }
static inline Class  cls(const char* n) { return (Class)objc_getClass(n); }

static id ns_str(const char* s) {
    return ((id(*)(Class,SEL,const char*))objc_msgSend)(cls("NSString"), sel("stringWithUTF8String:"), s);
}
static const char* to_cstr(id s) {
    if (!s) return "";
    return ((const char*(*)(id,SEL))objc_msgSend)(s, sel("UTF8String"));
}
static id ns_int(int v) {
    return ((id(*)(Class,SEL,int))objc_msgSend)(cls("NSNumber"), sel("numberWithInt:"), v);
}
static id ns_ulong(unsigned long v) {
    return ((id(*)(Class,SEL,unsigned long))objc_msgSend)(cls("NSNumber"), sel("numberWithUnsignedLong:"), v);
}
static id ns_data_nocopy(void* p, unsigned long len) {
    return ((id(*)(Class,SEL,void*,unsigned long,bool))objc_msgSend)(
        cls("NSData"), sel("dataWithBytesNoCopy:length:freeWhenDone:"), p, len, true);
}
static id ns_data(const void* p, unsigned long len) {
    return ((id(*)(Class,SEL,const void*,unsigned long))objc_msgSend)(
        cls("NSData"), sel("dataWithBytes:length:"), p, len);
}
static id ns_dict(id* keys, id* values, unsigned long count) {
    return ((id(*)(Class,SEL,id*,id*,unsigned long))objc_msgSend)(
        cls("NSDictionary"), sel("dictionaryWithObjects:forKeys:count:"), values, keys, count);
}
static id ns_empty_dict() {
    return ((id(*)(Class,SEL))objc_msgSend)(cls("NSDictionary"), sel("dictionary"));
}
static id ns_mutable_array(unsigned long cap) {
    return ((id(*)(Class,SEL,unsigned long))objc_msgSend)(
        cls("NSMutableArray"), sel("arrayWithCapacity:"), cap);
}
static void ns_array_add(id arr, id obj) {
    ((void(*)(id,SEL,id))objc_msgSend)(arr, sel("addObject:"), obj);
}
static id objc_retain_obj(id o) { return ((id(*)(id,SEL))objc_msgSend)(o, sel("retain")); }
static void objc_release_obj(id o) { if (o) ((void(*)(id,SEL))objc_msgSend)(o, sel("release")); }

// ============ C file helpers ============

static void mkdir_p(const std::string& path) {
    std::string tmp;
    for (size_t i = 0; i < path.size(); i++) {
        tmp += path[i];
        if (path[i] == '/' || i == path.size() - 1)
            mkdir(tmp.c_str(), 0755);
    }
}
static void write_file(const std::string& path, const void* data, size_t len) {
    FILE* f = fopen(path.c_str(), "wb");
    if (f) { fwrite(data, 1, len, f); fclose(f); }
}
static bool file_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}
static int nftw_rm_cb(const char* fpath, const struct stat*, int, struct FTW*) {
    return remove(fpath);
}
static void remove_dir(const std::string& path) {
    nftw(path.c_str(), nftw_rm_cb, 64, FTW_DEPTH | FTW_PHYS);
}

// ============ ANEKernel struct ============

struct ANEKernel {
    id model;           // _ANEInMemoryModel (retained)
    IOSurfaceRef* ioInputs;
    IOSurfaceRef* ioOutputs;
    id request;         // _ANERequest (retained)
    std::string tmpDir;
    int nInputs, nOutputs;
    size_t* inputBytes;
    size_t* outputBytes;
};

// ============ Global state ============

static Class g_ANEDesc = nullptr, g_ANEInMem = nullptr, g_ANEReq = nullptr, g_ANEIO = nullptr;
static bool g_ane_ok = false;
static int g_compile_count = 0;
static bool g_ane_persist_cache = true;
static int g_ane_cache_load_count = 0;

void ane_set_persist_cache(bool enabled) { g_ane_persist_cache = enabled; }
int ane_compile_count() { return g_compile_count; }
int ane_cache_loads() { return g_ane_cache_load_count; }

static std::string g_marker_root;
static std::once_flag g_marker_once;

static const std::string& ane_marker_root_dir() {
    std::call_once(g_marker_once, []() {
        const char* home = getenv("HOME");
        g_marker_root = std::string(home ? home : "/tmp") + "/Library/Caches/ane_lm/compiled_markers";
    });
    return g_marker_root;
}

static void ane_remove_compile_dir(const std::string& td, bool force_remove) {
    if (force_remove || !g_ane_persist_cache) remove_dir(td);
}

static std::once_flag g_ane_once;

void ane_init() {
    std::call_once(g_ane_once, []() {
        void* handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        if (!handle) {
            fprintf(stderr, "Warning: Failed to load AppleNeuralEngine.framework: %s\n", dlerror());
            return;
        }
        g_ANEDesc  = cls("_ANEInMemoryModelDescriptor");
        g_ANEInMem = cls("_ANEInMemoryModel");
        g_ANEReq   = cls("_ANERequest");
        g_ANEIO    = cls("_ANEIOSurfaceObject");
        if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
            fprintf(stderr, "Warning: ANE private classes not found\n");
            g_ANEDesc = g_ANEInMem = g_ANEReq = g_ANEIO = nullptr;
            return;
        }
        g_ane_ok = true;
    });
}

bool ane_available() { ane_init(); return g_ane_ok; }

// ============ IOSurface helpers ============

static IOSurfaceRef ane_create_surface(size_t bytes) {
    if (bytes == 0) bytes = 4;
    id keys[] = {
        (id)kIOSurfaceWidth, (id)kIOSurfaceHeight, (id)kIOSurfaceBytesPerElement,
        (id)kIOSurfaceBytesPerRow, (id)kIOSurfaceAllocSize, (id)kIOSurfacePixelFormat
    };
    id values[] = {
        ns_ulong(bytes), ns_int(1), ns_int(1),
        ns_ulong(bytes), ns_ulong(bytes), ns_int(0)
    };
    id dict = ns_dict(keys, values, 6);
    return IOSurfaceCreate((CFDictionaryRef)dict);
}

static bool ane_zero_surface(IOSurfaceRef surface) {
    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) {
        fprintf(stderr, "ANE: IOSurfaceLock failed while zeroing surface\n");
        return false;
    }
    memset(IOSurfaceGetBaseAddress(surface), 0, IOSurfaceGetAllocSize(surface));
    IOSurfaceUnlock(surface, 0, NULL);
    return true;
}

// ============ Weight blob builder ============

static id build_weight_blob(const uint16_t* bf16_data, int num_elements) {
    size_t wsize = (size_t)num_elements * 2;
    size_t total = 64 + 64 + wsize;
    uint8_t* buf = (uint8_t*)calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t* chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128;

    uint16_t* fp16 = (uint16_t*)(buf + 128);
    bf16_to_f16_vec(fp16, bf16_data, num_elements);

    return ns_data_nocopy(buf, total);
}

static id ns_weight_entry(id blob) {
    id keys[]   = { ns_str("offset"), ns_str("data") };
    id values[] = { ns_int(0), blob };
    return ns_dict(keys, values, 2);
}

static id build_weight_dict_1(const uint16_t* bf16, int numel, const char* name) {
    id blob = build_weight_blob(bf16, numel);
    char kbuf[128]; snprintf(kbuf, sizeof(kbuf), "@model_path/weights/%s.bin", name);
    id k = ns_str(kbuf); id v = ns_weight_entry(blob);
    return ns_dict(&k, &v, 1);
}

static id build_weight_dict_2(
    const uint16_t* bf16_a, int numel_a, const char* name_a,
    const uint16_t* bf16_b, int numel_b, const char* name_b)
{
    id ba = build_weight_blob(bf16_a, numel_a);
    id bb = build_weight_blob(bf16_b, numel_b);
    char ka[128], kb[128];
    snprintf(ka, sizeof(ka), "@model_path/weights/%s.bin", name_a);
    snprintf(kb, sizeof(kb), "@model_path/weights/%s.bin", name_b);
    id keys[]   = { ns_str(ka), ns_str(kb) };
    id values[] = { ns_weight_entry(ba), ns_weight_entry(bb) };
    return ns_dict(keys, values, 2);
}

static id build_weight_dict_3(
    const uint16_t* bf16_a, int numel_a, const char* name_a,
    const uint16_t* bf16_b, int numel_b, const char* name_b,
    const uint16_t* bf16_c, int numel_c, const char* name_c)
{
    id ba = build_weight_blob(bf16_a, numel_a);
    id bb = build_weight_blob(bf16_b, numel_b);
    id bc = build_weight_blob(bf16_c, numel_c);
    char ka[128], kb[128], kc[128];
    snprintf(ka, sizeof(ka), "@model_path/weights/%s.bin", name_a);
    snprintf(kb, sizeof(kb), "@model_path/weights/%s.bin", name_b);
    snprintf(kc, sizeof(kc), "@model_path/weights/%s.bin", name_c);
    id keys[]   = { ns_str(ka), ns_str(kb), ns_str(kc) };
    id values[] = { ns_weight_entry(ba), ns_weight_entry(bb), ns_weight_entry(bc) };
    return ns_dict(keys, values, 3);
}

// ============ MIL program generation ============

#define MIL_HEADER \
    "program(1.0)\n" \
    "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n" \
    "{\n"

#define SP ANE_SPATIAL

static id mil_gen_matmul(int out_dim, int in_dim) {
    char buf[4096];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = W, x = x)"
        "[name = tensor<string, []>(\"cv\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP,
        out_dim, in_dim, out_dim, in_dim,
        out_dim, SP);
    return ns_data(buf, n);
}

static id mil_gen_fused_2(int a_out, int b_out, int in_dim) {
    int total_out = a_out + b_out;
    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wa = const()[name = tensor<string, []>(\"Wa\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wa.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wb = const()[name = tensor<string, []>(\"Wb\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wb.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> ya = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wa, x = x)"
        "[name = tensor<string, []>(\"ca\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> yb = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wb, x = x)"
        "[name = tensor<string, []>(\"cb\")];\n"
        "        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n"
        "        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = concat(values = (ya, yb), axis = ax, "
        "interleave = ci)[name = tensor<string, []>(\"cc\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP,
        a_out, in_dim, a_out, in_dim,
        b_out, in_dim, b_out, in_dim,
        a_out, SP,
        b_out, SP,
        total_out, SP);
    return ns_data(buf, n);
}

static id mil_gen_fused_3(int a_out, int b_out, int c_out, int in_dim) {
    int total_out = a_out + b_out + c_out;
    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wa = const()[name = tensor<string, []>(\"Wa\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wa.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wb = const()[name = tensor<string, []>(\"Wb\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wb.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wc = const()[name = tensor<string, []>(\"Wc\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wc.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> ya = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wa, x = x)"
        "[name = tensor<string, []>(\"ca\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> yb = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wb, x = x)"
        "[name = tensor<string, []>(\"cb\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> yc = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wc, x = x)"
        "[name = tensor<string, []>(\"cc\")];\n"
        "        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n"
        "        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = concat(values = (ya, yb, yc), axis = ax, "
        "interleave = ci)[name = tensor<string, []>(\"ct\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP,
        a_out, in_dim, a_out, in_dim,
        b_out, in_dim, b_out, in_dim,
        c_out, in_dim, c_out, in_dim,
        a_out, SP,
        b_out, SP,
        c_out, SP,
        total_out, SP);
    return ns_data(buf, n);
}

static id mil_gen_fused_ffn(int dim, int inter_ch) {
    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wg = const()[name = tensor<string, []>(\"Wg\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wg.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wu = const()[name = tensor<string, []>(\"Wu\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wu.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wd = const()[name = tensor<string, []>(\"Wd\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wd.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wg, x = x)[name = tensor<string, []>(\"cg\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wu, x = x)[name = tensor<string, []>(\"cu\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sg\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"sl\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> out = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wd, x = fused)[name = tensor<string, []>(\"cd\")];\n"
        "    } -> (out);\n"
        "}\n",
        dim, SP,
        inter_ch, dim, inter_ch, dim,
        inter_ch, dim, inter_ch, dim,
        dim, inter_ch, dim, inter_ch,
        inter_ch, SP, inter_ch, SP,
        inter_ch, SP, inter_ch, SP, inter_ch, SP,
        dim, SP);
    return ns_data(buf, n);
}

// ============ Chunked FFN: accumulator-chain MIL ============
// Input: [1, 2*dim, 1, SP] = [x | accumulator] packed in channels
// Output: [1, dim, 1, SP] = acc + down_proj(silu(gate(x)) * up(x))
// All computation on ANE — host only packs input buffer between dispatches.

static id mil_gen_ffn_chunk_accum(int dim, int chunk_inter) {
    char buf[16384];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> packed) {\n"
        // Slice constants
        "        tensor<int32, [4]> sb0 = const()[name = tensor<string, []>(\"sb0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [4]> se0 = const()[name = tensor<string, []>(\"se0\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        "        tensor<int32, [4]> sb1 = const()[name = tensor<string, []>(\"sb1\"), val = tensor<int32, [4]>([0, %d, 0, 0])];\n"
        "        tensor<int32, [4]> se1 = const()[name = tensor<string, []>(\"se1\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        // Slice x and acc from packed input
        "        tensor<fp16, [1, %d, 1, %d]> x = slice_by_index(x = packed, begin = sb0, end = se0)"
        "[name = tensor<string, []>(\"sx\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> acc = slice_by_index(x = packed, begin = sb1, end = se1)"
        "[name = tensor<string, []>(\"sa\")];\n"
        // Conv constants
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        // Weight constants
        "        tensor<fp16, [%d, %d, 1, 1]> Wg = const()[name = tensor<string, []>(\"Wg\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wg.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wu = const()[name = tensor<string, []>(\"Wu\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wu.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wd = const()[name = tensor<string, []>(\"Wd\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wd.bin\"), offset = tensor<uint64, []>(64)))];\n"
        // gate + up projections
        "        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wg, x = x)[name = tensor<string, []>(\"cg\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wu, x = x)[name = tensor<string, []>(\"cu\")];\n"
        // SiLU activation
        "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sg\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"sl\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n"
        // down projection (partial)
        "        tensor<fp16, [1, %d, 1, %d]> partial = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wd, x = fused)[name = tensor<string, []>(\"cd\")];\n"
        // Accumulate on ANE
        "        tensor<fp16, [1, %d, 1, %d]> out = add(x = acc, y = partial)[name = tensor<string, []>(\"ad\")];\n"
        "    } -> (out);\n"
        "}\n",
        /* packed input */ 2 * dim, SP,
        /* se0 */ dim, SP,
        /* sb1 */ dim,
        /* se1 */ 2 * dim, SP,
        /* x slice */ dim, SP,
        /* acc slice */ dim, SP,
        /* Wg */ chunk_inter, dim, chunk_inter, dim,
        /* Wu */ chunk_inter, dim, chunk_inter, dim,
        /* Wd */ dim, chunk_inter, dim, chunk_inter,
        /* gate */ chunk_inter, SP,
        /* up */ chunk_inter, SP,
        /* sig */ chunk_inter, SP,
        /* silu */ chunk_inter, SP,
        /* fused */ chunk_inter, SP,
        /* partial */ dim, SP,
        /* out */ dim, SP);
    return ns_data(buf, n);
}

// Determine how many chunks needed so each fits under ANE compile limit.
// LM head matmul kernels compile fine at ~34MB (16384*1024*2). Fused FFN
// with 3 weight matrices works at ~22MB total (0.8B). Conservative limit
// per chunk: 48MB (single kernels can be larger than fused multi-weight ones).
static constexpr size_t ANE_FFN_CHUNK_MAX_BYTES = 150 * 1024 * 1024;

int ane_ffn_chunk_count(int dim, int inter_ch) {
    size_t total = (size_t)3 * inter_ch * dim * 2; // 3 matrices, fp16
    if (total <= ANE_FFN_CHUNK_MAX_BYTES) return 1;
    int n = (int)((total + ANE_FFN_CHUNK_MAX_BYTES - 1) / ANE_FFN_CHUNK_MAX_BYTES);
    // Round up to a divisor of inter_ch
    while (inter_ch % n != 0) n++;
    return n;
}

// ============ Core compile/eval/free ============

static ANEKernel* ane_compile_raw(id milText, id wdict,
                                   int nInputs, size_t* inputSizes,
                                   int nOutputs, size_t* outputSizes) {
    if (!ane_available()) return nullptr;

    void* local_pool = objc_autoreleasePoolPush();

    // Create descriptor
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, sel("modelWithMILText:weights:optionsPlist:"),
        milText, wdict ? wdict : ns_empty_dict(), (id)nullptr);
    if (!desc) {
        fprintf(stderr, "ANE: modelWithMILText failed\n");
        objc_autoreleasePoolPop(local_pool);
        return nullptr;
    }

    // Create in-memory model
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, sel("inMemoryModelWithDescriptor:"), desc);
    if (!mdl) {
        fprintf(stderr, "ANE: inMemoryModelWithDescriptor returned nil\n");
        objc_autoreleasePoolPop(local_pool);
        return nullptr;
    }

    // Get hex identifier for cache key
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, sel("hexStringIdentifier"));
    std::string modelId;
    if (hx) {
        bool isStr = ((bool(*)(id,SEL,Class))objc_msgSend)(hx, sel("isKindOfClass:"), cls("NSString"));
        if (isStr) {
            unsigned long len = ((unsigned long(*)(id,SEL))objc_msgSend)(hx, sel("length"));
            if (len > 0) modelId = to_cstr(hx);
        }
    }
    if (modelId.empty()) {
        id uuid = ((id(*)(Class,SEL))objc_msgSend)(cls("NSUUID"), sel("UUID"));
        id uuidStr = ((id(*)(id,SEL))objc_msgSend)(uuid, sel("UUIDString"));
        modelId = to_cstr(uuidStr);
    }

    const char* tmpenv = getenv("TMPDIR");
    std::string td = std::string(tmpenv ? tmpenv : "/tmp") + "/" + modelId;
    const std::string& markerRoot = ane_marker_root_dir();
    mkdir_p(markerRoot);
    std::string compiledMarker = markerRoot + "/" + modelId + ".ok";
    mkdir_p(td + "/weights");

    // Write MIL text to file
    const void* milBytes = ((const void*(*)(id,SEL))objc_msgSend)(milText, sel("bytes"));
    unsigned long milLen = ((unsigned long(*)(id,SEL))objc_msgSend)(milText, sel("length"));
    write_file(td + "/model.mil", milBytes, milLen);

    // Write weight files
    if (wdict) {
        id allKeys = ((id(*)(id,SEL))objc_msgSend)(wdict, sel("allKeys"));
        unsigned long keyCount = ((unsigned long(*)(id,SEL))objc_msgSend)(allKeys, sel("count"));
        for (unsigned long i = 0; i < keyCount; i++) {
            id key = ((id(*)(id,SEL,unsigned long))objc_msgSend)(allKeys, sel("objectAtIndex:"), i);
            std::string keyStr = to_cstr(key);
            std::string relPath = keyStr;
            size_t pos = relPath.find("@model_path/");
            if (pos != std::string::npos) relPath.erase(pos, strlen("@model_path/"));
            std::string fullPath = td + "/" + relPath;

            id entry = ((id(*)(id,SEL,id))objc_msgSend)(wdict, sel("objectForKey:"), key);
            id data = ((id(*)(id,SEL,id))objc_msgSend)(entry, sel("objectForKey:"), ns_str("data"));
            const void* dataBytes = ((const void*(*)(id,SEL))objc_msgSend)(data, sel("bytes"));
            unsigned long dataLen = ((unsigned long(*)(id,SEL))objc_msgSend)(data, sel("length"));
            write_file(fullPath, dataBytes, dataLen);
        }
    }

    // Cache check / compile / load
    id e = nullptr;
    bool loaded_from_cache = false;
    if (g_ane_persist_cache && file_exists(compiledMarker)) {
        e = nullptr;
        bool ok = ((bool(*)(id,SEL,unsigned int,id,id*))objc_msgSend)(
            mdl, sel("loadWithQoS:options:error:"), 21, ns_empty_dict(), &e);
        if (ok) {
            loaded_from_cache = true;
            g_ane_cache_load_count++;
        } else {
            remove(compiledMarker.c_str());
            e = nullptr;
        }
    }

    if (!loaded_from_cache) {
        e = nullptr;
        if (!((bool(*)(id,SEL,unsigned int,id,id*))objc_msgSend)(
                mdl, sel("compileWithQoS:options:error:"), 21, ns_empty_dict(), &e)) {
            fprintf(stderr, "ANE compile failed: %s\n",
                e ? to_cstr(((id(*)(id,SEL))objc_msgSend)(e, sel("description"))) : "unknown");
            remove(compiledMarker.c_str());
            ane_remove_compile_dir(td, true);
            objc_autoreleasePoolPop(local_pool);
            return nullptr;
        }
        e = nullptr;
        if (!((bool(*)(id,SEL,unsigned int,id,id*))objc_msgSend)(
                mdl, sel("loadWithQoS:options:error:"), 21, ns_empty_dict(), &e)) {
            fprintf(stderr, "ANE load failed: %s\n",
                e ? to_cstr(((id(*)(id,SEL))objc_msgSend)(e, sel("description"))) : "unknown");
            remove(compiledMarker.c_str());
            ane_remove_compile_dir(td, true);
            objc_autoreleasePoolPop(local_pool);
            return nullptr;
        }
        g_compile_count++;
        if (g_ane_persist_cache) {
            write_file(compiledMarker, "ok", 2);
        } else {
            remove(compiledMarker.c_str());
        }
    }

    // Create kernel struct
    ANEKernel* k = new ANEKernel();
    k->model = objc_retain_obj(mdl);
    k->tmpDir = td;
    k->nInputs = nInputs;
    k->nOutputs = nOutputs;
    k->inputBytes = (size_t*)malloc(nInputs * sizeof(size_t));
    k->outputBytes = (size_t*)malloc(nOutputs * sizeof(size_t));
    memcpy(k->inputBytes, inputSizes, nInputs * sizeof(size_t));
    memcpy(k->outputBytes, outputSizes, nOutputs * sizeof(size_t));

    // Create IOSurfaces
    k->ioInputs = (IOSurfaceRef*)malloc(nInputs * sizeof(IOSurfaceRef));
    k->ioOutputs = (IOSurfaceRef*)malloc(nOutputs * sizeof(IOSurfaceRef));
    for (int i = 0; i < nInputs; i++) {
        k->ioInputs[i] = ane_create_surface(inputSizes[i]);
        if (!k->ioInputs[i] || !ane_zero_surface(k->ioInputs[i])) {
            fprintf(stderr, "ANE: failed to init input IOSurface %d\n", i);
            delete k;
            objc_autoreleasePoolPop(local_pool);
            return nullptr;
        }
    }
    for (int i = 0; i < nOutputs; i++) {
        k->ioOutputs[i] = ane_create_surface(outputSizes[i]);
        if (!k->ioOutputs[i] || !ane_zero_surface(k->ioOutputs[i])) {
            fprintf(stderr, "ANE: failed to init output IOSurface %d\n", i);
            delete k;
            objc_autoreleasePoolPop(local_pool);
            return nullptr;
        }
    }

    // Create ANE request
    id wIns = ns_mutable_array(nInputs);
    id iIdx = ns_mutable_array(nInputs);
    for (int i = 0; i < nInputs; i++) {
        id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, sel("objectWithIOSurface:"), k->ioInputs[i]);
        ns_array_add(wIns, ioObj);
        ns_array_add(iIdx, ns_int(i));
    }
    id wOuts = ns_mutable_array(nOutputs);
    id oIdx = ns_mutable_array(nOutputs);
    for (int i = 0; i < nOutputs; i++) {
        id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, sel("objectWithIOSurface:"), k->ioOutputs[i]);
        ns_array_add(wOuts, ioObj);
        ns_array_add(oIdx, ns_int(i));
    }
    k->request = objc_retain_obj(
        ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq, sel("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
            wIns, iIdx, wOuts, oIdx, (id)nullptr, (id)nullptr, ns_int(0)));

    objc_autoreleasePoolPop(local_pool);
    return k;
}

// Cached ObjC objects for hot-path eval — avoid per-call objc_msgSend overhead
static id g_eval_empty_dict = nullptr;
static SEL g_eval_sel = nullptr;

static bool ane_eval_raw(ANEKernel* k) {
    if (__builtin_expect(!g_eval_sel, 0)) {
        g_eval_sel = sel("evaluateWithQoS:options:request:error:");
        g_eval_empty_dict = objc_retain_obj(ns_empty_dict());
    }
    id e = nullptr;
    bool ok = ((bool(*)(id,SEL,unsigned int,id,id,id*))objc_msgSend)(
        k->model, g_eval_sel, 21, g_eval_empty_dict, k->request, &e);
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n",
            e ? to_cstr(((id(*)(id,SEL))objc_msgSend)(e, sel("description"))) : "unknown");
    }
    return ok;
}

// ============ Public API implementations ============

#if defined(__aarch64__) || defined(__arm64__)
typedef __fp16 ane_fp16_t;
#define ANE_USE_NATIVE_FP16 1
#else
#define ANE_USE_NATIVE_FP16 0
#endif

// Skip IOSurface lock/unlock on hot path — ANE hardware manages coherency via eval.
// The lock/unlock is only needed for initial setup; once surfaces are allocated and
// base addresses cached, direct memory access works.
// Set ANE_SKIP_LOCKS=1 to enable.
static bool g_skip_locks = false;
static bool g_skip_locks_checked = false;

bool ane_matvec(ANEKernel* k, float* output, const float* input, int in_dim, int out_dim) {
    if (__builtin_expect(!g_skip_locks_checked, 0)) {
        g_skip_locks = getenv("ANE_SKIP_LOCKS") != nullptr;
        g_skip_locks_checked = true;
    }

    IOSurfaceRef in_surface = k->ioInputs[0];
    if (!g_skip_locks) {
        if (IOSurfaceLock(in_surface, 0, NULL) != kIOReturnSuccess) {
            fprintf(stderr, "ANE: IOSurfaceLock(input) failed\n");
            return false;
        }
    }
    uint16_t* in_base = (uint16_t*)IOSurfaceGetBaseAddress(in_surface);
#if ANE_USE_NATIVE_FP16
    ane_fp16_t* in_base_h = (ane_fp16_t*)in_base;
#pragma clang loop vectorize(enable)
    for (int c = 0, idx = 0; c < in_dim; c++, idx += ANE_SPATIAL)
        in_base_h[idx] = (ane_fp16_t)input[c];
#else
    for (int c = 0, idx = 0; c < in_dim; c++, idx += ANE_SPATIAL)
        in_base[idx] = f32_to_f16(input[c]);
#endif
    if (!g_skip_locks) IOSurfaceUnlock(in_surface, 0, NULL);

    if (!ane_eval_raw(k)) return false;

    IOSurfaceRef out_surface = k->ioOutputs[0];
    if (!g_skip_locks) {
        if (IOSurfaceLock(out_surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
            fprintf(stderr, "ANE: IOSurfaceLock(output) failed\n");
            return false;
        }
    }
    const uint16_t* out_base = (const uint16_t*)IOSurfaceGetBaseAddress(out_surface);
#if ANE_USE_NATIVE_FP16
    const ane_fp16_t* out_base_h = (const ane_fp16_t*)out_base;
#pragma clang loop vectorize(enable)
    for (int c = 0, idx = 0; c < out_dim; c++, idx += ANE_SPATIAL)
        output[c] = (float)out_base_h[idx];
#else
    for (int c = 0, idx = 0; c < out_dim; c++, idx += ANE_SPATIAL)
        output[c] = f16_to_f32(out_base[idx]);
#endif
    if (!g_skip_locks) IOSurfaceUnlock(out_surface, kIOSurfaceLockReadOnly, NULL);

    return true;
}

void ane_free(ANEKernel* k) {
    if (!k) return;
    id e = nullptr;
    ((bool(*)(id,SEL,unsigned int,id*))objc_msgSend)(
        k->model, sel("unloadWithQoS:error:"), 21, &e);
    for (int i = 0; i < k->nInputs; i++) CFRelease(k->ioInputs[i]);
    for (int i = 0; i < k->nOutputs; i++) CFRelease(k->ioOutputs[i]);
    if (!g_ane_persist_cache) {
        remove_dir(k->tmpDir);
    }
    free(k->ioInputs); free(k->ioOutputs);
    free(k->inputBytes); free(k->outputBytes);
    objc_release_obj(k->request);
    objc_release_obj(k->model);
    delete k;
}

void ane_free_layer(LayerANEKernels* lk) {
    ane_free(lk->first_proj);
    ane_free(lk->o_proj);
    ane_free(lk->fused_ffn);
    ane_free(lk->fused_oproj_norm);
    ane_free(lk->fused_oproj_ffn);
    ane_free(lk->oproj_add);
    ane_free_chunked_ffn(&lk->chunked_ffn);
    lk->first_proj = lk->o_proj = lk->fused_ffn = lk->fused_oproj_norm = lk->fused_oproj_ffn = lk->oproj_add = nullptr;
}

// ============ High-level compile functions ============

ANEKernel* ane_compile_matmul(const uint16_t* bf16_weights, int out_dim, int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = build_weight_dict_1(bf16_weights, out_dim * in_dim, "weight");
    id mil = mil_gen_matmul(out_dim, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)out_dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_2(const uint16_t* bf16_a, int a_out,
                                const uint16_t* bf16_b, int b_out,
                                int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = build_weight_dict_2(bf16_a, a_out * in_dim, "wa",
                                    bf16_b, b_out * in_dim, "wb");
    id mil = mil_gen_fused_2(a_out, b_out, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)(a_out + b_out) * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_3(const uint16_t* bf16_a, int a_out,
                                const uint16_t* bf16_b, int b_out,
                                const uint16_t* bf16_c, int c_out,
                                int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = build_weight_dict_3(bf16_a, a_out * in_dim, "wa",
                                    bf16_b, b_out * in_dim, "wb",
                                    bf16_c, c_out * in_dim, "wc");
    id mil = mil_gen_fused_3(a_out, b_out, c_out, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)(a_out + b_out + c_out) * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

// Forward declaration (defined later in blob section)
static id load_blob_file(const std::string& path);

// ============ Fused O_proj + residual add + RMSNorm ============
// 2 inputs:  attn_out [1, in_dim, 1, SP], x_residual [1, out_dim, 1, SP]
// 2 outputs: x_norm [1, out_dim, 1, SP], x_updated [1, out_dim, 1, SP]
// Ops: conv(O_proj, attn_out) → add(x_residual, o_proj_result) → RMSNorm → output both

static id mil_gen_fused_oproj_norm(int out_dim, int in_dim) {
    // out_dim = hidden_size (O_proj output, norm dim)
    // in_dim  = full_out_dim (O_proj input, attn head concat)
    std::string s;
    char buf[512];
    s += MIL_HEADER;
    snprintf(buf, sizeof(buf),
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> attn,"
        " tensor<fp16, [1, %d, 1, %d]> xres) {\n", in_dim, SP, out_dim, SP);
    s += buf;

    // Conv params (shared)
    s += "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n";

    // O_proj conv
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [%d, %d, 1, 1]> Wo = const()[name = tensor<string, []>(\"Wo\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/oproj.bin\"), offset = tensor<uint64, []>(64)))];\n",
        out_dim, in_dim, out_dim, in_dim);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> oproj = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wo, x = attn)"
        "[name = tensor<string, []>(\"op\")];\n", out_dim, SP);
    s += buf;

    // Residual add
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> x = add(x = xres, y = oproj)"
        "[name = tensor<string, []>(\"ad\")];\n", out_dim, SP);
    s += buf;

    // RMSNorm: x² → reduce_sum → ×(1/dim) → +eps → sqrt → 1/sqrt → ×x → ×weight
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> sq = mul(x = x, y = x)"
        "[name = tensor<string, []>(\"sq\")];\n", out_dim, SP);
    s += buf;
    s += "        tensor<int32, [1]> axes = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
         "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n";
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> ss = reduce_sum(x = sq, axes = axes, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> di = const()[name = tensor<string, []>(\"di\"), "
        "val = tensor<fp16, [1, 1, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/dim_inv.bin\"), offset = tensor<uint64, []>(64)))];\n", SP, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> ms = mul(x = ss, y = di)"
        "[name = tensor<string, []>(\"ms\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> ep = const()[name = tensor<string, []>(\"ep\"), "
        "val = tensor<fp16, [1, 1, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/eps.bin\"), offset = tensor<uint64, []>(64)))];\n", SP, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> ae = add(x = ms, y = ep)"
        "[name = tensor<string, []>(\"ae\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> sr = sqrt(x = ae)"
        "[name = tensor<string, []>(\"sr\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> on = const()[name = tensor<string, []>(\"on\"), "
        "val = tensor<fp16, [1, 1, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/ones.bin\"), offset = tensor<uint64, []>(64)))];\n", SP, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> sc = real_div(x = on, y = sr)"
        "[name = tensor<string, []>(\"is\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> xs = mul(x = x, y = sc)"
        "[name = tensor<string, []>(\"xs\")];\n", out_dim, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> wn = const()[name = tensor<string, []>(\"wn\"), "
        "val = tensor<fp16, [1, %d, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/nw.bin\"), offset = tensor<uint64, []>(64)))];\n",
        out_dim, SP, out_dim, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> normed = mul(x = xs, y = wn)"
        "[name = tensor<string, []>(\"nm\")];\n", out_dim, SP);
    s += buf;

    // Two outputs: normed (for FFN) and x (updated residual)
    s += "    } -> (normed, x);\n}\n";
    return ns_data(s.c_str(), s.size());
}

// Build a BF16 weight blob from float* norm weights, laid out as [dim, SP] with one value per row
static id build_norm_weight_blob(const float* norm_weight, int dim) {
    int numel = dim * SP;
    uint16_t* bf16 = (uint16_t*)calloc(numel, sizeof(uint16_t));
    for (int i = 0; i < dim; i++)
        bf16[i * SP] = f32_to_bf16(norm_weight[i]);
    id blob = build_weight_blob(bf16, numel);
    free(bf16);
    return blob;
}

// Build scalar const blob [1, 1, 1, SP] with value in position 0
static id build_scalar_blob(float value) {
    uint16_t bf16[SP] = {};
    bf16[0] = f32_to_bf16(value);
    return build_weight_blob(bf16, SP);
}

ANEKernel* ane_compile_fused_oproj_norm(const uint16_t* oproj_bf16,
                                         const float* norm_weight,
                                         int out_dim, int in_dim, float eps) {
    void* pool = objc_autoreleasePoolPush();

    id w_oproj = build_weight_blob(oproj_bf16, out_dim * in_dim);
    id w_nw = build_norm_weight_blob(norm_weight, out_dim);
    id w_di = build_scalar_blob(1.0f / out_dim);
    id w_ep = build_scalar_blob(eps);
    id w_on = build_scalar_blob(1.0f);

    id keys[] = {
        ns_str("@model_path/weights/oproj.bin"),
        ns_str("@model_path/weights/nw.bin"),
        ns_str("@model_path/weights/dim_inv.bin"),
        ns_str("@model_path/weights/eps.bin"),
        ns_str("@model_path/weights/ones.bin"),
    };
    id values[] = {
        ns_weight_entry(w_oproj),
        ns_weight_entry(w_nw),
        ns_weight_entry(w_di),
        ns_weight_entry(w_ep),
        ns_weight_entry(w_on),
    };
    id wdict = ns_dict(keys, values, 5);

    id mil = mil_gen_fused_oproj_norm(out_dim, in_dim);
    size_t in_sizes[2] = {
        (size_t)in_dim * SP * sizeof(uint16_t),
        (size_t)out_dim * SP * sizeof(uint16_t),
    };
    size_t out_sizes[2] = {
        (size_t)out_dim * SP * sizeof(uint16_t),
        (size_t)out_dim * SP * sizeof(uint16_t),
    };
    ANEKernel* r = ane_compile_raw(mil, wdict, 2, in_sizes, 2, out_sizes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_oproj_norm_blob(const std::string& oproj_path,
                                              const float* norm_weight,
                                              int out_dim, int in_dim, float eps) {
    void* pool = objc_autoreleasePoolPush();

    id w_oproj = load_blob_file(oproj_path);
    if (!w_oproj) { objc_autoreleasePoolPop(pool); return nullptr; }
    id w_nw = build_norm_weight_blob(norm_weight, out_dim);
    id w_di = build_scalar_blob(1.0f / out_dim);
    id w_ep = build_scalar_blob(eps);
    id w_on = build_scalar_blob(1.0f);

    id keys[] = {
        ns_str("@model_path/weights/oproj.bin"),
        ns_str("@model_path/weights/nw.bin"),
        ns_str("@model_path/weights/dim_inv.bin"),
        ns_str("@model_path/weights/eps.bin"),
        ns_str("@model_path/weights/ones.bin"),
    };
    id values[] = {
        ns_weight_entry(w_oproj),
        ns_weight_entry(w_nw),
        ns_weight_entry(w_di),
        ns_weight_entry(w_ep),
        ns_weight_entry(w_on),
    };
    id wdict = ns_dict(keys, values, 5);

    id mil = mil_gen_fused_oproj_norm(out_dim, in_dim);
    size_t in_sizes[2] = {
        (size_t)in_dim * SP * sizeof(uint16_t),
        (size_t)out_dim * SP * sizeof(uint16_t),
    };
    size_t out_sizes[2] = {
        (size_t)out_dim * SP * sizeof(uint16_t),
        (size_t)out_dim * SP * sizeof(uint16_t),
    };
    ANEKernel* r = ane_compile_raw(mil, wdict, 2, in_sizes, 2, out_sizes);
    objc_autoreleasePoolPop(pool);
    return r;
}

bool ane_eval_fused_oproj_norm(ANEKernel* k, float* x_norm, float* x_updated,
                                const float* attn_out, const float* x_residual,
                                int in_dim, int out_dim) {
    float* ins[2] = { (float*)attn_out, (float*)x_residual };
    int in_chs[2] = { in_dim, out_dim };
    float* outs[2] = { x_norm, x_updated };
    int out_chs[2] = { out_dim, out_dim };
    return ane_eval_multi(k, ins, in_chs, outs, out_chs);
}

// ============ Simplified O_proj + residual add (no RMSNorm) ============
// Just conv + add — the RMSNorm chain was redundant (CPU redoes it for precision)

static id mil_gen_oproj_add(int out_dim, int in_dim) {
    std::string s;
    char buf[512];
    s += MIL_HEADER;
    snprintf(buf, sizeof(buf),
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> attn,"
        " tensor<fp16, [1, %d, 1, %d]> xres) {\n", in_dim, SP, out_dim, SP);
    s += buf;

    // Conv params
    s += "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n";

    // O_proj conv
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [%d, %d, 1, 1]> Wo = const()[name = tensor<string, []>(\"Wo\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/oproj.bin\"), offset = tensor<uint64, []>(64)))];\n",
        out_dim, in_dim, out_dim, in_dim);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> oproj = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wo, x = attn)"
        "[name = tensor<string, []>(\"op\")];\n", out_dim, SP);
    s += buf;

    // Residual add — single output
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> y = add(x = xres, y = oproj)"
        "[name = tensor<string, []>(\"ad\")];\n", out_dim, SP);
    s += buf;

    s += "    } -> (y);\n}\n";
    return ns_data(s.c_str(), s.size());
}

ANEKernel* ane_compile_oproj_add(const uint16_t* oproj_bf16, int out_dim, int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id w_oproj = build_weight_blob(oproj_bf16, out_dim * in_dim);
    id keys[] = { ns_str("@model_path/weights/oproj.bin") };
    id values[] = { ns_weight_entry(w_oproj) };
    id wdict = ns_dict(keys, values, 1);

    id mil = mil_gen_oproj_add(out_dim, in_dim);
    size_t in_sizes[2] = {
        (size_t)in_dim * SP * sizeof(uint16_t),
        (size_t)out_dim * SP * sizeof(uint16_t),
    };
    size_t out_size = (size_t)out_dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 2, in_sizes, 1, &out_size);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_oproj_add_blob(const std::string& oproj_path, int out_dim, int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id w_oproj = load_blob_file(oproj_path);
    if (!w_oproj) { objc_autoreleasePoolPop(pool); return nullptr; }
    id keys[] = { ns_str("@model_path/weights/oproj.bin") };
    id values[] = { ns_weight_entry(w_oproj) };
    id wdict = ns_dict(keys, values, 1);

    id mil = mil_gen_oproj_add(out_dim, in_dim);
    size_t in_sizes[2] = {
        (size_t)in_dim * SP * sizeof(uint16_t),
        (size_t)out_dim * SP * sizeof(uint16_t),
    };
    size_t out_size = (size_t)out_dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 2, in_sizes, 1, &out_size);
    objc_autoreleasePoolPop(pool);
    return r;
}

bool ane_eval_oproj_add(ANEKernel* k, float* x_updated,
                         const float* attn_out, const float* x_residual,
                         int in_dim, int out_dim) {
    float* ins[2] = { (float*)attn_out, (float*)x_residual };
    int in_chs[2] = { in_dim, out_dim };
    float* outs[1] = { x_updated };
    int out_chs[1] = { out_dim };
    return ane_eval_multi(k, ins, in_chs, outs, out_chs);
}

// ============ Fused O_proj + add + RMSNorm + SwiGLU FFN ============
// The mega-kernel: 2 runtime inputs, 4 const convs, RMSNorm, SiLU
// This replaces 2 ANE dispatches (oproj_norm + FFN) with 1

static id mil_gen_fused_oproj_ffn(int dim, int in_dim, int inter_ch) {
    std::string s;
    char buf[1024];
    s += MIL_HEADER;

    // 2 runtime inputs
    snprintf(buf, sizeof(buf),
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> attn,"
        " tensor<fp16, [1, %d, 1, %d]> xres) {\n", in_dim, SP, dim, SP);
    s += buf;

    // Conv shared params
    s += "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n";

    // --- O_proj conv ---
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [%d, %d, 1, 1]> Wo = const()[name = tensor<string, []>(\"Wo\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/oproj.bin\"), offset = tensor<uint64, []>(64)))];\n",
        dim, in_dim, dim, in_dim);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> oproj = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wo, x = attn)"
        "[name = tensor<string, []>(\"op\")];\n", dim, SP);
    s += buf;

    // --- Residual add ---
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> x = add(x = xres, y = oproj)"
        "[name = tensor<string, []>(\"ad\")];\n", dim, SP);
    s += buf;

    // --- RMSNorm (pre-scaled for fp16 safety) ---
    // RMSNorm is scale-invariant: RMSNorm(αx) = RMSNorm(x)
    // Pre-scale by 1/8 so sum(x²)/64 stays within fp16 range (max 65504)
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> rsk = const()[name = tensor<string, []>(\"rsk\"), "
        "val = tensor<fp16, [1, 1, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/rms_scale.bin\"), offset = tensor<uint64, []>(64)))];\n", SP, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> xsc = mul(x = x, y = rsk)"
        "[name = tensor<string, []>(\"xsc\")];\n", dim, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> sq = mul(x = xsc, y = xsc)"
        "[name = tensor<string, []>(\"sq\")];\n", dim, SP);
    s += buf;
    s += "        tensor<int32, [1]> axes = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
         "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n";
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> ss = reduce_sum(x = sq, axes = axes, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> di = const()[name = tensor<string, []>(\"di\"), "
        "val = tensor<fp16, [1, 1, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/dim_inv.bin\"), offset = tensor<uint64, []>(64)))];\n", SP, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> ms = mul(x = ss, y = di)"
        "[name = tensor<string, []>(\"ms\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> ep = const()[name = tensor<string, []>(\"ep\"), "
        "val = tensor<fp16, [1, 1, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/eps.bin\"), offset = tensor<uint64, []>(64)))];\n", SP, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> ae = add(x = ms, y = ep)"
        "[name = tensor<string, []>(\"ae\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> sr = sqrt(x = ae)"
        "[name = tensor<string, []>(\"sr\")];\n", SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> on = const()[name = tensor<string, []>(\"on\"), "
        "val = tensor<fp16, [1, 1, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/ones.bin\"), offset = tensor<uint64, []>(64)))];\n", SP, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, 1, 1, %d]> sc = real_div(x = on, y = sr)"
        "[name = tensor<string, []>(\"is\")];\n", SP);
    s += buf;
    // Apply inv_rms to SCALED x (not original) — scale cancels: (x/k)*(k/rms(x)) = x/rms(x)
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> xs = mul(x = xsc, y = sc)"
        "[name = tensor<string, []>(\"xs\")];\n", dim, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> wn = const()[name = tensor<string, []>(\"wn\"), "
        "val = tensor<fp16, [1, %d, 1, %d]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/nw.bin\"), offset = tensor<uint64, []>(64)))];\n",
        dim, SP, dim, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> normed = mul(x = xs, y = wn)"
        "[name = tensor<string, []>(\"nm\")];\n", dim, SP);
    s += buf;

    // --- SwiGLU FFN: gate, up, silu, mul, down ---
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [%d, %d, 1, 1]> Wg = const()[name = tensor<string, []>(\"Wg\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/wg.bin\"), offset = tensor<uint64, []>(64)))];\n",
        inter_ch, dim, inter_ch, dim);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [%d, %d, 1, 1]> Wu = const()[name = tensor<string, []>(\"Wu\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/wu.bin\"), offset = tensor<uint64, []>(64)))];\n",
        inter_ch, dim, inter_ch, dim);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [%d, %d, 1, 1]> Wd = const()[name = tensor<string, []>(\"Wd\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>"
        "(\"@model_path/weights/wd.bin\"), offset = tensor<uint64, []>(64)))];\n",
        dim, inter_ch, dim, inter_ch);
    s += buf;

    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wg, x = normed)"
        "[name = tensor<string, []>(\"cg\")];\n", inter_ch, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wu, x = normed)"
        "[name = tensor<string, []>(\"cu\")];\n", inter_ch, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)"
        "[name = tensor<string, []>(\"sg\")];\n", inter_ch, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)"
        "[name = tensor<string, []>(\"sl\")];\n", inter_ch, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> fused = mul(x = silu, y = up)"
        "[name = tensor<string, []>(\"fu\")];\n", inter_ch, SP);
    s += buf;
    snprintf(buf, sizeof(buf),
        "        tensor<fp16, [1, %d, 1, %d]> ffn = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wd, x = fused)"
        "[name = tensor<string, []>(\"cd\")];\n", dim, SP);
    s += buf;

    // 2 outputs: ffn_out (for residual add) and x (updated residual before norm)
    s += "    } -> (ffn, x);\n}\n";
    return ns_data(s.c_str(), s.size());
}

ANEKernel* ane_compile_fused_oproj_ffn(const uint16_t* oproj_bf16,
                                        const uint16_t* gate_bf16,
                                        const uint16_t* up_bf16,
                                        const uint16_t* down_bf16,
                                        const float* norm_weight,
                                        int dim, int in_dim, int inter_ch, float eps) {
    void* pool = objc_autoreleasePoolPush();

    id w_oproj = build_weight_blob(oproj_bf16, dim * in_dim);
    id w_nw = build_norm_weight_blob(norm_weight, dim);
    id w_di = build_scalar_blob(1.0f / dim);
    id w_ep = build_scalar_blob(eps);
    id w_on = build_scalar_blob(1.0f);
    id w_rsk = build_scalar_blob(1.0f / 8.0f);  // pre-scale for fp16 RMSNorm safety
    id w_gate = build_weight_blob(gate_bf16, inter_ch * dim);
    id w_up = build_weight_blob(up_bf16, inter_ch * dim);
    id w_down = build_weight_blob(down_bf16, dim * inter_ch);

    id keys[] = {
        ns_str("@model_path/weights/oproj.bin"),
        ns_str("@model_path/weights/nw.bin"),
        ns_str("@model_path/weights/dim_inv.bin"),
        ns_str("@model_path/weights/eps.bin"),
        ns_str("@model_path/weights/ones.bin"),
        ns_str("@model_path/weights/rms_scale.bin"),
        ns_str("@model_path/weights/wg.bin"),
        ns_str("@model_path/weights/wu.bin"),
        ns_str("@model_path/weights/wd.bin"),
    };
    id values[] = {
        ns_weight_entry(w_oproj),
        ns_weight_entry(w_nw),
        ns_weight_entry(w_di),
        ns_weight_entry(w_ep),
        ns_weight_entry(w_on),
        ns_weight_entry(w_rsk),
        ns_weight_entry(w_gate),
        ns_weight_entry(w_up),
        ns_weight_entry(w_down),
    };
    id wdict = ns_dict(keys, values, 9);

    id mil = mil_gen_fused_oproj_ffn(dim, in_dim, inter_ch);
    size_t in_sizes[2] = {
        (size_t)in_dim * SP * sizeof(uint16_t),
        (size_t)dim * SP * sizeof(uint16_t),
    };
    size_t out_sizes[2] = {
        (size_t)dim * SP * sizeof(uint16_t),
        (size_t)dim * SP * sizeof(uint16_t),
    };
    ANEKernel* r = ane_compile_raw(mil, wdict, 2, in_sizes, 2, out_sizes);
    objc_autoreleasePoolPop(pool);
    return r;
}

bool ane_eval_fused_oproj_ffn(ANEKernel* k, float* ffn_out, float* x_updated,
                               const float* attn_out, const float* x_residual,
                               int in_dim, int dim) {
    float* ins[2] = { (float*)attn_out, (float*)x_residual };
    int in_chs[2] = { in_dim, dim };
    float* outs[2] = { ffn_out, x_updated };
    int out_chs[2] = { dim, dim };
    return ane_eval_multi(k, ins, in_chs, outs, out_chs);
}

ANEKernel* ane_compile_fused_ffn(const uint16_t* gate_bf16, const uint16_t* up_bf16,
                                  const uint16_t* down_bf16, int dim, int inter_ch) {
    void* pool = objc_autoreleasePoolPush();
    id wg = build_weight_blob(gate_bf16, inter_ch * dim);
    id wu = build_weight_blob(up_bf16, inter_ch * dim);
    id wd = build_weight_blob(down_bf16, dim * inter_ch);

    id keys[]   = { ns_str("@model_path/weights/wg.bin"),
                    ns_str("@model_path/weights/wu.bin"),
                    ns_str("@model_path/weights/wd.bin") };
    id values[] = { ns_weight_entry(wg), ns_weight_entry(wu), ns_weight_entry(wd) };
    id wdict = ns_dict(keys, values, 3);

    id mil = mil_gen_fused_ffn(dim, inter_ch);
    size_t in_size = (size_t)dim * SP * sizeof(uint16_t);
    size_t out_size = (size_t)dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_size, 1, &out_size);
    objc_autoreleasePoolPop(pool);
    return r;
}

// ============ Blob file loading ============

static id load_blob_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ANE: cannot open blob %s\n", path.c_str());
        return (id)nullptr;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* buf = malloc(len);
    fread(buf, 1, len, f);
    fclose(f);
    return ns_data_nocopy(buf, len);
}

static id blob_weight_dict_1(const std::string& path, const char* name) {
    id blob = load_blob_file(path);
    if (!blob) return (id)nullptr;
    char kbuf[128]; snprintf(kbuf, sizeof(kbuf), "@model_path/weights/%s.bin", name);
    id k = ns_str(kbuf); id v = ns_weight_entry(blob);
    return ns_dict(&k, &v, 1);
}

static id blob_weight_dict_2(const std::string& a_path, const char* name_a,
                              const std::string& b_path, const char* name_b) {
    id ba = load_blob_file(a_path);
    id bb = load_blob_file(b_path);
    if (!ba || !bb) return (id)nullptr;
    char ka[128], kb[128];
    snprintf(ka, sizeof(ka), "@model_path/weights/%s.bin", name_a);
    snprintf(kb, sizeof(kb), "@model_path/weights/%s.bin", name_b);
    id keys[]   = { ns_str(ka), ns_str(kb) };
    id values[] = { ns_weight_entry(ba), ns_weight_entry(bb) };
    return ns_dict(keys, values, 2);
}

static id blob_weight_dict_3(const std::string& a_path, const char* name_a,
                              const std::string& b_path, const char* name_b,
                              const std::string& c_path, const char* name_c) {
    id ba = load_blob_file(a_path);
    id bb = load_blob_file(b_path);
    id bc = load_blob_file(c_path);
    if (!ba || !bb || !bc) return (id)nullptr;
    char ka[128], kb[128], kc[128];
    snprintf(ka, sizeof(ka), "@model_path/weights/%s.bin", name_a);
    snprintf(kb, sizeof(kb), "@model_path/weights/%s.bin", name_b);
    snprintf(kc, sizeof(kc), "@model_path/weights/%s.bin", name_c);
    id keys[]   = { ns_str(ka), ns_str(kb), ns_str(kc) };
    id values[] = { ns_weight_entry(ba), ns_weight_entry(bb), ns_weight_entry(bc) };
    return ns_dict(keys, values, 3);
}

// ============ High-level compile from blob files ============

ANEKernel* ane_compile_matmul_blob(const std::string& blob_path, int out_dim, int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = blob_weight_dict_1(blob_path, "weight");
    if (!wdict) { objc_autoreleasePoolPop(pool); return nullptr; }
    id mil = mil_gen_matmul(out_dim, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)out_dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_2_blob(const std::string& a_path, int a_out,
                                     const std::string& b_path, int b_out,
                                     int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = blob_weight_dict_2(a_path, "wa", b_path, "wb");
    if (!wdict) { objc_autoreleasePoolPop(pool); return nullptr; }
    id mil = mil_gen_fused_2(a_out, b_out, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)(a_out + b_out) * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_3_blob(const std::string& a_path, int a_out,
                                     const std::string& b_path, int b_out,
                                     const std::string& c_path, int c_out,
                                     int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = blob_weight_dict_3(a_path, "wa", b_path, "wb", c_path, "wc");
    if (!wdict) { objc_autoreleasePoolPop(pool); return nullptr; }
    id mil = mil_gen_fused_3(a_out, b_out, c_out, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)(a_out + b_out + c_out) * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_ffn_blob(const std::string& gate_path, const std::string& up_path,
                                       const std::string& down_path, int dim, int inter_ch) {
    void* pool = objc_autoreleasePoolPush();
    id wg = load_blob_file(gate_path);
    id wu = load_blob_file(up_path);
    id wd = load_blob_file(down_path);
    if (!wg || !wu || !wd) { objc_autoreleasePoolPop(pool); return nullptr; }

    id keys[]   = { ns_str("@model_path/weights/wg.bin"),
                    ns_str("@model_path/weights/wu.bin"),
                    ns_str("@model_path/weights/wd.bin") };
    id values[] = { ns_weight_entry(wg), ns_weight_entry(wu), ns_weight_entry(wd) };
    id wdict = ns_dict(keys, values, 3);

    id mil = mil_gen_fused_ffn(dim, inter_ch);
    size_t in_size = (size_t)dim * SP * sizeof(uint16_t);
    size_t out_size = (size_t)dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_size, 1, &out_size);
    objc_autoreleasePoolPop(pool);
    return r;
}

// ============ Chunked FFN compile/eval ============

// Helper: build weight blob from a SLICE of bf16 data
// gate/up: rows [chunk_start..chunk_start+chunk_inter) of [inter_ch, dim] matrix
// down:    cols [chunk_start..chunk_start+chunk_inter) of [dim, inter_ch] matrix
static id build_weight_blob_rows(const uint16_t* bf16_data, int total_rows, int cols,
                                  int row_start, int row_count) {
    int numel = row_count * cols;
    size_t wsize = (size_t)numel * 2;
    size_t total = 64 + 64 + wsize;
    uint8_t* buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t* chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128;

    uint16_t* fp16 = (uint16_t*)(buf + 128);
    const uint16_t* src = bf16_data + (size_t)row_start * cols;
    bf16_to_f16_vec(fp16, src, numel);
    return ns_data_nocopy(buf, total);
}

// down_proj is [dim, inter_ch] — we need cols [chunk_start..+chunk_inter)
// In memory layout (row-major): row r, col c = data[r * inter_ch + c]
// We extract a submatrix [dim, chunk_inter]
static id build_weight_blob_cols(const uint16_t* bf16_data, int rows, int total_cols,
                                  int col_start, int col_count) {
    int numel = rows * col_count;
    size_t wsize = (size_t)numel * 2;
    size_t total = 64 + 64 + wsize;
    uint8_t* buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t* chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128;

    uint16_t* fp16 = (uint16_t*)(buf + 128);
    for (int r = 0; r < rows; r++) {
        const uint16_t* src_row = bf16_data + (size_t)r * total_cols + col_start;
        uint16_t* dst_row = fp16 + (size_t)r * col_count;
        bf16_to_f16_vec(dst_row, src_row, col_count);
    }
    return ns_data_nocopy(buf, total);
}

bool ane_compile_chunked_ffn(ChunkedFFN* out, const uint16_t* gate_bf16,
                              const uint16_t* up_bf16, const uint16_t* down_bf16,
                              int dim, int inter_ch, int num_chunks) {
    int chunk_inter = inter_ch / num_chunks;
    out->chunks = (ANEKernel**)calloc(num_chunks, sizeof(ANEKernel*));
    out->num_chunks = num_chunks;
    out->dim = dim;
    out->chunk_inter = chunk_inter;

    id mil = mil_gen_ffn_chunk_accum(dim, chunk_inter);
    size_t in_size = (size_t)(2 * dim) * SP * sizeof(uint16_t);
    size_t out_size = (size_t)dim * SP * sizeof(uint16_t);

    for (int c = 0; c < num_chunks; c++) {
        void* pool = objc_autoreleasePoolPush();
        int offset = c * chunk_inter;

        // gate/up: slice rows [offset..offset+chunk_inter) of [inter_ch, dim]
        id wg = build_weight_blob_rows(gate_bf16, inter_ch, dim, offset, chunk_inter);
        id wu = build_weight_blob_rows(up_bf16, inter_ch, dim, offset, chunk_inter);
        // down: slice cols [offset..offset+chunk_inter) of [dim, inter_ch]
        id wd = build_weight_blob_cols(down_bf16, dim, inter_ch, offset, chunk_inter);

        id keys[]   = { ns_str("@model_path/weights/wg.bin"),
                        ns_str("@model_path/weights/wu.bin"),
                        ns_str("@model_path/weights/wd.bin") };
        id values[] = { ns_weight_entry(wg), ns_weight_entry(wu), ns_weight_entry(wd) };
        id wdict = ns_dict(keys, values, 3);

        out->chunks[c] = ane_compile_raw(mil, wdict, 1, &in_size, 1, &out_size);
        objc_autoreleasePoolPop(pool);

        if (!out->chunks[c]) {
            fprintf(stderr, "ANE chunked FFN compile failed at chunk %d/%d\n", c + 1, num_chunks);
            ane_free_chunked_ffn(out);
            return false;
        }
    }
    return true;
}

bool ane_compile_chunked_ffn_blob(ChunkedFFN* out, const std::string& gate_path,
                                   const std::string& up_path, const std::string& down_path,
                                   int dim, int inter_ch, int num_chunks) {
    // Load full blobs, extract bf16 data pointers, delegate to bf16 version
    // For blob files: header is 128 bytes, then fp16 data (already converted)
    // We need to slice them — load as raw, skip header, treat as bf16
    FILE* fg = fopen(gate_path.c_str(), "rb");
    FILE* fu = fopen(up_path.c_str(), "rb");
    FILE* fd = fopen(down_path.c_str(), "rb");
    if (!fg || !fu || !fd) {
        if (fg) fclose(fg); if (fu) fclose(fu); if (fd) fclose(fd);
        return false;
    }

    size_t gate_size = (size_t)inter_ch * dim;
    size_t down_size = (size_t)dim * inter_ch;

    // Blob files have 128-byte header then fp16 data
    // But we need bf16 for the build_weight_blob helpers... 
    // Actually blob files are already fp16. We'll handle this differently:
    // Just fall back to the non-blob path since we need to slice anyway
    fclose(fg); fclose(fu); fclose(fd);
    fprintf(stderr, "Chunked FFN blob path not yet supported, use non-blob path\n");
    return false;
}

// Eval: chain through all chunks. Host packs [x | acc] between dispatches.
bool ane_eval_chunked_ffn(const ChunkedFFN* cffn, float* output, const float* input) {
    int dim = cffn->dim;
    int packed_ch = 2 * dim;

    for (int c = 0; c < cffn->num_chunks; c++) {
        ANEKernel* k = cffn->chunks[c];
        IOSurfaceRef in_surface = k->ioInputs[0];

        // Lock and write packed input: [x | acc]
        if (IOSurfaceLock(in_surface, 0, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
        ane_fp16_t* in_base = (ane_fp16_t*)IOSurfaceGetBaseAddress(in_surface);
#else
        uint16_t* in_base = (uint16_t*)IOSurfaceGetBaseAddress(in_surface);
#endif

        // Write x into first dim channels
#pragma clang loop vectorize(enable)
        for (int i = 0, idx = 0; i < dim; i++, idx += ANE_SPATIAL) {
#if ANE_USE_NATIVE_FP16
            in_base[idx] = (ane_fp16_t)input[i];
#else
            in_base[idx] = f32_to_f16(input[i]);
#endif
        }

        // Write accumulator into next dim channels
        if (c == 0) {
            // First chunk: acc = zeros
            for (int i = 0, idx = dim * ANE_SPATIAL; i < dim; i++, idx += ANE_SPATIAL) {
#if ANE_USE_NATIVE_FP16
                in_base[idx] = (ane_fp16_t)0.0f;
#else
                in_base[idx] = 0;
#endif
            }
        } else {
            // Subsequent chunks: acc = previous output (already in output buffer)
            // Read from previous chunk's output surface and write to this chunk's input
            IOSurfaceRef prev_out = cffn->chunks[c - 1]->ioOutputs[0];
            if (IOSurfaceLock(prev_out, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
                IOSurfaceUnlock(in_surface, 0, NULL);
                return false;
            }
#if ANE_USE_NATIVE_FP16
            const ane_fp16_t* prev_base = (const ane_fp16_t*)IOSurfaceGetBaseAddress(prev_out);
#else
            const uint16_t* prev_base = (const uint16_t*)IOSurfaceGetBaseAddress(prev_out);
#endif
            // Copy prev output (dim channels) into acc portion of packed input
            size_t acc_offset = (size_t)dim * ANE_SPATIAL;
#pragma clang loop vectorize(enable)
            for (int i = 0, idx = 0; i < dim; i++, idx += ANE_SPATIAL) {
                in_base[acc_offset + idx] = prev_base[idx];
            }
            IOSurfaceUnlock(prev_out, kIOSurfaceLockReadOnly, NULL);
        }

        IOSurfaceUnlock(in_surface, 0, NULL);

        // Dispatch on ANE
        if (!ane_eval_raw(k)) return false;
    }

    // Read final output from last chunk
    ANEKernel* last = cffn->chunks[cffn->num_chunks - 1];
    IOSurfaceRef out_surface = last->ioOutputs[0];
    if (IOSurfaceLock(out_surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
    const ane_fp16_t* out_base = (const ane_fp16_t*)IOSurfaceGetBaseAddress(out_surface);
#else
    const uint16_t* out_base = (const uint16_t*)IOSurfaceGetBaseAddress(out_surface);
#endif
#pragma clang loop vectorize(enable)
    for (int i = 0, idx = 0; i < dim; i++, idx += ANE_SPATIAL) {
#if ANE_USE_NATIVE_FP16
        output[i] = (float)out_base[idx];
#else
        output[i] = f16_to_f32(out_base[idx]);
#endif
    }
    IOSurfaceUnlock(out_surface, kIOSurfaceLockReadOnly, NULL);
    return true;
}

void ane_free_chunked_ffn(ChunkedFFN* cffn) {
    if (!cffn || !cffn->chunks) return;
    for (int i = 0; i < cffn->num_chunks; i++) ane_free(cffn->chunks[i]);
    free(cffn->chunks);
    cffn->chunks = nullptr;
    cffn->num_chunks = 0;
}

// ============ Generic MIL compile ============

ANEKernel* ane_compile_mil(const char* mil_text, int n_inputs, size_t* input_sizes,
                            int n_outputs, size_t* output_sizes) {
    if (!ane_available()) return nullptr;
    void* pool = objc_autoreleasePoolPush();
    id mil = ns_data(mil_text, strlen(mil_text));
    ANEKernel* k = ane_compile_raw(mil, (id)nullptr, n_inputs, input_sizes, n_outputs, output_sizes);
    objc_autoreleasePoolPop(pool);
    return k;
}

// ============ Generic MIL compile with weight blobs ============

ANEKernel* ane_compile_mil_weighted(const char* mil_text,
                                     int n_inputs, size_t* input_sizes,
                                     int n_outputs, size_t* output_sizes,
                                     MILWeight* weights, int n_weights) {
    if (!ane_available()) return nullptr;
    void* pool = objc_autoreleasePoolPush();
    id mil = ns_data(mil_text, strlen(mil_text));

    // Build weight dict from MILWeight array
    id wdict = (id)nullptr;
    if (n_weights > 0) {
        id* keys = (id*)alloca(n_weights * sizeof(id));
        id* values = (id*)alloca(n_weights * sizeof(id));
        for (int i = 0; i < n_weights; i++) {
            char kbuf[128];
            snprintf(kbuf, sizeof(kbuf), "@model_path/weights/%s.bin", weights[i].name);
            keys[i] = ns_str(kbuf);
            values[i] = ns_weight_entry(build_weight_blob(weights[i].bf16, weights[i].numel));
        }
        wdict = ns_dict(keys, values, (unsigned long)n_weights);
    }

    ANEKernel* k = ane_compile_raw(mil, wdict, n_inputs, input_sizes, n_outputs, output_sizes);
    objc_autoreleasePoolPop(pool);
    return k;
}

// ============ Generic MIL compile with raw weight blobs ============

ANEKernel* ane_compile_mil_raw(const char* mil_text,
                                int n_inputs, size_t* input_sizes,
                                int n_outputs, size_t* output_sizes,
                                MILRawWeight* weights, int n_weights) {
    if (!ane_available()) return nullptr;
    void* pool = objc_autoreleasePoolPush();
    id mil = ns_data(mil_text, strlen(mil_text));

    // Build weight dict from raw blob data (no bf16→fp16 conversion)
    id wdict = (id)nullptr;
    if (n_weights > 0) {
        id* keys = (id*)alloca(n_weights * sizeof(id));
        id* values = (id*)alloca(n_weights * sizeof(id));
        for (int i = 0; i < n_weights; i++) {
            char kbuf[128];
            snprintf(kbuf, sizeof(kbuf), "@model_path/weights/%s.bin", weights[i].name);
            keys[i] = ns_str(kbuf);
            // Wrap raw blob bytes in NSData — written verbatim to disk by ane_compile_raw
            id blob = ns_data(weights[i].data, (unsigned long)weights[i].size);
            values[i] = ns_weight_entry(blob);
        }
        wdict = ns_dict(keys, values, (unsigned long)n_weights);
    }

    ANEKernel* k = ane_compile_raw(mil, wdict, n_inputs, input_sizes, n_outputs, output_sizes);
    objc_autoreleasePoolPop(pool);
    return k;
}

// ============ Dynamic-weight conv ============
// MIL program with TWO inputs: x and W (weights passed at runtime, not baked in)
// Uses conv() op — ANE accepts non-constant weight inputs for conv!

static id mil_gen_dynamic_conv(int out_dim, int in_dim) {
    char buf[4096];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, 1]> W) {\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = W, x = x)"
        "[name = tensor<string, []>(\"cv\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP, out_dim, in_dim,
        out_dim, SP);
    return ns_data(buf, n);
}

// Dynamic FFN: 4 inputs (x, gate_W, up_W, down_W), all conv ops with runtime weights
static id mil_gen_dynamic_ffn_conv(int dim, int inter_ch) {
    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>("
        "tensor<fp16, [1, %d, 1, %d]> x, "
        "tensor<fp16, [%d, %d, 1, 1]> Wg, "
        "tensor<fp16, [%d, %d, 1, 1]> Wu, "
        "tensor<fp16, [%d, %d, 1, 1]> Wd) {\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        // gate = Wg conv x
        "        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wg, x = x)"
        "[name = tensor<string, []>(\"cg\")];\n"
        // up = Wu conv x
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wu, x = x)"
        "[name = tensor<string, []>(\"cu\")];\n"
        // silu(gate) = gate * sigmoid(gate)
        "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sg\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"sl\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n"
        // down = Wd conv fused
        "        tensor<fp16, [1, %d, 1, %d]> out = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wd, x = fused)"
        "[name = tensor<string, []>(\"cd\")];\n"
        "    } -> (out);\n"
        "}\n",
        /* inputs */ dim, SP, inter_ch, dim, inter_ch, dim, dim, inter_ch,
        /* gate */ inter_ch, SP,
        /* up */ inter_ch, SP,
        /* sig */ inter_ch, SP,
        /* silu */ inter_ch, SP,
        /* fused */ inter_ch, SP,
        /* down */ dim, SP);
    return ns_data(buf, n);
}

ANEKernel* ane_compile_dynamic_conv(int out_dim, int in_dim) {
    if (!ane_available()) return nullptr;
    void* pool = objc_autoreleasePoolPush();
    
    id mil = mil_gen_dynamic_conv(out_dim, in_dim);
    
    // ANE 4D tensor format: [N, C, H, W] — innermost dim (W) is padded to SP=32
    // For weight [out_dim, in_dim, 1, 1]: N=out_dim, C=in_dim, H=1, W=1
    // Each (n,c,h) row has W=1 padded to SP=32
    // Total IOSurface size: out_dim * in_dim * 1 * SP * sizeof(fp16)
    size_t input_sizes[2] = {
        (size_t)in_dim * SP * sizeof(uint16_t),              // x [1, in_dim, 1, SP]
        (size_t)out_dim * in_dim * SP * sizeof(uint16_t)     // W [out_dim, in_dim, 1, 1] padded
    };
    size_t output_size = (size_t)out_dim * SP * sizeof(uint16_t);
    
    ANEKernel* k = ane_compile_raw(mil, (id)nullptr, 2, input_sizes, 1, &output_size);
    objc_autoreleasePoolPop(pool);
    return k;
}

bool ane_dynamic_conv_eval(ANEKernel* k, float* output, const float* input,
                            const uint16_t* fp16_weights, int in_dim, int out_dim) {
    if (!k) return false;
    
    // Write input x into IOSurface 0 (strided: every SP-th element)
    IOSurfaceRef x_surf = k->ioInputs[0];
    if (IOSurfaceLock(x_surf, 0, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
    ane_fp16_t* x_base = (ane_fp16_t*)IOSurfaceGetBaseAddress(x_surf);
#pragma clang loop vectorize(enable)
    for (int c = 0, idx = 0; c < in_dim; c++, idx += SP)
        x_base[idx] = (ane_fp16_t)input[c];
#else
    uint16_t* x_base = (uint16_t*)IOSurfaceGetBaseAddress(x_surf);
    for (int c = 0, idx = 0; c < in_dim; c++, idx += SP)
        x_base[idx] = f32_to_f16(input[c]);
#endif
    IOSurfaceUnlock(x_surf, 0, NULL);
    
    // Write weights into IOSurface 1
    // ANE format for [out_dim, in_dim, 1, 1]: each element at stride SP
    // Element [o, i, 0, 0] → offset (o * in_dim * SP + i * SP) * sizeof(fp16)
    IOSurfaceRef w_surf = k->ioInputs[1];
    if (IOSurfaceLock(w_surf, 0, NULL) != kIOReturnSuccess) return false;
    uint16_t* w_base = (uint16_t*)IOSurfaceGetBaseAddress(w_surf);
    // Zero the entire surface first (padding positions)
    memset(w_base, 0, IOSurfaceGetAllocSize(w_surf));
    // Write each weight element at its strided position
    for (int o = 0; o < out_dim; o++) {
        for (int i = 0; i < in_dim; i++) {
            w_base[(size_t)o * in_dim * SP + (size_t)i * SP] = fp16_weights[(size_t)o * in_dim + i];
        }
    }
    IOSurfaceUnlock(w_surf, 0, NULL);
    
    // Eval on ANE
    if (!ane_eval_raw(k)) return false;
    
    // Read output (strided: every SP-th element)
    IOSurfaceRef out_surf = k->ioOutputs[0];
    if (IOSurfaceLock(out_surf, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
    const ane_fp16_t* out_base = (const ane_fp16_t*)IOSurfaceGetBaseAddress(out_surf);
#pragma clang loop vectorize(enable)
    for (int c = 0, idx = 0; c < out_dim; c++, idx += SP)
        output[c] = (float)out_base[idx];
#else
    const uint16_t* out_base = (const uint16_t*)IOSurfaceGetBaseAddress(out_surf);
    for (int c = 0, idx = 0; c < out_dim; c++, idx += SP)
        output[c] = f16_to_f32(out_base[idx]);
#endif
    IOSurfaceUnlock(out_surf, kIOSurfaceLockReadOnly, NULL);
    return true;
}

ANEKernel* ane_compile_dynamic_ffn(int dim, int inter_ch) {
    if (!ane_available()) return nullptr;
    void* pool = objc_autoreleasePoolPush();
    
    id mil = mil_gen_dynamic_ffn_conv(dim, inter_ch);
    
    // 4 inputs: x, gate_W, up_W, down_W — all with SP padding on innermost dim
    size_t input_sizes[4] = {
        (size_t)dim * SP * sizeof(uint16_t),                // x [1, dim, 1, SP]
        (size_t)inter_ch * dim * SP * sizeof(uint16_t),     // Wg [inter, dim, 1, 1] padded
        (size_t)inter_ch * dim * SP * sizeof(uint16_t),     // Wu [inter, dim, 1, 1] padded
        (size_t)dim * inter_ch * SP * sizeof(uint16_t),     // Wd [dim, inter, 1, 1] padded
    };
    size_t output_size = (size_t)dim * SP * sizeof(uint16_t);
    
    ANEKernel* k = ane_compile_raw(mil, (id)nullptr, 4, input_sizes, 1, &output_size);
    objc_autoreleasePoolPop(pool);
    return k;
}

bool ane_dynamic_ffn_eval(ANEKernel* k, float* output, const float* input,
                           const uint16_t* gate_fp16, const uint16_t* up_fp16,
                           const uint16_t* down_fp16, int dim, int inter_ch) {
    if (!k) return false;
    
    // Write x into IOSurface 0
    IOSurfaceRef x_surf = k->ioInputs[0];
    if (IOSurfaceLock(x_surf, 0, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
    ane_fp16_t* x_base = (ane_fp16_t*)IOSurfaceGetBaseAddress(x_surf);
    for (int c = 0, idx = 0; c < dim; c++, idx += SP)
        x_base[idx] = (ane_fp16_t)input[c];
#else
    uint16_t* x_base = (uint16_t*)IOSurfaceGetBaseAddress(x_surf);
    for (int c = 0, idx = 0; c < dim; c++, idx += SP)
        x_base[idx] = f32_to_f16(input[c]);
#endif
    IOSurfaceUnlock(x_surf, 0, NULL);
    
    // Helper lambda: write fp16 weights with ANE SP stride into IOSurface
    auto write_strided_weights = [](IOSurfaceRef surf, const uint16_t* src, int rows, int cols) -> bool {
        if (IOSurfaceLock(surf, 0, NULL) != kIOReturnSuccess) return false;
        uint16_t* base = (uint16_t*)IOSurfaceGetBaseAddress(surf);
        memset(base, 0, IOSurfaceGetAllocSize(surf));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                base[(size_t)r * cols * SP + (size_t)c * SP] = src[(size_t)r * cols + c];
        IOSurfaceUnlock(surf, 0, NULL);
        return true;
    };
    
    // Write gate weights [inter, dim] into IOSurface 1
    if (!write_strided_weights(k->ioInputs[1], gate_fp16, inter_ch, dim)) return false;
    // Write up weights [inter, dim] into IOSurface 2
    if (!write_strided_weights(k->ioInputs[2], up_fp16, inter_ch, dim)) return false;
    // Write down weights [dim, inter] into IOSurface 3
    if (!write_strided_weights(k->ioInputs[3], down_fp16, dim, inter_ch)) return false;
    
    // Eval
    if (!ane_eval_raw(k)) return false;
    
    // Read output
    IOSurfaceRef out_surf = k->ioOutputs[0];
    if (IOSurfaceLock(out_surf, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
    const ane_fp16_t* out_base = (const ane_fp16_t*)IOSurfaceGetBaseAddress(out_surf);
    for (int c = 0, idx = 0; c < dim; c++, idx += SP)
        output[c] = (float)out_base[idx];
#else
    const uint16_t* out_base = (const uint16_t*)IOSurfaceGetBaseAddress(out_surf);
    for (int c = 0, idx = 0; c < dim; c++, idx += SP)
        output[c] = f16_to_f32(out_base[idx]);
#endif
    IOSurfaceUnlock(out_surf, kIOSurfaceLockReadOnly, NULL);
    return true;
}

// ============ Generic multi-input eval ============

bool ane_eval_multi(ANEKernel* k,
                    float** inputs, int* input_channels,
                    float** outputs, int* output_channels) {
    if (!k) return false;
    
    // Write all inputs (SP-strided fp16)
    for (int i = 0; i < k->nInputs; i++) {
        IOSurfaceRef surf = k->ioInputs[i];
        if (IOSurfaceLock(surf, 0, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
        ane_fp16_t* base = (ane_fp16_t*)IOSurfaceGetBaseAddress(surf);
#else
        uint16_t* base = (uint16_t*)IOSurfaceGetBaseAddress(surf);
#endif
        memset(base, 0, IOSurfaceGetAllocSize(surf));
        int ch = input_channels[i];
        for (int c = 0; c < ch; c++) {
#if ANE_USE_NATIVE_FP16
            base[(size_t)c * SP] = (ane_fp16_t)inputs[i][c];
#else
            base[(size_t)c * SP] = f32_to_f16(inputs[i][c]);
#endif
        }
        IOSurfaceUnlock(surf, 0, NULL);
    }
    
    // Eval
    if (!ane_eval_raw(k)) return false;
    
    // Read all outputs (SP-strided fp16)
    for (int i = 0; i < k->nOutputs; i++) {
        IOSurfaceRef surf = k->ioOutputs[i];
        if (IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
        const ane_fp16_t* base = (const ane_fp16_t*)IOSurfaceGetBaseAddress(surf);
#else
        const uint16_t* base = (const uint16_t*)IOSurfaceGetBaseAddress(surf);
#endif
        int ch = output_channels[i];
        for (int c = 0; c < ch; c++) {
#if ANE_USE_NATIVE_FP16
            outputs[i][c] = (float)base[(size_t)c * SP];
#else
            outputs[i][c] = f16_to_f32(base[(size_t)c * SP]);
#endif
        }
        IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
    }
    return true;
}

bool ane_write_surface_raw(ANEKernel* k, int input_idx, const uint16_t* data, size_t bytes) {
    if (!k || input_idx >= k->nInputs) return false;
    IOSurfaceRef surf = k->ioInputs[input_idx];
    if (IOSurfaceLock(surf, 0, NULL) != kIOReturnSuccess) return false;
    size_t alloc = IOSurfaceGetAllocSize(surf);
    size_t to_write = bytes < alloc ? bytes : alloc;
    memset(IOSurfaceGetBaseAddress(surf), 0, alloc);
    memcpy(IOSurfaceGetBaseAddress(surf), data, to_write);
    IOSurfaceUnlock(surf, 0, NULL);
    return true;
}

bool ane_write_surface_strided(ANEKernel* k, int input_idx, const uint16_t* data, int channels) {
    if (!k || input_idx >= k->nInputs) return false;
    IOSurfaceRef surf = k->ioInputs[input_idx];
    if (IOSurfaceLock(surf, 0, NULL) != kIOReturnSuccess) return false;
    uint16_t* base = (uint16_t*)IOSurfaceGetBaseAddress(surf);
    memset(base, 0, IOSurfaceGetAllocSize(surf));
    for (int c = 0; c < channels; c++)
        base[(size_t)c * SP] = data[c];
    IOSurfaceUnlock(surf, 0, NULL);
    return true;
}

size_t ane_get_input_size(ANEKernel* k, int input_idx) {
    if (!k || input_idx >= k->nInputs) return 0;
    return k->inputBytes[input_idx];
}

size_t ane_get_output_size(ANEKernel* k, int output_idx) {
    if (!k || output_idx >= k->nOutputs) return 0;
    return k->outputBytes[output_idx];
}

int ane_get_n_inputs(ANEKernel* k) { return k ? k->nInputs : 0; }
int ane_get_n_outputs(ANEKernel* k) { return k ? k->nOutputs : 0; }

bool ane_eval_raw_outputs(ANEKernel* k, float** outputs, int* output_channels) {
    if (!k) return false;
    if (!ane_eval_raw(k)) return false;
    for (int i = 0; i < k->nOutputs; i++) {
        IOSurfaceRef surf = k->ioOutputs[i];
        if (IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
#if ANE_USE_NATIVE_FP16
        const ane_fp16_t* base = (const ane_fp16_t*)IOSurfaceGetBaseAddress(surf);
#else
        const uint16_t* base = (const uint16_t*)IOSurfaceGetBaseAddress(surf);
#endif
        int ch = output_channels[i];
        for (int c = 0; c < ch; c++) {
#if ANE_USE_NATIVE_FP16
            outputs[i][c] = (float)base[(size_t)c * SP];
#else
            outputs[i][c] = f16_to_f32(base[(size_t)c * SP]);
#endif
        }
        IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
    }
    return true;
}


bool ane_read_output_raw(ANEKernel* k, int output_idx, float* data, int count) {
    if (!k || output_idx >= k->nOutputs) return false;
    IOSurfaceRef surf = k->ioOutputs[output_idx];
    if (IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    const uint16_t* base = (const uint16_t*)IOSurfaceGetBaseAddress(surf);
    size_t alloc = IOSurfaceGetAllocSize(surf);
    int max_count = (int)(alloc / sizeof(uint16_t));
    if (count > max_count) count = max_count;
#if ANE_USE_NATIVE_FP16
    const ane_fp16_t* base_h = (const ane_fp16_t*)base;
    for (int i = 0; i < count; i++) data[i] = (float)base_h[i];
#else
    for (int i = 0; i < count; i++) data[i] = f16_to_f32(base[i]);
#endif
    IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
    return true;
}

bool ane_write_input_tiled(ANEKernel* k, int input_idx, const float* data,
                           int N, int C, int H, int W) {
    if (!k || input_idx >= k->nInputs) return false;
    IOSurfaceRef surf = k->ioInputs[input_idx];
    if (IOSurfaceLock(surf, 0, NULL) != kIOReturnSuccess) return false;
    uint16_t* base = (uint16_t*)IOSurfaceGetBaseAddress(surf);
    memset(base, 0, IOSurfaceGetAllocSize(surf));
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    size_t idx = (size_t)(((n * C + c) * H + h) * W + w);
                    base[idx] = f32_to_f16(data[idx]);
                }
    IOSurfaceUnlock(surf, 0, NULL);
    return true;
}

// ============ API Tuning ============

void* ane_get_model(ANEKernel* k) { return k ? (void*)k->model : nullptr; }
void* ane_get_request(ANEKernel* k) { return k ? (void*)k->request : nullptr; }

bool ane_eval_qos(ANEKernel* k, int qos) {
    if (!k) return false;
    SEL eval_sel = sel("evaluateWithQoS:options:request:error:");
    id e = nullptr;
    bool ok = ((bool(*)(id,SEL,unsigned int,id,id,id*))objc_msgSend)(
        k->model, eval_sel, (unsigned int)qos, ns_empty_dict(), k->request, &e);
    if (!ok) {
        fprintf(stderr, "ANE eval failed (qos=%d): %s\n", qos,
            e ? to_cstr(((id(*)(id,SEL))objc_msgSend)(e, sel("description"))) : "unknown");
    }
    return ok;
}

void ane_print_model_info(ANEKernel* k) {
    if (!k) return;
    id model = k->model;

    // State
    int state = ((int(*)(id,SEL))objc_msgSend)(model, sel("state"));
    fprintf(stderr, "  state: %d\n", state);

    // Queue depth
    int qd = ((int(*)(id,SEL))objc_msgSend)(model, sel("queueDepth"));
    fprintf(stderr, "  queueDepth: %d\n", qd);

    // PerfStatsMask
    unsigned int mask = ((unsigned int(*)(id,SEL))objc_msgSend)(model, sel("perfStatsMask"));
    fprintf(stderr, "  perfStatsMask: 0x%08X\n", mask);

    // Model attributes
    id attrs = ((id(*)(id,SEL))objc_msgSend)(model, sel("modelAttributes"));
    if (attrs) {
        const char* desc = to_cstr(((id(*)(id,SEL))objc_msgSend)(attrs, sel("description")));
        fprintf(stderr, "  modelAttributes: %s\n", desc);
    } else {
        fprintf(stderr, "  modelAttributes: nil\n");
    }

    // hexStringIdentifier
    id hex = ((id(*)(id,SEL))objc_msgSend)(model, sel("hexStringIdentifier"));
    if (hex) {
        fprintf(stderr, "  hexStringIdentifier: %s\n", to_cstr(hex));
    }

    // isMILModel
    bool isMIL = ((bool(*)(id,SEL))objc_msgSend)(model, sel("isMILModel"));
    fprintf(stderr, "  isMILModel: %s\n", isMIL ? "yes" : "no");

    // programHandle (skip description — it can crash on some types)
    id ph = ((id(*)(id,SEL))objc_msgSend)(model, sel("programHandle"));
    fprintf(stderr, "  programHandle: %s\n", ph ? "present" : "nil");

    // Request info
    id req = k->request;
    id perfStats = ((id(*)(id,SEL))objc_msgSend)(req, sel("perfStats"));
    fprintf(stderr, "  request.perfStats: %s\n", perfStats ? "present" : "nil");

    id perfArr = ((id(*)(id,SEL))objc_msgSend)(req, sel("perfStatsArray"));
    if (perfArr) {
        unsigned long count = ((unsigned long(*)(id,SEL))objc_msgSend)(perfArr, sel("count"));
        fprintf(stderr, "  request.perfStatsArray: %lu entries\n", count);
    } else {
        fprintf(stderr, "  request.perfStatsArray: nil\n");
    }

    // Shared connection info
    id client = ((id(*)(Class,SEL))objc_msgSend)(cls("_ANEClient"), sel("sharedConnection"));
    if (client) {
        fprintf(stderr, "  _ANEClient: connected\n");
        // Check for real-time support
        SEL rt_eval = sel("evaluateRealTimeWithModel:options:request:error:");
        bool has_rt = class_getInstanceMethod(object_getClass(client), rt_eval) != nullptr;
        fprintf(stderr, "  hasRealTimeEval: %s\n", has_rt ? "yes" : "no");

        SEL begin_rt = sel("beginRealTimeTask");
        bool has_begin_rt = class_getInstanceMethod(object_getClass(client), begin_rt) != nullptr;
        fprintf(stderr, "  hasBeginRealTimeTask: %s\n", has_begin_rt ? "yes" : "no");

        // Check for chaining
        SEL chain = sel("prepareChainingWithModel:options:chainingReq:qos:error:");
        bool has_chain = class_getInstanceMethod(object_getClass(client), chain) != nullptr;
        fprintf(stderr, "  hasChaining: %s\n", has_chain ? "yes" : "no");

        // Check for direct eval
        SEL direct = sel("doEvaluateDirectWithModel:options:request:qos:error:");
        bool has_direct = class_getInstanceMethod(object_getClass(client), direct) != nullptr;
        fprintf(stderr, "  hasDirectEval: %s\n", has_direct ? "yes" : "no");

        // Check for buffersReady
        SEL bufReady = sel("buffersReadyWithModel:inputBuffers:options:qos:error:");
        bool has_bufReady = class_getInstanceMethod(object_getClass(client), bufReady) != nullptr;
        fprintf(stderr, "  hasBuffersReady: %s\n", has_bufReady ? "yes" : "no");

        // Check for enqueueSets
        SEL enqueue = sel("enqueueSetsWithModel:outputSet:options:qos:error:");
        bool has_enqueue = class_getInstanceMethod(object_getClass(client), enqueue) != nullptr;
        fprintf(stderr, "  hasEnqueueSets: %s\n", has_enqueue ? "yes" : "no");
    }

    fprintf(stderr, "  nInputs: %d, nOutputs: %d\n", k->nInputs, k->nOutputs);
    for (int i = 0; i < k->nInputs; i++)
        fprintf(stderr, "  input[%d]: %zu bytes\n", i, k->inputBytes[i]);
    for (int i = 0; i < k->nOutputs; i++)
        fprintf(stderr, "  output[%d]: %zu bytes\n", i, k->outputBytes[i]);
}

void ane_set_queue_depth(ANEKernel* k, int depth) {
    if (!k) return;
    ((void(*)(id,SEL,int))objc_msgSend)(k->model, sel("setQueueDepth:"), depth);
}

void ane_set_perf_stats_mask(ANEKernel* k, unsigned int mask) {
    if (!k) return;
    ((void(*)(id,SEL,unsigned int))objc_msgSend)(k->model, sel("setPerfStatsMask:"), mask);
}

} // namespace ane_lm
