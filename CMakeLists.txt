cmake_minimum_required(VERSION 3.20)
project(ane.cpp VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS deployment target
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target")

# Use new Accelerate LAPACK API
add_compile_definitions(ACCELERATE_NEW_LAPACK)

# Fetch dependencies
include(FetchContent)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(json tokenizers_cpp)

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# Source files
set(VENDOR_SOURCES
    vendor/jinja/value.cpp
    vendor/jinja/string.cpp
    vendor/jinja/lexer.cpp
    vendor/jinja/parser.cpp
    vendor/jinja/runtime.cpp
    vendor/jinja/caps.cpp
)

set(CORE_SOURCES
    common.cpp
    core/safetensors.cpp
    core/model_loader.cpp
    core/weight_cache.cpp
    core/tokenizer.cpp
    core/cpu_ops.cpp
    core/sampling.cpp
    core/ane_runtime.cpp
    core/metal_gemm.mm
    utils.cpp
    generate.cpp
)

set(MODEL_SOURCES
    models/llm/qwen3_5.cpp
    models/llm/qwen3.cpp
)

set(MAIN_SOURCE
    main.cpp
)

# Compile Metal shaders to metallib
set(METAL_SOURCE "${CMAKE_SOURCE_DIR}/core/kernels.metal")
set(METALLIB_OUTPUT "${CMAKE_BINARY_DIR}/kernels.metallib")
set(METAL_AIR "${CMAKE_BINARY_DIR}/kernels.air")

add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND xcrun -sdk macosx metal -c -target air64-apple-macos13.0 -o ${METAL_AIR} ${METAL_SOURCE}
    COMMAND xcrun -sdk macosx metallib -o ${METALLIB_OUTPUT} ${METAL_AIR}
    DEPENDS ${METAL_SOURCE}
    COMMENT "Compiling Metal shaders → kernels.metallib"
)
add_custom_target(metal_shaders DEPENDS ${METALLIB_OUTPUT})

# Executable
add_executable(ane_cpp
    ${VENDOR_SOURCES}
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${MAIN_SOURCE}
)
add_dependencies(ane_cpp metal_shaders)

# Pass metallib path as compile definition
target_compile_definitions(ane_cpp PRIVATE
    METALLIB_PATH="${METALLIB_OUTPUT}"
)

set_target_properties(ane_cpp PROPERTIES OUTPUT_NAME "ane.cpp")

# Link frameworks and ObjC runtime
target_link_libraries(ane_cpp PRIVATE
    nlohmann_json::nlohmann_json
    tokenizers_cpp
    "-framework Foundation"
    "-framework IOSurface"
    "-framework Accelerate"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    "-lobjc"
)

# Optimization for release
target_compile_options(ane_cpp PRIVATE
    $<$<CONFIG:Release>:-O3 -ffast-math>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Test target for quantized weight benchmarks (minimal deps)
add_executable(test_quant_perf EXCLUDE_FROM_ALL
    ../apple-neural-engine-field-guide/tests/test_quantized_weight_perf_v2.cpp
    core/ane_runtime.cpp
    common.cpp
)
target_link_libraries(test_quant_perf PRIVATE
    "-framework Foundation"
    "-framework IOSurface"
    "-framework Accelerate"
    "-lobjc"
)
target_compile_options(test_quant_perf PRIVATE -O3)

# INT4 benchmark test
add_executable(test_int4_perf EXCLUDE_FROM_ALL
    ../apple-neural-engine-field-guide/tests/test_int4_weight_perf.cpp
    core/ane_runtime.cpp
    common.cpp
)
target_link_libraries(test_int4_perf PRIVATE
    "-framework Foundation"
    "-framework IOSurface"
    "-framework Accelerate"
    "-lobjc"
)
target_compile_options(test_int4_perf PRIVATE -O3)
