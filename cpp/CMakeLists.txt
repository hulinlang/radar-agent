cmake_minimum_required(VERSION 3.16)
project(radar_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O2> $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
  add_compile_options(-march=native)
endif()

find_package(OpenMP)

# B4：端到端 CLI（分词 + 前向 + greedy 生成）
add_executable(radar-infer src/gguf.cpp src/tokenizer.cpp src/ops.cpp src/quants.cpp src/simd_avx2.cpp src/model.cpp src/main.cpp)

# B1：GGUF 加载器 + 规范化 dump 工具
add_executable(gguf-dump src/gguf.cpp src/gguf_dump.cpp)

# B2：byte-level BPE 分词器 + 对拍测试
add_executable(tokenizer-test src/gguf.cpp src/tokenizer.cpp src/tokenizer_test.cpp)

# B3：CPU 前向（语义基准）+ 逐层对拍
add_executable(forward-test src/gguf.cpp src/ops.cpp src/quants.cpp src/simd_avx2.cpp src/model.cpp src/forward_test.cpp)

# B5：OpenMP 多线程（行分区并行，逐 bit 不变）
if(OpenMP_CXX_FOUND)
  foreach(t radar-infer forward-test)
    target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
  endforeach()
endif()

# B5b：OpenBLAS 对照后端（可选；--backend blas）
find_library(OPENBLAS_LIB openblas)
if(OPENBLAS_LIB)
  target_compile_definitions(radar-infer PRIVATE USE_OPENBLAS)
  target_link_libraries(radar-infer PRIVATE ${OPENBLAS_LIB})
  message(STATUS "OpenBLAS found: ${OPENBLAS_LIB} (blas backend enabled)")
else()
  message(STATUS "OpenBLAS not found (blas backend disabled)")
endif()

# B6b：CUDA 后端（可选；--backend cuda；全部 kernel 手写，decode gemv 为自定义 kernel）
# B7：Q4_K/Q6_K fused dequant-dot gemv kernel（gpu_quants.cu）+ 单元对拍
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit)
  add_library(radar_gpu STATIC src/gpu_model.cu src/gpu_quants.cu)
  set_target_properties(radar_gpu PROPERTIES
    CUDA_STANDARD 17 CUDA_ARCHITECTURES "89" POSITION_INDEPENDENT_CODE ON)
  target_include_directories(radar_gpu PRIVATE src)
  target_link_libraries(radar_gpu PRIVATE CUDA::cudart)
  target_compile_definitions(radar-infer PRIVATE USE_CUDA)
  target_link_libraries(radar-infer PRIVATE radar_gpu CUDA::cudart)
  add_executable(gpu-quants-test src/gguf.cpp src/quants.cpp tests/gpu_quants_test.cpp)
  target_include_directories(gpu-quants-test PRIVATE src)
  target_link_libraries(gpu-quants-test PRIVATE radar_gpu CUDA::cudart)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu-quants-test PRIVATE OpenMP::OpenMP_CXX)
  endif()
  add_executable(gemv-bench src/gguf.cpp tests/gemv_bench.cpp)
  target_include_directories(gemv-bench PRIVATE src)
  target_link_libraries(gemv-bench PRIVATE radar_gpu CUDA::cudart)
  message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (cuda backend enabled, sm_89)")
else()
  message(STATUS "CUDA not found (cuda backend disabled)")
endif()
