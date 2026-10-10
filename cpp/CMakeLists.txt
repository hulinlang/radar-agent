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

# B9a：host 采样器单元测试（纯 CPU，无模型依赖）
add_executable(sampler-test tests/sampler_test.cpp)

# B11：int8 激活（Q8_1 对齐）× Q4_K/Q6_K 的 host 参考实现 + 层①对拍
#   ⚠️ -ffp-contract=off 是**语义要求**而非优化选项：B11 的浮点括号化是规格的
#   一部分（docs/34 §3.2.1），GCC 默认 -ffp-contract=fast 会把 a*b+c 收缩成 FMA，
#   从而与 NumPy 参考（和 CUDA 侧的 __fmul_rn/__fadd_rn）产生最后一位差异。
add_executable(int8-ref-test src/gguf.cpp src/int8_act.cpp tests/int8_ref_test.cpp)
target_include_directories(int8-ref-test PRIVATE src)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/int8_act.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

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
# B9b：radar-serve（最小 HTTP + LCP 会话复用）
  add_executable(radar-serve src/gguf.cpp src/tokenizer.cpp src/serve.cpp)
  target_include_directories(radar-serve PRIVATE src)
  target_link_libraries(radar-serve PRIVATE radar_gpu CUDA::cudart)
  add_executable(gpu-quants-test src/gguf.cpp src/quants.cpp tests/gpu_quants_test.cpp)
  target_include_directories(gpu-quants-test PRIVATE src)
  target_link_libraries(gpu-quants-test PRIVATE radar_gpu CUDA::cudart)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu-quants-test PRIVATE OpenMP::OpenMP_CXX)
  endif()
  add_executable(gemv-bench src/gguf.cpp tests/gemv_bench.cpp)
  target_include_directories(gemv-bench PRIVATE src)
  target_link_libraries(gemv-bench PRIVATE radar_gpu CUDA::cudart)
# B11：层①对拍的 CUDA 半边（同一份参考文件，逐元素 bit-exact）
  add_executable(int8-ref-test-cuda src/gguf.cpp src/int8_act.cpp tests/int8_ref_test.cpp)
  target_include_directories(int8-ref-test-cuda PRIVATE src)
  target_compile_definitions(int8-ref-test-cuda PRIVATE USE_CUDA)
  target_link_libraries(int8-ref-test-cuda PRIVATE radar_gpu CUDA::cudart)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/int8_act.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()
  message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (cuda backend enabled, sm_89)")
else()
  message(STATUS "CUDA not found (cuda backend disabled)")
endif()
