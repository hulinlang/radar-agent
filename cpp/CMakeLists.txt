cmake_minimum_required(VERSION 3.16)
project(radar_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O2 -Wall -Wextra)
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
