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

# B0 骨架：仅冒烟可执行文件；B2 起接入 tokenizer / model
add_executable(radar-infer src/main.cpp)

# B1：GGUF 加载器 + 规范化 dump 工具
add_executable(gguf-dump src/gguf.cpp src/gguf_dump.cpp)
