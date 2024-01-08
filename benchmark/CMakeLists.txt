cmake_minimum_required(VERSION 3.25)

set(CMAKE_OSX_ARCHITECTURES arm64;x86_64)

project(simd_benchmark)

add_executable(benchmark benchmark.c ../simd_math.c)

if(LINUX)
    set(CMAKE_EXE_LINKER_FLAGS "-lm")
endif()

if(MSVC)
    target_compile_options(benchmark PRIVATE /O3 /std:c17)
else()
    target_compile_options(benchmark PRIVATE -O3 -mavx2 -mfma)
endif()