cmake_minimum_required(VERSION 3.25)

set(CMAKE_OSX_ARCHITECTURES arm64;x86_64)

project(simd_unit_tests)

add_executable(test test_simd.c)

if (LINUX)
    target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -Werror -mavx -lm)
else()
    target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -Werror -mavx)
endif()