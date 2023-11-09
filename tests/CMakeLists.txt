cmake_minimum_required(VERSION 3.25)

project(simd_unit_tests)

add_executable(test test_simd.c)

target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -Werror -mavx)