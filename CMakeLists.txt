cmake_minimum_required(VERSION 3.24)

project(forgeserve_vector_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")

add_executable(vector_add src/main.cu)
target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
