cmake_minimum_required(VERSION 3.0.0)
project(kalman-gpu VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS "-use_fast_math -Xptxas -O3")

include(CTest)
enable_testing()


find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE Debug)

add_executable(kalman-gpu
                src/observation_parser.cu
                src/filter.cpp
                src/segdet.cpp
                src/segdet_gpu.cpp
                src/filter_gpu.cpp
                src/matrix_tools.cpp
                src/main.cpp)

include_directories(include)
get_target_property(OUT kalman-gpu LINK_LIBRARIES)
message(STATUS ${OUT})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
