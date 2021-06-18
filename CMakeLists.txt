cmake_minimum_required(VERSION 3.0.0)
project(kalman-gpu VERSION 0.1.0 LANGUAGES CXX CUDA)
 
set(CUDA_NVCC_FLAGS "-use_fast_math -Xptxas -O3")

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE Debug)

add_executable(kalman-gpu 
                src/test_gpu.cu
                # src/matrix_tools_gpu.cu
                # src/filter_gpu.cu
                src/filter.cpp 
                src/segdet.cpp
                src/segdet_batch.cpp
                src/filter_batch.cpp
                src/matrix_tools.cpp
                include/filter.hpp
                include/linearregression.hpp
                include/parameter.hpp
                include/segdet.hpp
                include/segment.hpp
                src/main.cpp)

set_property(TARGET kalman-gpu PROPERTY CUDA_ARCHITECTURES 61)

include_directories(include)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
