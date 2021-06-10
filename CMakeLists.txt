cmake_minimum_required(VERSION 3.0.0)
project(kalman-gpu VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE Debug)


add_executable(kalman-gpu
        src/filter.cpp
        src/segdet.cpp
        src/main.cpp
        src/observation_parser.cu
        src/observation_parser.cpp)

target_link_libraries(kalman-gpu Eigen3::Eigen)
include_directories(include)

get_target_property(OUT kalman-gpu LINK_LIBRARIES)
message(STATUS ${OUT})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
