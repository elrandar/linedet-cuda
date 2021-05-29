cmake_minimum_required(VERSION 3.0.0)
project(kalman-gpu VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Eigen3 REQUIRED NO_MODULE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_executable(kalman-gpu 
                src/filter.cpp 
                src/segdet.cpp
                src/main.cpp
                include/filter.hpp
                include/linearregression.hpp
                include/parameter.hpp
                include/segdet.hpp
                include/segment.hpp)

target_link_libraries(kalman-gpu Eigen3::Eigen)

get_target_property(OUT kalman-gpu LINK_LIBRARIES)
message(STATUS ${OUT})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
