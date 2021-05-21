cmake_minimum_required (VERSION 2.8.11)
set(CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")
set(CMAKE_CXX_COMPILER "g++")
project (kalman_filter_gpu)
add_executable(kalman src/main.cpp
    src/file.hh
    src/parse.hh)
