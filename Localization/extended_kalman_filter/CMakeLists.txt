cmake_minimum_required(VERSION 3.0)

project(ekf)

message(STATUS "Building ${CMAKE_PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DCMAKE_BUILD_TYPE=Debug)

set(EIGEN_PATH "/Users/sw/Code/eigen-3.4.0/")

include_directories(${EIGEN_PATH})
add_executable(ekf ekf.cpp)