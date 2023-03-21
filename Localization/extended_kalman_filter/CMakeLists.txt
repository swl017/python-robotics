cmake_minimum_required(VERSION 3.0)

project(ekf)

# include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DCMAKE_BUILD_TYPE=Debug)

add_subdirectory(matplotlibcpp)

set(EIGEN_PATH "/Users/sw/Code/eigen-3.4.0/")

include_directories(${EIGEN_PATH})
add_executable(ekf ekf.cpp)
target_link_libraries(ekf matplotlib_cpp)

add_executable(test test.cpp)
target_link_libraries(test matplotlib_cpp)