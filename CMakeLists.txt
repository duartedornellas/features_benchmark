cmake_minimum_required(VERSION 2.8)
project( features_benchmark )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV 3.3.1 PATHS /home/ddornellas/libs-ddornellas/OpenCV/installed)

add_executable( features_benchmark features_benchmark.cpp )
target_link_libraries( features_benchmark ${OpenCV_LIBS} )
