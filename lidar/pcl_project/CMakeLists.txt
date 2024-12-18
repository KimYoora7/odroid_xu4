cmake_minimum_required(VERSION 3.10)
project(pcl_project)

find_package(PCL REQUIRED)

add_executable(pcl_project main.cpp)

target_link_libraries(pcl_project ${PCL_LIBRARIES})
include_directories(${PCL_INCLUDE_DIRS})

