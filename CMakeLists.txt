cmake_minimum_required(VERSION 3.14...3.19)

project(Arktos)
set(ARKTOS_BUILD_TESTING ON)

include(CTest)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})

add_subdirectory(Arktos)