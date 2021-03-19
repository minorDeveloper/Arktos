cmake_minimum_required(VERSION 3.17)

project(Arktos)
set(ARKTOS_BUILD_TESTING ON)

include(CTest)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})
add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")
add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:-DDEBUG>")

add_subdirectory(Arktos)