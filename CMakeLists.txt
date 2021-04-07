cmake_minimum_required(VERSION 3.17)

project(Arktos)
set(ARKTOS_BUILD_TESTING ON)

include(CTest)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})
add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")
add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:-DDEBUG>")

option(COVERALLS "Generate coveralls data" OFF)

if (COVERALLS)
    include(Coveralls)
    coveralls_turn_on_coverage()
endif()

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

add_definitions(-DCUDA_ENABLED=${CMAKE_CUDA_COMPILER})

if (CMAKE_CUDA_COMPILER)
    message("CUDA Found")
    add_subdirectory(Eos)
endif(CMAKE_CUDA_COMPILER)

add_subdirectory(Arktos)