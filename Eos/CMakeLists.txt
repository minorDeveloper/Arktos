
cmake_minimum_required(VERSION 3.17)

# Project name and a few useful settings.
project(Eos
        VERSION 0.1
        DESCRIPTION "CUDA Computational Back-end"
        LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    include(CTest)
    # Docs only available if this is the main app
    find_package(Doxygen)
    if(Doxygen_FOUND)
        add_subdirectory(docs)
    else()
        message(STATUS "Doxygen not found, not building docs")
    endif()
endif()

# FetchContent added in CMake 3.11, downloads during the configure step
include(FetchContent)

set(MSVC2019_COMPATIBILITY ON)

# The compiled library code is here
add_subdirectory(src)

# Final application is here
#add_subdirectory(apps)

# Testing only available if this is the main app
# Emergency override ARKTOS_BUILD_TESTING)provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR ARKTOS_BUILD_TESTING)
        AND BUILD_TESTING)
    message("Testing enabled")
    add_subdirectory(tests)
endif()

