
cmake_minimum_required(VERSION 3.11...3.19)


# Project name and a few useful settings.
project(Arktos
	VERSION 0.1
	DESCRIPTION "Orbital manouever optimiser"
	LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Testing only available if this is the main app
  # Note this needs to be done in the main CMakeLists
  # since it calls enable_testing, which must be in the
  # main CMakeLists.
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
# FetchContent_MakeAvailable was not added until CMake 3.14; use our shim
if(${CMAKE_VERSION} VERSION_LESS 3.14)
  include(cmake/add_FetchContent_MakeAvailable.cmake)
endif()

# Add Corrade as a subproject
add_subdirectory(extern/corrade EXCLUDE_FROM_ALL)

# Add Magnum as a subproject, enable Sdl2Application
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/extern/SDL2-2.0.10 ${CMAKE_PREFIX_PATH})
set(WITH_SDL2APPLICATION ON)
add_subdirectory(extern/magnum EXCLUDE_FROM_ALL)

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(apps)

# Testing only available if this is the main app
# Emergency override ARKTOS_BUILD_TESTING)provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR ARKTOS_BUILD_TESTING)
   AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
