find_package(spdlog)

# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

################################################################################
# Source groups
################################################################################
set(Header_Files
    "${Ichnea_SOURCE_DIR}/include/Ichnea.h"
    "${Ichnea_SOURCE_DIR}/include/Ichnea/core.h"
    "${Ichnea_SOURCE_DIR}/include/Ichnea/EntryPoint.h"
    "${Ichnea_SOURCE_DIR}/include/Ichnea/Log.h"
)

set(Source_Files
    "${Ichnea_SOURCE_DIR}/src/Ichnea/Log.cpp"
)

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

# Make an automatic library - will be static or dynamic based on user setting
add_library(Ichnea STATIC ${ALL_FILES})

# We need this directory, and users of our library will need it too
target_include_directories(Ichnea PUBLIC ../include)
target_link_libraries(Ichnea PRIVATE spdlog)

# All users of this library will need at least C++11
target_compile_features(Ichnea PUBLIC cxx_std_11)

# IDEs should put the headers in a nice place
