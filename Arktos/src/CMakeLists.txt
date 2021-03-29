find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED GL Shaders Sdl2Application)
find_package(MagnumIntegration REQUIRED ImGui)
find_package(spdlog)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

set(HEADER_LIST
    "${Arktos_SOURCE_DIR}/include/Arktos.h"
    "${Arktos_SOURCE_DIR}/include/Arktos/Core.h"
    "${Arktos_SOURCE_DIR}/include/Arktos/Log.h"
    "${Arktos_SOURCE_DIR}/include/Arktos/Physics/Nbody.h"
    "${Arktos_SOURCE_DIR}/include/Arktos/Maths/Vec.h")

set(SOURCE_LIST
    "${Arktos_SOURCE_DIR}/src/Arktos.cpp"
    "${Arktos_SOURCE_DIR}/src/Arktos/Log.cpp"
    "${Arktos_SOURCE_DIR}/src/Arktos/Physics/Nbody.cpp"
    "${Arktos_SOURCE_DIR}/src/Arktos/Physics/System.cpp"
    "${Arktos_SOURCE_DIR}/src/Arktos/Maths/Vec.cpp")

add_library(Arktos_Lib STATIC ${HEADER_LIST} ${SOURCE_LIST})

target_include_directories(
        Arktos_Lib
        PUBLIC ${LUA_EXTERNALS_DIR}
        PUBLIC ${LUA_DIR}
)

target_include_directories(Arktos_Lib PUBLIC ../include)

target_link_libraries(Arktos_Lib PUBLIC
    Corrade::Main
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::Shaders
    MagnumIntegration::ImGui
    spdlog
    lua)

target_compile_features(Arktos_Lib PUBLIC cxx_std_17)

if (COVERALLS)
    set(COVERAGE_SRC ${SOURCE_LIST} ${HEADER_LIST})
    coveralls_setup("${COVERAGE_SRC}" ON)
endif()
