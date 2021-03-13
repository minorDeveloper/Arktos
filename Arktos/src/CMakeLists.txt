find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED GL Shaders Sdl2Application)
find_package(MagnumIntegration REQUIRED ImGui)
find_package(spdlog)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

add_executable(Arktos WIN32
        app.cpp
        Arktos/Core.h
        Arktos/Log.cpp
        Arktos/Log.h
        Arktos/Physics/Nbody.cpp
        Arktos/Physics/Nbody.h Arktos/Physics/System.cpp Arktos/Physics/System.h Arktos/Maths/Vec.cpp Arktos/Maths/Vec.h)

target_link_libraries(Arktos PRIVATE
    Corrade::Main
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::Shaders
    MagnumIntegration::ImGui
    spdlog
    lua)

target_include_directories(
        Arktos
        PRIVATE ${EXTERNALS_DIR}
        PRIVATE ${LUA_DIR}
)

target_compile_features(Arktos PRIVATE cxx_std_17)