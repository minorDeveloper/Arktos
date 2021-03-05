find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED GL Shaders Sdl2Application)
find_package(MagnumIntegration REQUIRED ImGui)
find_package(spdlog)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

add_executable(Arktos WIN32
        app.cpp
        Arktos/Core.h
        Arktos/Log.cpp
        Arktos/Log.h)


target_link_libraries(Arktos PRIVATE
    Corrade::Main
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::Shaders
    MagnumIntegration::ImGui
    spdlog)

target_compile_features(Arktos PRIVATE cxx_std_17)