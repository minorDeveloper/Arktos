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
    spdlog
    lua)


target_include_directories(
        Arktos
        PRIVATE ${EXTERNALS_DIR}
        PRIVATE ${LUA_DIR} # as of sol2 3.2.2, lua.hpp must be accesible through root include '#include <lua.hpp>'
)

target_compile_features(Arktos PRIVATE cxx_std_17)