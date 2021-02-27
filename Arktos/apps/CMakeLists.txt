find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED GL Shaders Sdl2Application)
find_package(Ichnea 0.1 REQUIRED)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

add_executable(Arktos WIN32 app.cpp)

target_link_libraries(Arktos PRIVATE
    Corrade::Main
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::Shaders
    Ichnea::Ichnea)

target_compile_features(Arktos PRIVATE cxx_std_17)