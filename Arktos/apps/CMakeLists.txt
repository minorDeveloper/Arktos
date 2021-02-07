find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED GL Shaders Sdl2Application)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

add_executable(app WIN32 app.cpp)
target_link_libraries(app PRIVATE
    Corrade::Main
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::Shaders)

target_compile_features(app PRIVATE cxx_std_17)

# Make the executable a default target to build & run in Visual Studio
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MyApplication)