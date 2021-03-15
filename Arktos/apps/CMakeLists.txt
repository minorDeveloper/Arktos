
add_executable(Arktos_App WIN32 app.cpp)

target_link_libraries(Arktos_App PRIVATE Arktos_Lib)

target_compile_features(Arktos_App PRIVATE cxx_std_17)