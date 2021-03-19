
add_executable(Arktos WIN32 app.cpp)

target_link_libraries(Arktos PRIVATE Arktos_Lib lua)

target_compile_features(Arktos PRIVATE cxx_std_17)