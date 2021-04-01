
add_executable(Eos WIN32 main.cu)

target_link_libraries(Eos PRIVATE Eos_Lib)

target_compile_features(Eos PRIVATE cxx_std_17)