
add_executable(Eos WIN32 main.cu)

set_target_properties(
        Eos
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        Eos
        PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(Eos PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(Eos PRIVATE Eos_Lib)

target_compile_features(Eos PRIVATE cxx_std_17)