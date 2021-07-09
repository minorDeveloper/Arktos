
add_executable(Eos main.cu)

set_target_properties(Eos PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Eos PROPERTIES CUDA_ARCHITECTURES "35;50;72")
target_link_libraries(Eos PRIVATE Eos_lib)