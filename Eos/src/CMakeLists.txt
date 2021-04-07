
set(HEADER_LIST
        "${Eos_SOURCE_DIR}/include/Eos/particle.cuh"
        "${Eos_SOURCE_DIR}/include/Eos/v3.cuh"
        "${Eos_SOURCE_DIR}/include/Eos.cuh")

set(SOURCE_LIST
        "${Eos_SOURCE_DIR}/src/Eos/particle.cu"
        "${Eos_SOURCE_DIR}/src/Eos/v3.cu")

add_library(Eos_lib STATIC
        ${HEADER_LIST} ${SOURCE_LIST}
        )

target_compile_features(Eos_lib PUBLIC cxx_std_11)

set_target_properties(Eos_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(Eos_lib PUBLIC ../include)

