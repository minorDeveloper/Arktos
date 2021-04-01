
set(HEADER_LIST
        "${Eos_SOURCE_DIR}/include/Eos/particle.cuh"
        "${Eos_SOURCE_DIR}/include/Eos/v3.cuh")

set(SOURCE_LIST
        "${Eos_SOURCE_DIR}/src/Eos/particle.cu"
        "${Eos_SOURCE_DIR}/src/Eos/v3.cu")

add_library(Eos_Lib STATIC ${HEADER_LIST} ${SOURCE_LIST})

target_include_directories(Eos_Lib PUBLIC ../include)

target_compile_features(Eos_Lib PUBLIC cxx_std_17)

if (COVERALLS)
    set(COVERAGE_SRC ${SOURCE_LIST} ${HEADER_LIST})
    coveralls_setup("${COVERAGE_SRC}" ON)
endif()
