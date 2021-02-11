find_package(spdlog)

add_executable(Utility app.cpp)
target_link_libraries(Utility PRIVATE spdlog)

target_compile_features(Utility PRIVATE cxx_std_17)