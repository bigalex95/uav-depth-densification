cmake_minimum_required(VERSION 3.22)
project(depth_densifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DD_WERROR "Treat compiler warnings as errors" OFF)
add_compile_options(-Wall -Wextra)
if(DD_WERROR)
    add_compile_options(-Werror)
endif()

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(TIFF REQUIRED)

# Library target
add_library(depth_densifier_lib
    src/depth_densifier.cpp
)

set_target_properties(depth_densifier_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(depth_densifier_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(depth_densifier_lib
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)

# Test executable
add_executable(depth_densifier_test
    tests/depth_densifier_test.cpp
)

target_link_libraries(depth_densifier_test
    PRIVATE
        depth_densifier_lib
        GTest::gtest_main
)

# Main densification executable
add_executable(densify_main
    src/densify_main.cpp
)

target_link_libraries(densify_main
    PRIVATE
        depth_densifier_lib
        TIFF::TIFF
)

# Enable testing
enable_testing()
include(GoogleTest)
gtest_discover_tests(depth_densifier_test)
