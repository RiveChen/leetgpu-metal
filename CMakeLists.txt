cmake_minimum_required(VERSION 3.20)
project(MetalComputeDemos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(leetgpu_metal SHARED
    src/metal_impl.cpp
    src/harness.cpp
)

target_include_directories(leetgpu_metal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp
)

target_link_libraries(leetgpu_metal PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
)