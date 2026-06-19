cmake_minimum_required(VERSION 3.22.1)
project(expression_postprocess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NOTE: do NOT enable -ffast-math; IEEE float ordering must match PC parity.
add_compile_options(-O2)

add_library(expression_postprocess SHARED
    expression_postprocess.cpp
    jni_bridge.cpp)

find_library(log-lib log)
target_link_libraries(expression_postprocess ${log-lib})
