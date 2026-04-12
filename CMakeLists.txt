cmake_minimum_required(VERSION 3.18)
project(LIGHTWEIGHT_MNIST_OCR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA must be explicitly enabled
enable_language(CUDA)

find_package(pybind11 REQUIRED)

add_library(mnistocr MODULE
    src/logistic.cu
    src/bindings.cu
)

target_link_libraries(mnistocr PRIVATE
    pybind11::module
)

set_target_properties(mnistocr PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CUDA_SEPARABLE_COMPILATION ON
)
