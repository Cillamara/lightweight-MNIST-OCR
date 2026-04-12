cmake_minimum_required(VERSION 3.18)

# Point CMake at CUDA_HOME if set (e.g. MATLAB-bundled nvcc)
if(DEFINED ENV{CUDA_HOME})
    set(CMAKE_CUDA_COMPILER $ENV{CUDA_HOME}/bin/nvcc)
    set(CUDAToolkit_ROOT $ENV{CUDA_HOME})
    set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT $ENV{CUDA_HOME})
endif()

project(LIGHTWEIGHT_MNIST_OCR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

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
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
