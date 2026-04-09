cmake_minimum_required(VERSION 3.12)
project(LIGHTWEIGHT_MNIST_OCR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Get pybind11 include paths from pip
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Split into separate paths
separate_arguments(PYBIND11_INCLUDES_LIST UNIX_COMMAND "${PYBIND11_INCLUDES_RAW}")

# Remove the leading '-I' from each entry
set(PYBIND11_INCLUDE_DIRS "")
foreach(path ${PYBIND11_INCLUDES_LIST})
    string(REGEX REPLACE "^-I" "" clean_path "${path}")
    list(APPEND PYBIND11_INCLUDE_DIRS "${clean_path}")
endforeach()

# Add include directories
include_directories(${PYBIND11_INCLUDE_DIRS})
include_directories(${Python_INCLUDE_DIRS})

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Build the Python extension
add_library(mnistocr MODULE
    src/logistic.cpp
    src/bindings.cpp 
)

# Correct Python extension suffix for Linux
set_target_properties(mnistocr PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)