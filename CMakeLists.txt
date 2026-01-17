cmake_minimum_required(VERSION 3.14)
project(EdgeCortex VERSION 0.1.0 LANGUAGES CXX)

# Force Python 3 discovery
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags for high performance
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/v2.11.1.zip
)
FetchContent_MakeAvailable(pybind11)

# Source files
set(SOURCES
    src/allocator.cpp
    src/bindings.cpp
    src/ops.cpp
)

# Create Python extension module
pybind11_add_module(edgecortex_core ${SOURCES})

# Installation (optional, for structure)
install(TARGETS edgecortex_core DESTINATION lib)
install(DIRECTORY include/edgecortex DESTINATION include)

# Testing (GTest)
enable_testing()
add_subdirectory(tests)
