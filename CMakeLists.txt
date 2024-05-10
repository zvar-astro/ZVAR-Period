# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.8)

# Project's name
project(cudaFPW)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Enable CUDA support
enable_language(CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add the executable
add_executable(cudaFPW main.cu ../periodograms/FPW/kernels.cu ../periodograms/FPW/cpu_functions.cpp ../periodograms/FPW/wrappers.cu)

# Specify the CUDA standard
set_property(TARGET cudaFPW 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA and cuRAND libraries
target_link_libraries(cudaFPW PRIVATE ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# Include Eigen headers
include_directories("/home/swhitebook/other-packages/eigen-3.4.0")
include_directories("/home/swhitebook/other-packages/Gpufit/Gpufit")

target_link_libraries(cudaFPW PRIVATE /home/swhitebook/other-packages/Gpufit/build/Gpufit/libGpufit.so)

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
