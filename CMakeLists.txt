# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.9)

# Project's name
project(cudaFPW)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Enable CUDA support
enable_language(CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

find_package(OpenMP REQUIRED)

# set(HDF5_ROOT /usr/local)
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# Add the executable
add_executable(cudaFPW main.cu
               ../periodograms/frequencyGrid.cpp
               ../periodograms/FPW/kernels.cu 
               ../periodograms/FPW/cpu_functions.cpp 
               ../periodograms/FPW/wrappers.cu 
               ../statistics/polyfit.cu
               ../statistics/curve_statistics.cu
               ../statistics/peak_utils.cu
               ../readwrite/hdf5_utils.cpp
               ../readwrite/filepaths.cpp
               ../msc/diagnostics.cpp
               runFile.cu)

# Specify the CUDA standard
set(CMAKE_BUILD_TYPE Release)
set_property(TARGET cudaFPW 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA and cuRAND libraries
target_link_libraries(cudaFPW PRIVATE ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(cudaFPW PRIVATE ${HDF5_LIBRARIES})

# Include Eigen headers
include_directories("/data/swhitebook/Packages/eigen")
include_directories("/data/swhitebook/Packages/Gpufit/Gpufit")

target_link_libraries(cudaFPW PRIVATE /data/swhitebook/Packages/Gpufit/build/libGpufit.so)
if(OpenMP_CXX_FOUND)
    target_compile_options(cudaFPW PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    target_link_libraries(cudaFPW PRIVATE OpenMP::OpenMP_CXX)
endif()

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")