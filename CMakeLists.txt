# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.8)

# Project's name
project(ZVAR)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Enable CUDA support
enable_language(CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})
include_directories("/data/swhitebook/Packages/eigen")
include_directories("/data/swhitebook/Packages/Gpufit/Gpufit")

# Add the executable
add_executable(oldExecutable main.cu
               ../periodograms/frequencyGrid.cpp
               ../periodograms/FPW/kernels.cu 
               ../periodograms/FPW/cpu_functions.cpp 
               ../periodograms/FPW/wrappers.cu 
               ../statistics/polyfit.cu
               ../statistics/curve_statistics.cu
               ../statistics/peak_utils.cu
               ../readwrite/hdf5_utils.cpp
               ../readwrite/filepaths.cpp
               ../execution/diagnostics.cpp
               runFile.cu)

# Specify the CUDA standard
set_property(TARGET oldExecutable 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA and cuRAND libraries
target_link_libraries(oldExecutable PRIVATE ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(oldExecutable PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(oldExecutable PRIVATE /data/swhitebook/Packages/Gpufit/build/libGpufit.so)

add_executable(processMetaData ../execution/processMetaData.cu
               ../periodograms/frequencyGrid.cpp
               ../periodograms/FPW/kernels.cu 
               ../periodograms/FPW/cpu_functions.cpp 
               ../periodograms/FPW/wrappers.cu 
               ../statistics/polyfit.cu
               ../statistics/curve_statistics.cu
               ../statistics/peak_utils.cu
               ../readwrite/hdf5_utils.cpp
               ../readwrite/filepaths.cpp
               ../execution/diagnostics.cpp
               runFile.cu)

# Specify the CUDA standard
set_property(TARGET processMetaData
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA and cuRAND libraries
target_link_libraries(processMetaData PRIVATE ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(processMetaData PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(processMetaData PRIVATE /data/swhitebook/Packages/Gpufit/build/libGpufit.so)

add_executable(processSourceChunk ../execution/processSourceChunk.cu
               ../periodograms/frequencyGrid.cpp
               ../periodograms/FPW/kernels.cu 
               ../periodograms/FPW/cpu_functions.cpp 
               ../periodograms/FPW/wrappers.cu 
               ../statistics/polyfit.cu
               ../statistics/curve_statistics.cu
               ../statistics/peak_utils.cu
               ../readwrite/hdf5_utils.cpp
               ../readwrite/filepaths.cpp
               ../execution/diagnostics.cpp
               runFile.cu)

# Specify the CUDA standard
set_property(TARGET processSourceChunk
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA and cuRAND libraries
target_link_libraries(processSourceChunk PRIVATE ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(processSourceChunk PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(processSourceChunk PRIVATE /data/swhitebook/Packages/Gpufit/build/libGpufit.so)

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")