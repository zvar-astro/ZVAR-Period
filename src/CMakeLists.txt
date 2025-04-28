# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.9)

# Project's name
project(cudaFPW)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Enable CUDA support
enable_language(CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

find_package(OpenMP REQUIRED)

# set(HDF5_ROOT /usr/local)
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

#Add Catch2 subdirectory
add_subdirectory($ENV{CATCH2_DIR} Catch2-build)

# Add the executable
add_executable(cudaFPW main.cu
               periodograms/frequencyGrid.cpp
               periodograms/FPW/kernels.cu 
               periodograms/FPW/wrappers.cu 
               statistics/polyfit.cu
               statistics/curve_statistics.cu
               statistics/peak_utils.cu
               readwrite/hdf5_utils.cpp
               readwrite/filepaths.cpp
               msc/diagnostics.cpp
               msc/semaphores.cpp
               msc/memory_managers.cu
               processing.cu
               )

add_executable(tests ../tests/test_main.cu
               ../tests/test_utils.cu
               periodograms/FPW/wrappers.cu
               periodograms/FPW/kernels.cu
               statistics/peak_utils.cu
               statistics/polyfit.cu)

# Specify the CUDA standard
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
set_property(TARGET cudaFPW 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA and cuRAND libraries
target_link_libraries(cudaFPW PRIVATE ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(cudaFPW PRIVATE ${HDF5_LIBRARIES})

target_link_libraries(tests PRIVATE ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# Include Eigen headers
include_directories($ENV{EIGEN3_INCLUDE_DIR})
# Include Gpufit headers
include_directories($ENV{GPUFIT_INCLUDE_DIR})

# Include the path to the Gpufit library that was built
# target_link_libraries(cudaFPW PRIVATE /data/dev/ZVAR-Period/data/Gpufit-build/Gpufit/libGpufit.so)
# instead use an environment variable
target_link_libraries(cudaFPW PRIVATE $ENV{GPUFIT_LIB_DIR}/libGpufit.so)
target_link_libraries(tests PRIVATE $ENV{GPUFIT_LIB_DIR}/libGpufit.so)
if(OpenMP_CXX_FOUND)
    target_compile_options(cudaFPW PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    target_link_libraries(cudaFPW PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link Catch2 to the test executable
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_include_directories(tests PRIVATE $ENV{CATCH2_DIR}/../src)

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")