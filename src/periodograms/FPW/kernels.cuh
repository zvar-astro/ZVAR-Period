#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void cudaMakeIndices(const float* times, const float* freqs, int* indices, 
                                const int N_bins, const int N_dat, const int N_freqs);

__global__ void cudaDeltaChi2(float* y, float* ivar, int* indices, float* deltaChiArr,
                              int N_bins, int N_dat, int N_freqs, int N_curves);
                              
__global__ void cudaDeltaChi2Multibin(float* y, float* ivar, int* indices, float* deltaChiArr,
                                      int* N_bins, int N_dat, int N_freqs, int N_curves, int N_samples, int N_bins_total);

#endif