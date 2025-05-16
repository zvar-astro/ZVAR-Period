#ifndef WRAPPERS_CUH
#define WRAPPERS_CUH

#include "kernels.cuh"
#include "../../msc/diagnostics.h"
#include "../../msc/semaphores.h"
#include <vector>
#include <omp.h>

unsigned char* createIndices(const float* times, const float* freqs, const int N_bins, const int N_dat, const int N_freqs);

float* runFPWChunk(unsigned char* indices_device, float* y_host, float* dy_host, int N_dat, unsigned long N_src, unsigned long N_freqs, const int* N_bins, int N_samples);

std::vector<float> cudaRunFPW(float* t_host, float* y_host, float* dy_host, float* freqs_host,
                    int N_dat, unsigned long N_curves, unsigned long N_freqs,
                    const int* N_bins, unsigned long N_freqs_batch, unsigned long N_curves_batch, int N_samples);

#endif // WRAPPERS_CUH