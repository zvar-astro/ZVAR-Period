#ifndef WRAPPERS_CUH
#define WRAPPERS_CUH

#include "kernels.cuh"
#include "../../msc/diagnostics.h"
#include "../../msc/semaphores.h"
#include <vector>
#include <omp.h>

std::vector<float> cudaRunFPW(float* t_host, float* y_host, float* dy_host, float* freqs_host,
                    int N_dat, unsigned long N_curves, unsigned long N_freqs,
                    int* N_bins, unsigned long N_freqs_batch, unsigned long N_curves_batch, int N_samples);

#endif // WRAPPERS_CUH