#ifndef WRAPPERS_CUH
#define WRAPPERS_CUH

#include "kernels.cuh"
#include "cpu_functions.h"
#include "../../msc/diagnostics.h"

MatrixXf cudaRunFPW(VectorXf t_vec, MatrixXf y_mat, MatrixXf dy_mat, VectorXf freqs_vec,
                    int N_bins, int N_freqs_batch, int N_curves_batch);

std::vector<float> cudaRunFPW(float* t_host, float* y_host, float* dy_host, float* freqs_host,
                    int N_dat, int N_curves, int N_freqs,
                    int* N_bins, int N_freqs_batch, int N_curves_batch, int N_samples);

#endif // WRAPPERS_CUH