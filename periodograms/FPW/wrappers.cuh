#ifndef WRAPPERS_CUH
#define WRAPPERS_CUH

#include "kernels.cuh"
#include "cpu_functions.h"

MatrixXf cudaRunFPW(VectorXf t_vec, MatrixXf y_mat, MatrixXf dy_mat, VectorXf freqs_vec,
                    int N_bins, int N_freqs_batch, int N_curves_batch);

float* cudaRunFPW(VectorXf t_vec, MatrixXf y_mat, MatrixXf dy_mat, VectorXf freqs_vec,
                    int* N_bins, int N_freqs_batch, int N_curves_batch, int N_samples);

#endif // WRAPPERS_CUH