#ifndef WRAPPERS_CUH
#define WRAPPERS_CUH

#include "kernels.cuh"
#include "cpu_functions.h"

MatrixXf cudaRunFPW(VectorXf t_vec, MatrixXf y_mat, MatrixXf dy_mat, VectorXf freqs_vec,
                    int N_bins, int N_freqs_batch, int N_curves_batch);

MatrixXf cudaRunFPW(VectorXf t_vec, MatrixXf y_mat, MatrixXf dy_mat, VectorXf freqs_vec,
                    int N_bins, int N_freqs_batch, int N_curves_batch, int* downsamples, int N_downsamples);

#endif // WRAPPERS_CUH