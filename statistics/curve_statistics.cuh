#ifndef CURVE_STATISTICS_CUH
#define CURVE_STATISTICS_CUH

__global__ void findArgMax(float* data, int* indices, int N_freqs, int N_curves);

int* findArgMaxWrapper(float* data, int N_freqs, int N_curves);

__global__ void findSTD(float* pgram, float* sigma, int N_freqs, int N_curves);

float* findSignificanceWrapper(float* pgram, int* ind, int N_freqs, int N_curves);

#endif // CURVE_STATISTICS_CUH