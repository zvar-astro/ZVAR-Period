#ifndef CURVE_STATISTICS_CUH
#define CURVE_STATISTICS_CUH

__global__ void findArgMax(float* data, int* indices, int N_freqs, int N_curves);

#endif // CURVE_STATISTICS_CUH