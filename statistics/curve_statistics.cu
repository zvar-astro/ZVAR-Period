#include "curve_statistics.cuh"

__global__ void findArgMax(float* data, int* indices, int N_freqs, int N_curves) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N_curves) {
        float max_val = 0.0f;
        int max_idx = -1;
        for (int i = 0; i < N_freqs; i++) {
            float val = data[idx * N_freqs + i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        indices[idx] = max_idx;
    }
}