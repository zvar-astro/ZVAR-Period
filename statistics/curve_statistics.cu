#include "curve_statistics.cuh"

__global__ void findArgMax(float* pgram, int* indices, int N_freqs, int N_curves) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N_curves) {
        float max_val = 0.0f;
        int max_idx = -1;
        for (int i = 0; i < N_freqs; i++) {
            float val = pgram[idx * N_freqs + i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        indices[idx] = max_idx;
    }
}

int* findArgMaxWrapper(float* pgram, int N_freqs, int N_curves) {
    float* pgram_device;
    cudaMalloc(&pgram_device, N_curves * N_freqs * sizeof(float));
    cudaMemcpy(pgram_device, pgram, N_curves * N_freqs * sizeof(float), cudaMemcpyHostToDevice);

    int* indices_device;
    cudaMalloc(&indices_device, N_curves * sizeof(int));
    findArgMax<<<(N_curves + 255) / 256, 256>>>(pgram_device, indices_device, N_freqs, N_curves);
    cudaDeviceSynchronize();
    
    int* indices = new int[N_curves];
    cudaMemcpy(indices, indices_device, N_curves * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(pgram_device);
    cudaFree(indices_device);
    return indices;
}

__global__ void findSTD(float* pgram, float* sigma, int N_freqs, int N_curves) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N_curves) {

        //Find the standard deviation of the periodogram
        float sum = 0.0f;
        for (int i = 0; i < N_freqs; i++) {
            sum += pgram[idx * N_freqs + i];
        }
        float mean = sum / N_freqs;
        for (int i = 0; i < N_freqs; i++) {
            sigma[idx] += (pgram[idx * N_freqs + i] - mean) * (pgram[idx * N_freqs + i] - mean);
        }
        sigma[idx] = sqrt(sigma[idx] / N_freqs);
    }
}

float* findSignificanceWrapper(float* pgram, int* ind, int N_freqs, int N_curves) {
    //Find the standard deviation of each periodogram
    float* pgram_device;
    cudaMalloc(&pgram_device, N_curves * N_freqs * sizeof(float));
    cudaMemcpy(pgram_device, pgram, N_curves * N_freqs * sizeof(float), cudaMemcpyHostToDevice);

    float* sigma_device;
    cudaMalloc(&sigma_device, N_curves * sizeof(float));
    findSTD<<<(N_curves + 255) / 256, 256>>>(pgram_device, sigma_device, N_freqs, N_curves);
    cudaDeviceSynchronize();

    float* sigma = new float[N_curves];
    cudaMemcpy(sigma, sigma_device, N_curves * sizeof(float), cudaMemcpyDeviceToHost);

    //Find the maximum power and then significance of each periodogram
    float* significance = new float[N_curves];
    for (int i = 0; i < N_curves; i++) {
        significance[i] = pgram[i * N_freqs + ind[i]] / sigma[i];
    }

    delete[] sigma;
    cudaFree(pgram_device);
    cudaFree(sigma_device);
    return significance;
}