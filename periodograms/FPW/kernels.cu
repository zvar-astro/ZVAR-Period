#include "kernels.cuh"

__global__ void cudaMakeIndices(const float* times, const float* freqs, int* indices, 
                                const int N_bins, const int N_dat, const int N_freqs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int freqIdx = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < N_dat && freqIdx < N_freqs) {
        float tf = times[idx] * freqs[freqIdx];
        tf = tf - (int)tf;
        indices[idx * N_freqs + freqIdx] = (int)(tf * N_bins);  // Coalesced access
    }
}

__global__ void cudaDeltaChi2(float* y, float* ivar, int* indices, float* deltaChiArr,
                              int N_bins, int N_dat, int N_freqs, int N_curves) {
    //Allocate shared memory for the intermediate arrays
    extern __shared__ float sharedMemory[];
    float* acc_shared = sharedMemory;
    float* ytCinvV_shared = &sharedMemory[N_bins * blockDim.x * blockDim.y];

    int freq = threadIdx.x + blockIdx.x * blockDim.x;
    int curve = threadIdx.y + blockIdx.y * blockDim.y;

    // Initialize shared memory
    for (int i = 0; i < N_bins; i++) {
        acc_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] = 0.0f;
        ytCinvV_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] = 0.0f;
    }

    if (freq < N_freqs && curve < N_curves) {
        // Update acc and ytCinvV
        for (int i = 0; i < N_dat; i++) {
            int index = indices[i * N_freqs + freq];  // Coalesced access
            float ivar_y = ivar[i + curve * N_dat] * y[i + curve * N_dat];  // Compute once and store in local memory

            acc_shared[index + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] += ivar[i + curve * N_dat];
            ytCinvV_shared[index + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] += ivar_y;
        }

        // Calculate deltaChiArr
        for (int i = 0; i < N_bins; i++) {
            float Minv = __fdividef(1.0f, acc_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] + 1.0f);
            float ytCinvV_current = ytCinvV_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y];
            deltaChiArr[curve * N_freqs + freq] += ytCinvV_current * Minv * ytCinvV_current;
        }
    }
}

__global__ void cudaDeltaChi2Multibin(float* y, float* ivar, int* indices, float* deltaChiArr,
                                      int N_bins, int N_dat, int N_freqs, int N_curves,
                                      int* downsamples, int N_downsamples) {
    //Allocate shared memory for the intermediate arrays
    extern __shared__ float sharedMemory[];
    float* acc_shared = sharedMemory;
    float* ytCinvV_shared = &sharedMemory[N_bins * blockDim.x * blockDim.y];

    int freq = threadIdx.x + blockIdx.x * blockDim.x;
    int curve = threadIdx.y + blockIdx.y * blockDim.y;

    // Initialize shared memory
    for (int i = 0; i < N_bins; i++) {
        acc_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] = 0.0f;
        ytCinvV_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] = 0.0f;
    }

    if (freq < N_freqs && curve < N_curves) {
        // Update acc and ytCinvV
        for (int i = 0; i < N_dat; i++) {
            int index = indices[i * N_freqs + freq];  // Coalesced access
            float ivar_y = ivar[i + curve * N_dat] * y[i + curve * N_dat];  // Compute once and store in local memory

            acc_shared[index + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] += ivar[i + curve * N_dat];
            ytCinvV_shared[index + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] += ivar_y;
        }

        // Calculate deltaChiArr
        for (int i = 0; i < N_bins; i++) {
            float Minv = __fdividef(1.0f, acc_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] + 1.0f);
            float ytCinvV_current = ytCinvV_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y];
            deltaChiArr[curve * N_freqs + freq] += ytCinvV_current * Minv * ytCinvV_current;
        }
    }
}