#include "kernels.cuh"

__global__ void cudaMakeIndices(const float* times, const float* freqs, unsigned char* indices, 
                                    const int N_bins, const int N_dat, const int N_freqs) {
    int freqIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int timeIdx = threadIdx.y + blockIdx.y * blockDim.y;

    if (timeIdx < N_dat && freqIdx < N_freqs) {
        float tf = times[timeIdx] * freqs[freqIdx];
        tf = tf - (int)tf;
        // indices[timeIdx * N_freqs + freqIdx] = (unsigned char)(tf * N_bins);  // Coalesced access
        indices[freqIdx * N_dat + timeIdx] = (unsigned char)(tf * N_bins);  // Coalesced access
    }
}

__global__ void cudaDeltaChi2(float* y, float* ivar, unsigned char* indices, float* deltaChiArr,
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
            int index = static_cast<int>(indices[freq * N_dat + i]);  // Coalesced access
            float ivar_y = ivar[i + curve * N_dat] * y[i + curve * N_dat];  // Compute once and store in local memory

            acc_shared[index + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] += ivar[i + curve * N_dat];
            ytCinvV_shared[index + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y] += ivar_y;
        }

        // Calculate deltaChiArr
        for (int i = 0; i < N_bins; i++) {
            float Minv = __fdividef(1.0f, 2.0f * acc_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y]);
            // float Minv = __fdividef(1.0f, acc_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y]);
            float ytCinvV_current = ytCinvV_shared[i + N_bins * threadIdx.x + N_bins * blockDim.x * threadIdx.y];
            deltaChiArr[curve * N_freqs + freq] += ytCinvV_current * Minv * ytCinvV_current;
        }
    }
}


__global__ void cudaDeltaChi2Multibin(float* y, float* ivar, const unsigned char* indices, float* deltaChiArr,
                                      int* N_bins, int N_dat, int N_freqs, int N_curves, int N_samples) {

    size_t freqAddress = threadIdx.x;
    size_t curveAddress = N_bins[0] * blockDim.x * threadIdx.y;
    size_t dataAddress = curveAddress + freqAddress;

    // Allocate shared memory for the intermediate arrays
    extern __shared__ float sharedMemory[];
    float* accShared = sharedMemory;
    float* ytCinvVShared = &sharedMemory[N_bins[0] * blockDim.x * blockDim.y];

    // Initialize shared memory
    for (int i = 0; i < N_bins[0]; i++) {
        accShared[i * blockDim.x + dataAddress] = 0.0f;
        ytCinvVShared[i * blockDim.x + dataAddress] = 0.0f;
    }
    
    size_t freq = threadIdx.x + blockIdx.x * blockDim.x;
    size_t curve = threadIdx.y + blockIdx.y * blockDim.y;

    if (freq < (size_t)N_freqs && curve < (size_t)N_curves) {
        // Update acc and ytCinvV for the maximum number of bins
        for (size_t i = 0; i < (size_t)N_dat; i++) {
            size_t index = static_cast<size_t>(indices[freq * N_dat + i]);  // Coalesced access, now using unsigned char
            float ivar_y = ivar[i + curve * N_dat] * y[i + curve * N_dat];  // Compute once and store in local memory

            accShared[index * blockDim.x + dataAddress] += ivar[i + curve * N_dat];
            ytCinvVShared[index * blockDim.x + dataAddress] += ivar_y;
        }

        // Calculate deltaChiArr
        for (size_t i = 0; i < (size_t)N_samples; i++) {
            int downsampleFactor = N_bins[0] / N_bins[i];
            for (int j = 0; j < N_bins[i]; j++) {
                float acc = 0.0f;
                float ytCinvV = 0.0f;
                for (int k = 0; k < downsampleFactor; k++) {
                    int index = j * downsampleFactor + k;
                    acc += accShared[index * blockDim.x + dataAddress];
                    ytCinvV += ytCinvVShared[index * blockDim.x + dataAddress];
                }
                float Minv = __fdividef(1.0f, 2.0f * acc);
                // float Minv = __fdividef(1.0f, acc);
                // size_t idx = freq + curve * (size_t)N_freqs + i * (size_t)N_curves * (size_t)N_freqs;
                size_t idx = freq + i * (size_t)N_freqs + curve * (size_t)N_samples * N_freqs;
                deltaChiArr[idx] += ytCinvV * Minv * ytCinvV;
            }
        }
    }
}