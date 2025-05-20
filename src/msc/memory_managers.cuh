#ifndef MEMORY_MANAGERS_H
#define MEMORY_MANAGERS_H

#include <iostream>
#include <sys/sysinfo.h>
#include <cuda_runtime.h>

// Store the memory parameters
struct BatchParams {
    size_t N_src_chunk;
    size_t N_src_per_chunk;
    size_t N_freqs_chunk;
    size_t N_freqs_per_chunk;
    size_t N_src_chunk_baseline;
    size_t N_src_per_chunk_baseline;
};

size_t getHostMemory();

size_t getDeviceMemory(int deviceNum);

unsigned long calculateMaxCPUBatch(size_t N_freqs, size_t N_samples, size_t N_exp, size_t N_src);

// unsigned long calculateMaxGPUBatch(int deviceNum, size_t N_dat, size_t N_freqs_batch, size_t N_samples, size_t N_src);

unsigned long calculateNChunkBaseline(int deviceNum, size_t N_src_per_chunk, size_t N_dat, int modelID);

std::pair<size_t, size_t> calculateGPUBatches(int deviceNum, size_t N_dat, size_t N_samples, size_t N_src, size_t N_freqs, size_t N_peaks, float percent_indices, size_t blockDimX, size_t blockDimY);

#endif //MEMORY_MANAGERS_H