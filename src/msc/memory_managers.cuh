#ifndef MEMORY_MANAGERS_H
#define MEMORY_MANAGERS_H

#include <iostream>
#include <sys/sysinfo.h>
#include <cuda_runtime.h>

// Store the memory parameters
struct BatchParams {
    size_t N_chunk_CPU; // Number of CPU chunks to process
    size_t N_src_per_chunk_GPU;
    size_t N_src_per_chunk_GPU_baseline;

};

size_t getHostMemory();

size_t getDeviceMemory(int deviceNum);

unsigned long calculateMaxCPUBatch(size_t N_freqs, size_t N_samples, size_t N_exp, size_t N_src);

unsigned long calculateMaxGPUBatch(int deviceNum, size_t N_dat, size_t N_freqs_batch, size_t N_samples, size_t N_src);

unsigned long calculateNChunkBaseline(int deviceNum, size_t N_src_per_chunk, size_t N_dat, int modelID);

struct BatchParams calculateAllBatching(int deviceNum, size_t N_dat, size_t N_freqs, size_t N_samples, size_t N_src, int modelID);

#endif //MEMORY_MANAGERS_H