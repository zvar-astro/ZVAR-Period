#ifndef MEMORY_MANAGERS_H
#define MEMORY_MANAGERS_H

#include <iostream>
#include <sys/sysinfo.h>
#include <cuda_runtime.h>

size_t getHostMemory();

unsigned long calculateMaxCPUBatch(size_t N_freqs, size_t N_samples, size_t NEXP, size_t NSRC);

size_t getDeviceMemory(int deviceNum);

unsigned long calculateMaxGPUBatch(int deviceNum, size_t N_dat, size_t N_freqs_batch, size_t N_samples, size_t N_curves);

unsigned long calculateNChunkBaseline(int deviceNum, size_t NSRC_PER_CHUNK, size_t N_dat, int modelID);
#endif //MEMORY_MANAGERS_H