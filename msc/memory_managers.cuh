#ifndef MEMORY_MANAGERS_H
#define MEMORY_MANAGERS_H

#include <iostream>
#include <sys/sysinfo.h>
#include <cuda_runtime.h>

size_t getHostMemory();

size_t getDeviceMemory(int deviceNum);
#endif //MEMORY_MANAGERS_H