#include "memory_managers.cuh"

/* Returns the total memory of the host system in bytes*/
size_t getHostMemory() {
    struct sysinfo memInfo;
    if (sysinfo(&memInfo) == 0) {
        return memInfo.totalram * memInfo.mem_unit;
    }
    else {
        std::cerr << "Error: Could not get memory information" << std::endl;
        return 0;
    }
}

size_t getDeviceMemory(int deviceNum) {
    cudaSetDevice(deviceNum);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, deviceNum);
    if (device_prop.totalGlobalMem != 0) {
        size_t bytesGlobal = device_prop.totalGlobalMem;
        return bytesGlobal;
    }
    else {
        std::cerr << "Error: Could not get device memory information" << std::endl;
        return 0;
    }
}
