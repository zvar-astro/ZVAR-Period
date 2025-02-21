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

unsigned long calculateMaxCPUBatch(size_t N_freqs, size_t N_samples, size_t NEXP, size_t NSRC) {
    size_t hostMemory = getHostMemory(); // Obtain the total host memory
    size_t allowedTotalMemory = static_cast<size_t>(0.9 * static_cast<double>(hostMemory)); // Cap TOTAL usage at 95% of total memory
    size_t allowedInstanceMemory = allowedTotalMemory / 8; // Divide allowed memory by number of instances

    // Decide how many CPU chunks to run based on memory
    size_t singleObjectPeriodogramMemory = N_freqs * N_samples * sizeof(float); // Memory for a single periodogram
    size_t singleObjectDataMemory = NEXP * 2 * sizeof(float); // Memory for a single object's data (flux and fluxErr)
    size_t singleObjectMetaDataMemory = sizeof(float) + sizeof(unsigned long) + 2 * sizeof(float); // Memory for a single object's metadata (ratioValid, psid, ra, dec)
    size_t singleObjectMemory = singleObjectPeriodogramMemory + singleObjectDataMemory + singleObjectMetaDataMemory; // Memory for a single object
    size_t allowedChunkMemory = static_cast<size_t>(0.95 * static_cast<double>(allowedInstanceMemory)); // Cap CHUNK usage at 95% of instance memory to allow for overhead
    size_t allowedSourcesMemory = allowedChunkMemory / singleObjectMemory; // Memory for sources in a chunk
    unsigned long NCHUNK = 1 + NSRC / allowedSourcesMemory; // Number of chunks to run

    return NCHUNK;
}

size_t getDeviceMemory(int deviceNum) {
    cudaSetDevice(deviceNum);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, deviceNum);
    if (device_prop.totalGlobalMem != 0) {
        size_t bytesGlobal = device_prop.totalGlobalMem;
        return bytesGlobal;
    } else {
        std::cerr << "Error: Could not get device memory information" << std::endl;
        return 0;
    }
}

unsigned long calculateMaxGPUBatch(int deviceNum, size_t N_dat, size_t N_freqs_batch, size_t N_samples, size_t N_curves) {
    // Memory usage per element in bytes
    const size_t float_size = sizeof(float);
    const size_t int_size = sizeof(int);

    // Get total device memory
    size_t total_mem = getDeviceMemory(deviceNum);
    if (total_mem == 0) {
        return 0; // Error in getting device memory
    }

    // Query available GPU memory
    size_t free_mem, total_mem_query;
    cudaMemGetInfo(&free_mem, &total_mem_query);

    // Calculate available memory with safety margin
    size_t available_memory = static_cast<size_t>(0.70 * static_cast<double>(free_mem));

    // Calculate memory usage for each array
    size_t t_memory = N_dat * float_size;
    size_t y_memory = N_dat * N_curves * float_size;
    size_t ivar_memory = N_dat * N_curves * float_size;
    size_t freqs_memory = N_freqs_batch * float_size;
    size_t indices_memory = N_dat * N_freqs_batch * int_size;
    size_t deltaChiArr_memory = N_freqs_batch * N_samples * float_size;
    size_t N_bins_dev_memory = N_samples * int_size;

    // Calculate total memory usage per curve
    size_t total_memory_per_curve = y_memory + ivar_memory + deltaChiArr_memory;

    // Calculate the maximum N_curves_batch that fits in the available memory
    size_t max_N_curves_batch = available_memory / total_memory_per_curve;

    // Get device properties to check grid size limits
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, deviceNum);

    // Define block size for cudaDeltaChi2Multibin
    dim3 blockSizeChi(64, 8);

    // Calculate maximum grid size for cudaDeltaChi2Multibin
    size_t maxGridSizeY = device_prop.maxGridSize[1];

    // Calculate the maximum number of curves per batch based on grid size limits
    size_t max_curves_per_grid = maxGridSizeY * blockSizeChi.y;

    // Adjust max_N_curves_batch to ensure all allocations fit in available memory and grid size limits
    while (true) {
        size_t total_memory_usage = t_memory + 
                                    max_N_curves_batch * (y_memory + ivar_memory) + 
                                    freqs_memory + 
                                    indices_memory + 
                                    max_N_curves_batch * deltaChiArr_memory + 
                                    N_bins_dev_memory;

        if (total_memory_usage <= available_memory && max_N_curves_batch <= max_curves_per_grid) {
            break;
        }

        if (max_N_curves_batch == 0) {
            std::cerr << "Error: Not enough memory to allocate even a single curve batch" << std::endl;
            return 0;
        }

        max_N_curves_batch--;
    }
    max_N_curves_batch = std::min(max_N_curves_batch, 500ul);

    return max_N_curves_batch;
}

unsigned long calculateNChunkBaseline(int deviceNum, size_t NSRC_PER_CHUNK, size_t N_dat, size_t modelparams) {
    // Memory usage per element in bytes
    const size_t float_size = sizeof(float);
    const size_t int_size = sizeof(int);

    // Get total device memory
    size_t total_mem = getDeviceMemory(deviceNum);
    if (total_mem == 0) {
        return 0; // Error in getting device memory
    }

    // Query available GPU memory
    size_t free_mem, total_mem_query;
    cudaMemGetInfo(&free_mem, &total_mem_query);

    // Calculate available memory with a conservative buffer (e.g., 70% of free memory)
    size_t available_memory = static_cast<size_t>(0.70 * static_cast<double>(free_mem));

    // Calculate memory usage for Gpufit arrays
    size_t times_memory = N_dat * float_size;
    size_t y_memory = N_dat * NSRC_PER_CHUNK * float_size;
    size_t dy_memory = N_dat * NSRC_PER_CHUNK * float_size;
    size_t parameters_memory = NSRC_PER_CHUNK * modelparams * float_size;
    size_t states_memory = NSRC_PER_CHUNK * int_size;
    size_t chi_squares_memory = NSRC_PER_CHUNK * float_size;
    size_t n_iterations_memory = NSRC_PER_CHUNK * int_size;

    // Calculate total memory usage per chunk
    size_t total_memory_per_chunk = times_memory + y_memory + dy_memory + parameters_memory + states_memory + chi_squares_memory + n_iterations_memory;

    // Estimate NCHUNK_BASELINE
    unsigned long NCHUNK_BASELINE = 1;
    while (true) {
        size_t total_memory_usage = NCHUNK_BASELINE * total_memory_per_chunk;
        if (total_memory_usage <= available_memory) {
            break;
        }
        NCHUNK_BASELINE++;
    }

    // Ensure that all objects are processed
    if (NCHUNK_BASELINE * NSRC_PER_CHUNK < N_dat) {
        NCHUNK_BASELINE = (N_dat + NSRC_PER_CHUNK - 1) / NSRC_PER_CHUNK;
    }

    return 5 * NCHUNK_BASELINE;
}