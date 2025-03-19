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
    // read the number of instances, the total cap percentage, and the chunk cap percentage
    // from the environment variables
    const char* instancesEnv = std::getenv("N_PROCESSES");
    const char* totalCapEnv = std::getenv("TOTAL_CPU_CAP");
    const char* chunkCapEnv = std::getenv("CHUNK_CPU_CAP");
    unsigned long N_INSTANCES = instancesEnv ? std::stoul(instancesEnv) : 1; // Default to 1 if not set
    double totalCap = totalCapEnv ? std::stod(totalCapEnv) : 0.95; // Default to 0.95 if not set
    double chunkCap = chunkCapEnv ? std::stod(chunkCapEnv) : 0.95; // Default to 0.95 if not set

    if (N_INSTANCES == 0) {
        std::cerr << "Error: N_INSTANCES cannot be zero" << std::endl;
        return 0;
    }
    if (totalCap <= 0 || totalCap > 1) {
        std::cerr << "Error: TOTAL_CAP must be between 0 and 1" << std::endl;
        return 0;
    }
    if (chunkCap <= 0 || chunkCap > 1) {
        std::cerr << "Error: CHUNK_CAP must be between 0 and 1" << std::endl;
        return 0;
    }
    size_t hostMemory = getHostMemory(); // Obtain the total host memory
    size_t allowedTotalMemory = static_cast<size_t>(totalCap * static_cast<double>(hostMemory)); // Cap TOTAL usage at 95% of total memory
    size_t allowedInstanceMemory = allowedTotalMemory / N_INSTANCES; // Divide allowed memory by number of instances

    // Decide how many CPU chunks to run based on memory
    size_t singleObjectPeriodogramMemory = N_freqs * N_samples * sizeof(float); // Memory for a single periodogram
    size_t singleObjectDataMemory = NEXP * 2 * sizeof(float); // Memory for a single object's data (flux and fluxErr)
    size_t singleObjectMetaDataMemory = sizeof(float) + sizeof(unsigned long) + 2 * sizeof(float); // Memory for a single object's metadata (ratioValid, psid, ra, dec)
    size_t singleObjectMemory = singleObjectPeriodogramMemory + singleObjectDataMemory + singleObjectMetaDataMemory; // Memory for a single object
    size_t allowedChunkMemory = static_cast<size_t>(chunkCap * static_cast<double>(allowedInstanceMemory)); // Cap CHUNK usage at 95% of instance memory to allow for overhead
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

    // read the total cap percentage from the environment variable
    const char* gpuCapEnv = std::getenv("GPU_CAP");
    double gpuCap = gpuCapEnv ? std::stod(gpuCapEnv) : 0.9; // Default to 0.7 if not set

    // Calculate available memory with safety margin
    size_t available_memory = static_cast<size_t>(gpuCap * static_cast<double>(free_mem));

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
    dim3 blockSizeChi(32, 8);

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

unsigned long calculateNChunkBaseline(int deviceNum, size_t NSRC_PER_CHUNK, size_t N_dat, int modelID) {
    int N_parameters = 0;
    switch (modelID) {
        case 13: // Quadratic model
            N_parameters = 3;
            break;
        case 14: // Cubic model
            N_parameters = 4;
            break;
        case 15: // 7 degree model
            N_parameters = 8;
            break;
        case 16: // 15 degree model
            N_parameters = 16;
            break;
        default:
            throw std::invalid_argument("Invalid model ID");
            break;
    }
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
    size_t parameters_memory = NSRC_PER_CHUNK * N_parameters * float_size;
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

    NCHUNK_BASELINE = 5 * NCHUNK_BASELINE; // Increase the baseline by a factor of 5 to allow for overhead
    
    // Ensure that NCHUNK_BASELINE can't be > NSRC_PER_CHUNK
    // As we couldn't possibly process more sources than we have in a chunk
    if (NCHUNK_BASELINE > NSRC_PER_CHUNK) {
        NCHUNK_BASELINE = NSRC_PER_CHUNK;
    }

    return NCHUNK_BASELINE;
}