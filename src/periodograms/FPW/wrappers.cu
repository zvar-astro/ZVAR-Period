#include "wrappers.cuh"

unsigned char* createIndices(const float* times, const float* freqs, const int N_bins, const int N_dat, const int N_freqs) {

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "unexpected CUDA error (indices): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate memory for device arrays
    float* times_device;
    float* freqs_device;
    unsigned char* indices_device;
    
    error = cudaMalloc(&times_device, N_dat * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc t: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&freqs_device, N_freqs * sizeof(float));  // Allocate memory for frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc freqs: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&indices_device, N_dat * N_freqs * sizeof(unsigned char));  // Allocate memory for indices for all frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc indices: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy data from host to device
    error = cudaMemcpy(times_device, times, N_dat * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy t: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(freqs_device, freqs, N_freqs * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy freqs: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 blockSizeInd(16, 16);  // Use a 2D array of threads in each block
    // dim3 gridSizeInd((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs + blockSizeInd.y - 1) / blockSizeInd.y);
    dim3 gridSizeInd((N_freqs + blockSizeInd.x - 1) / blockSizeInd.x, (N_dat + blockSizeInd.y - 1) / blockSizeInd.y);

    std::cout << "Grid size: " << gridSizeInd.x << " " << gridSizeInd.y << std::endl;

    cudaMakeIndices<<<gridSizeInd, blockSizeInd>>>(times_device, freqs_device, indices_device, N_bins, N_dat, N_freqs);

    //Clean up device memory
    error = cudaFree(times_device);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree t: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaFree(freqs_device);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree freqs: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    return indices_device;
}

float* runFPWChunk(unsigned char* indices_device, float* y_host, float* dy_host, int N_dat, unsigned long N_src, unsigned long N_freqs, const int* N_bins, int N_samples) {

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "unexpected CUDA error (wrappers): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate memory for device arrays
    float* y_device;
    float* ivar_device;
    float* periodograms_device;
    int* N_bins_device;
    
    error = cudaMalloc(&y_device, N_dat * N_src * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc y: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&ivar_device, N_dat * N_src * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc ivar: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&periodograms_device, N_freqs * N_src * N_samples * sizeof(float));  // Allocate memory for deltaChiArr for all frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc deltaChiArr: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&N_bins_device, N_samples * sizeof(int));  // Allocate memory for N_bins
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc N_bins_device: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    //Print the size of periodograms_device
    std::cout << "Size of periodograms_device: " << N_freqs * N_src * N_samples << std::endl;

    // Copy data from host to device
    error = cudaMemcpy(N_bins_device, N_bins, N_samples * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy N_bins_device: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Define block size and grid size for kernel launches
    dim3 blockSizeChi(32, 8);  // Use a 2D array of threads in each block
    dim3 gridSizeChi((N_freqs + blockSizeChi.x - 1) / blockSizeChi.x, (N_src + blockSizeChi.y - 1) / blockSizeChi.y);
    size_t sharedMemoryChi = 2 * N_bins[0] * blockSizeChi.x * blockSizeChi.y * sizeof(float);

    //print the size of the grid
    std::cout << "Grid size: " << gridSizeChi.x << " " << gridSizeChi.y << std::endl;
    std::cout << "Shared memory size: " << sharedMemoryChi / (1024.0) << " KB" << std::endl;

    // Copy y and ivar to device
    float* ivar_host = new float[N_dat * N_src];
    for (unsigned long i = 0; i < N_src * N_dat; i++) {
        ivar_host[i] = 1.0 / (dy_host[i] * dy_host[i]);
    }

    error = cudaMemsetAsync(periodograms_device, 0, N_freqs * N_src * N_samples * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemsetAsync periodograms_device: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(y_device, y_host, N_dat * N_src * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy y (wrappers): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(ivar_device, ivar_host, N_dat * N_src * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy ivar (wrappers): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error BEFORE cudaDeltaChi2Multibin (wrappers): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Check how much memory is allocated
    size_t freeMem, totalMem;
    error = cudaMemGetInfo(&freeMem, &totalMem);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemGetInfo: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Print the fraction of used memory

    std::cout << "Used memory: " << (totalMem - freeMem) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Used Fraction: " << (static_cast<double>(totalMem - freeMem) / static_cast<double>(totalMem)) * 100.0 << " %" << std::endl;

    cudaDeltaChi2Multibin<<<gridSizeChi, blockSizeChi, sharedMemoryChi>>>(
                            y_device, ivar_device, indices_device, periodograms_device, N_bins_device, N_dat, N_freqs, N_src, N_samples);

    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error DURING cudaDeltaChi2Multibin (wrappers): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }


    // Free device memory
    error = cudaFree(y_device);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree y: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaFree(ivar_device);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree ivar: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaFree(N_bins_device);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree N_bins_device: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Free host memory
    delete[] ivar_host;

    return periodograms_device;
}

/*Multibin FPW. Instead of passing a single integer value of N_bins we pass an array of N_bins values starting with the largest value.
The following values must be factors of N_bins[0]. N_samples is then the length of N_bins and determines how much memory must be allocated.*/
std::vector<float> cudaRunFPW(float* t_host, float* y_host, float* dy_host, float* freqs_host,
                    int N_dat, unsigned long N_curves, unsigned long N_freqs,
                    const int* N_bins, unsigned long N_freqs_batch, unsigned long N_src_batch, int N_samples) {

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "unexpected CUDA error (wrappers): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate memory for device arrays
    float* t;
    float* y;
    float* ivar;
    float* freqs;
    unsigned char* indices;
    float* deltaChiArr;
    int* N_bins_device;
    
    error = cudaMalloc(&t, N_dat * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc t: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&y, N_dat * N_src_batch * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc y: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&ivar, N_dat * N_src_batch * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc ivar: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&freqs, N_freqs_batch * sizeof(float));  // Allocate memory for frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc freqs: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&indices, N_dat * N_freqs_batch * sizeof(unsigned char));  // Allocate memory for indices for all frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc indices: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&deltaChiArr, N_freqs_batch * N_src_batch * N_samples * sizeof(float));  // Allocate memory for deltaChiArr for all frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc deltaChiArr: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&N_bins_device, N_samples * sizeof(int));  // Allocate memory for N_bins
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc N_bins_device: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy data from host to device
    error = cudaMemcpy(t, t_host, N_dat * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy t: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
        
    }
    error = cudaMemcpy(N_bins_device, N_bins, N_samples * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy N_bins_device: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Define block size and grid size for kernel launches
    dim3 blockSizeInd(16, 16);  // Use a 2D array of threads in each block
    // dim3 gridSizeInd((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs_batch + blockSizeInd.y - 1) / blockSizeInd.y);
    dim3 gridSizeInd((N_freqs_batch + blockSizeInd.x - 1) / blockSizeInd.x, (N_dat + blockSizeInd.y - 1) / blockSizeInd.y);
    dim3 blockSizeChi(32, 8);  // Use a 2D array of threads in each block
    dim3 gridSizeChi((N_freqs_batch + blockSizeChi.x - 1) / blockSizeChi.x, (N_src_batch + blockSizeChi.y - 1) / blockSizeChi.y);
    size_t sharedMemoryChi = 2 * N_bins[0] * blockSizeChi.x * blockSizeChi.y * sizeof(float);

    // Allocate memory for host array to hold the result
    float* deltaChi_host = new float[N_src_batch * N_freqs_batch * N_samples];

    // Create a matrix to store the periodograms for each light curve
    std::vector<float> periodograms(N_curves * N_freqs * N_samples);

    // Copy y and ivar to device
    float* ivar_host = new float[N_dat * N_curves];
    for (unsigned long i = 0; i < N_curves * N_dat; i++) {
        ivar_host[i] = 1.0 / (dy_host[i] * dy_host[i]);
    }

    // Run kernels for each batch of frequencies over all light curves
    for (unsigned long i = 0; i < N_freqs; i += N_freqs_batch) {
        // std::cout << "Processing frequency batch " << i / N_freqs_batch + 1 << " of " << (N_freqs + N_freqs_batch - 1) / N_freqs_batch << std::endl;
        unsigned long N_freqs_current = std::min(N_freqs_batch, N_freqs - i);

        error = cudaMemcpyAsync(freqs, freqs_host + i, N_freqs_current * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in cudaMemcpy freqs (wrappers): " << cudaGetErrorString(error) << std::endl;
            exit(EXIT_FAILURE);
        }

        // dim3 gridSizeInd_current((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs_current + blockSizeInd.y - 1) / blockSizeInd.y);
        dim3 gridSizeInd_current((N_freqs_current + blockSizeInd.x - 1) / blockSizeInd.x, (N_dat + blockSizeInd.y - 1) / blockSizeInd.y);

        cudaMakeIndices<<<gridSizeInd_current, blockSizeInd>>>(t, freqs, indices, N_bins[0], N_dat, N_freqs_current);

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in cudaMakeIndices (wrappers): " << cudaGetErrorString(error) << std::endl;
            exit(EXIT_FAILURE);
        }

        for (unsigned long j = 0; j < N_curves; j += N_src_batch) {
            unsigned long N_curves_current = std::min(N_src_batch, N_curves - j);

            error = cudaMemsetAsync(deltaChiArr, 0, N_freqs_current * N_curves_current * N_samples * sizeof(float));
            if (error != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemsetAsync deltaChiArr: " << cudaGetErrorString(error) << std::endl;
                exit(EXIT_FAILURE);
            }

            error = cudaMemcpy(y, y_host + j * N_dat, N_dat * N_curves_current * sizeof(float), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemcpy y (wrappers): " << cudaGetErrorString(error) << std::endl;
                exit(EXIT_FAILURE);
            }
            error = cudaMemcpy(ivar, ivar_host + j * N_dat, N_dat * N_curves_current * sizeof(float), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemcpy ivar (wrappers): " << cudaGetErrorString(error) << std::endl;
                exit(EXIT_FAILURE);
            }

            error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error BEFORE cudaDeltaChi2Multibin (wrappers): " << cudaGetErrorString(error) << std::endl;
                exit(EXIT_FAILURE);
            }

            cudaDeltaChi2Multibin<<<gridSizeChi, blockSizeChi, sharedMemoryChi>>>(
                                    y, ivar, indices, deltaChiArr, N_bins_device, N_dat, N_freqs_current, N_curves_current, N_samples);

            error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error AFTER cudaDeltaChi2Multibin (wrappers): " << cudaGetErrorString(error) << std::endl;
                exit(EXIT_FAILURE);
            }

            error = cudaMemcpyAsync(deltaChi_host, deltaChiArr, N_freqs_current * N_curves_current * N_samples * sizeof(float), cudaMemcpyDeviceToHost);

            if (error != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemcpyAsync deltaChiArr (wrappers): " << cudaGetErrorString(error) << std::endl;
                exit(EXIT_FAILURE);
            }

            // Use OpenMP to copy the results to the periodograms array
            #pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < N_samples; k++) {
                for (unsigned long l = 0; l < N_curves_current; l++) {
                    std::copy(
                        deltaChi_host + k * N_curves_current * N_freqs_current + l * N_freqs_current,  // Source
                        deltaChi_host + k * N_curves_current * N_freqs_current + (l + 1) * N_freqs_current,  // Source end
                        periodograms.begin() + (j + l) * N_samples * N_freqs + k * N_freqs + i  // Destination
                    );
                }
            }
        }
    }

    // Free device memory
    error = cudaFree(t);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree t: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaFree(y);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree y: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaFree(ivar);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree ivar: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaFree(freqs);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree freqs: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaFree(indices);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree indices: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaFree(deltaChiArr);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree deltaChiArr: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaFree(N_bins_device);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaFree N_bins_device: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Free host memory
    delete[] ivar_host;
    delete[] deltaChi_host;

    return periodograms;
}