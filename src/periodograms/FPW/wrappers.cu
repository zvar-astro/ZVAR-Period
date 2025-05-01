#include "wrappers.cuh"

/*Multibin FPW. Instead of passing a single integer value of N_bins we pass an array of N_bins values starting with the largest value.
The following values must be factors of N_bins[0]. N_samples is then the length of N_bins and determines how much memory must be allocated.*/
std::vector<float> cudaRunFPW(float* t_host, float* y_host, float* dy_host, float* freqs_host,
                    int N_dat, unsigned long N_curves, unsigned long N_freqs,
                    int* N_bins, unsigned long N_freqs_batch, unsigned long N_curves_batch, int N_samples) {

    int N_bins_total = 0;
    for (int i = 0; i < N_samples; i++) {
        N_bins_total += N_bins[i];
    }

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
    int* indices;
    float* deltaChiArr;
    int* N_bins_dev;
    
    error = cudaMalloc(&t, N_dat * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc t: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&y, N_dat * N_curves_batch * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc y: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&ivar, N_dat * N_curves_batch * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc ivar: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&freqs, N_freqs_batch * sizeof(float));  // Allocate memory for frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc freqs: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&indices, N_dat * N_freqs_batch * sizeof(int));  // Allocate memory for indices for all frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc indices: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&deltaChiArr, N_freqs_batch * N_curves_batch * N_samples * sizeof(float));  // Allocate memory for deltaChiArr for all frequencies
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc deltaChiArr: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc(&N_bins_dev, N_samples * sizeof(int));  // Allocate memory for N_bins
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc N_bins_dev: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy data from host to device
    error = cudaMemcpy(t, t_host, N_dat * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy t: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
        
    }
    error = cudaMemcpy(N_bins_dev, N_bins, N_samples * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy N_bins_dev: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Define block size and grid size for kernel launches
    dim3 blockSizeInd(16, 16);  // Use a 2D array of threads in each block
    dim3 gridSizeInd((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs_batch + blockSizeInd.y - 1) / blockSizeInd.y);
    dim3 blockSizeChi(32, 8);  // Use a 2D array of threads in each block
    dim3 gridSizeChi((N_freqs_batch + blockSizeChi.x - 1) / blockSizeChi.x, (N_curves_batch + blockSizeChi.y - 1) / blockSizeChi.y);
    size_t sharedMemoryChi = 2 * N_bins[0] * blockSizeChi.x * blockSizeChi.y * sizeof(float);

    // Allocate memory for host array to hold the result
    float* deltaChi_host = new float[N_curves_batch * N_freqs_batch * N_samples];

    // Create a matrix to store the periodograms for each light curve
    std::vector<float> periodograms(N_curves * N_freqs * N_samples);

    // Copy y and ivar to device
    float* ivar_host = new float[N_dat * N_curves];
    for (unsigned long i = 0; i < N_curves * N_dat; i++) {
        ivar_host[i] = 1.0 / (dy_host[i] * dy_host[i]);
    }

    // Run kernels for each batch of frequencies over all light curves
    for (unsigned long i = 0; i < N_freqs; i += N_freqs_batch) {
        std::cout << "Processing frequency batch " << i / N_freqs_batch + 1 << " of " << (N_freqs + N_freqs_batch - 1) / N_freqs_batch << std::endl;
        unsigned long N_freqs_current = std::min(N_freqs_batch, N_freqs - i);

        error = cudaMemcpyAsync(freqs, freqs_host + i, N_freqs_current * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in cudaMemcpy freqs (wrappers): " << cudaGetErrorString(error) << std::endl;
            exit(EXIT_FAILURE);
        }

        dim3 gridSizeInd_current((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs_current + blockSizeInd.y - 1) / blockSizeInd.y);

        cudaMakeIndices<<<gridSizeInd_current, blockSizeInd>>>(t, freqs, indices, N_bins[0], N_dat, N_freqs_current);

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in cudaMakeIndices (wrappers): " << cudaGetErrorString(error) << std::endl;
            exit(EXIT_FAILURE);
        }

        for (unsigned long j = 0; j < N_curves; j += N_curves_batch) {
            unsigned long N_curves_current = std::min(N_curves_batch, N_curves - j);

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
                                    y, ivar, indices, deltaChiArr, N_bins_dev, N_dat, N_freqs_current, N_curves_current, N_samples, N_bins_total);

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

    // Free host memory
    delete[] ivar_host;
    delete[] deltaChi_host;

    return periodograms;
}