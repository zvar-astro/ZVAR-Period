#include "wrappers.cuh"

MatrixXf cudaRunFPW(VectorXf t_vec, MatrixXf y_mat, MatrixXf dy_mat, VectorXf freqs_vec, 
                    int N_bins, int N_freqs_batch, int N_curves_batch) {

    float totalTime = 0.0f;
    //time stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int N_dat = t_vec.size();
    int N_curves = y_mat.cols();
    int N_freqs = freqs_vec.size();
    printf("N_dat: %d, N_curves: %d, N_freqs: %d\n", N_dat, N_curves, N_freqs);

    // Convert to arrays
    float* t_host = t_vec.data();
    float* freqs_host = freqs_vec.data();

    // Allocate memory for device arrays
    float* t;
    float* y;
    float* ivar;
    float* freqs;
    int* indices;
    float* deltaChiArr;
    cudaMalloc(&t, N_dat * sizeof(float));
    cudaMalloc(&y, N_dat * N_curves_batch * sizeof(float));
    cudaMalloc(&ivar, N_dat * N_curves_batch * sizeof(float));
    cudaMalloc(&freqs, N_freqs_batch * sizeof(float));  // Allocate memory for frequencies
    cudaMalloc(&indices, N_dat * N_freqs_batch * sizeof(int));  // Allocate memory for indices for all frequencies
    cudaMalloc(&deltaChiArr, N_freqs_batch * N_curves_batch * sizeof(float));  // Allocate memory for deltaChiArr for all frequencies

    // Copy data from host to device
    cudaMemcpy(t, t_host, N_dat * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and grid size for kernel launches
    dim3 blockSizeInd(16, 16);  // Use a 2D array of threads in each block
    dim3 gridSizeInd((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs_batch + blockSizeInd.y - 1) / blockSizeInd.y);
    dim3 blockSizeChi(128, 8);  // Use a 2D array of threads in each block
    dim3 gridSizeChi((N_freqs_batch + blockSizeChi.x - 1) / blockSizeChi.x, (N_curves + blockSizeChi.y - 1) / blockSizeChi.y);
    size_t sharedMemoryChi = 2 * N_bins * blockSizeChi.x * blockSizeChi.y * sizeof(float);

    // Allocate memory for host array to hold the result
    float* deltaChi_host = new float[N_curves_batch * N_freqs_batch];

    // Create a matrix to store the periodograms for each light curve
    MatrixXf periodograms(N_freqs, N_curves);

    //copy y and ivar to device
    float* y_host = y_mat.data();
    float* dy_host = dy_mat.data();
    float* ivar_host = new float[N_dat * N_curves];
    for (int i = 0; i < N_curves * N_dat; i++) {
        ivar_host[i] = 1.0 / (dy_host[i] * dy_host[i]);
    }

    // Run kernels for each batch of frequencies over all light curves
    for (int i=0; i < N_freqs; i += N_freqs_batch) {
        int N_freqs_current = min(N_freqs_batch, N_freqs - i);

        cudaMemcpyAsync(freqs, freqs_host + i, N_freqs_current * sizeof(float), cudaMemcpyHostToDevice);

        dim3 gridSizeInd_current((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs_current + blockSizeInd.y - 1) / blockSizeInd.y);

        cudaMakeIndices<<<gridSizeInd_current, blockSizeInd>>>(t, freqs, indices, N_bins, N_dat, N_freqs_current);

        for (int k = 0; k < N_curves; k += N_curves_batch) {
            int N_curves_current = min(N_curves_batch, N_curves - k);

            cudaMemsetAsync(deltaChiArr, 0, N_freqs_current * N_curves_current * sizeof(float));

            cudaMemcpy(y, y_host + k * N_dat, N_dat * N_curves_current * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(ivar, ivar_host + k * N_dat, N_dat * N_curves_current * sizeof(float), cudaMemcpyHostToDevice);

            cudaDeltaChi2<<<gridSizeChi, blockSizeChi, sharedMemoryChi>>>(
                            y, ivar, indices, deltaChiArr, N_bins, N_dat, N_freqs_current, N_curves_current);

            cudaMemcpyAsync(deltaChi_host, deltaChiArr, N_freqs_current * N_curves_current * sizeof(float), cudaMemcpyDeviceToHost);

            // Map the result to an Eigen vector and store it in the periodograms matrix
            for (int j = 0; j < N_curves_current; j++) {
                // periodograms.row(k + j).segment(i, N_freqs_current) = Eigen::Map<VectorXf>(deltaChi_host + j * N_freqs_current, N_freqs_current);
                periodograms.col(k + j).segment(i, N_freqs_current) = Eigen::Map<VectorXf>(deltaChi_host + j * N_freqs_current, N_freqs_current);
            }
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    totalTime += milliseconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("execution time per LC: %f ms\n", totalTime/ N_curves); // Print the total time

    // Free device memory
    cudaFree(t);
    cudaFree(y);
    cudaFree(ivar);
    cudaFree(freqs);
    cudaFree(indices);
    cudaFree(deltaChiArr);

    // Free host memory
    delete[] ivar_host;
    delete[] deltaChi_host;

    return periodograms;
}


/*Multibin FPW. Instead of passing a single integer value of N_bins we pass an array of N_bins values starting with the largest value.
The following values must be factors of N_bins[0]. N_samples is then the length of N_bins and determines how much memory must be allocated.*/
float* cudaRunFPW(float* t_host, float* y_host, float* dy_host, float* freqs_host,
                    int N_dat, int N_curves, int N_freqs,
                    int* N_bins, int N_freqs_batch, int N_curves_batch, int N_samples) {

    //time stuff
    float totalTime = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N_bins_total = 0;
    for (int i = 0; i < N_samples; i++) {
        N_bins_total += N_bins[i];
    }
    printf("N_dat: %d, N_curves: %d, N_freqs: %d\n", N_dat, N_curves, N_freqs);
    printf("N_bins_total: %d\n", N_bins_total);

    // Allocate memory for device arrays
    float* t;
    float* y;
    float* ivar;
    float* freqs;
    int* indices;
    float* deltaChiArr;
    int* N_bins_dev;
    cudaMalloc(&t, N_dat * sizeof(float));
    cudaMalloc(&y, N_dat * N_curves_batch * sizeof(float));
    cudaMalloc(&ivar, N_dat * N_curves_batch * sizeof(float));
    cudaMalloc(&freqs, N_freqs_batch * sizeof(float));  // Allocate memory for frequencies
    cudaMalloc(&indices, N_dat * N_freqs_batch * sizeof(int));  // Allocate memory for indices for all frequencies
    cudaMalloc(&deltaChiArr, N_freqs_batch * N_curves_batch * N_samples * sizeof(float));  // Allocate memory for deltaChiArr for all frequencies
    cudaMalloc(&N_bins_dev, N_samples * sizeof(int));  // Allocate memory for N_bins

    // Copy data from host to device
    cudaMemcpy(t, t_host, N_dat * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_bins_dev, N_bins, N_samples * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size for kernel launches
    dim3 blockSizeInd(16, 16);  // Use a 2D array of threads in each block
    dim3 gridSizeInd((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs_batch + blockSizeInd.y - 1) / blockSizeInd.y);
    dim3 blockSizeChi(64, 8);  // Use a 2D array of threads in each block
    dim3 gridSizeChi((N_freqs_batch + blockSizeChi.x - 1) / blockSizeChi.x, (N_curves + blockSizeChi.y - 1) / blockSizeChi.y);
    size_t sharedMemoryChi = 2 * N_bins[0] * blockSizeChi.x * blockSizeChi.y * sizeof(float);

    cout << "sharedMemoryChi: " << sharedMemoryChi / 1024 << endl;

    // Allocate memory for host array to hold the result
    float* deltaChi_host = new float[N_curves_batch * N_freqs_batch * N_samples];

    // Create a matrix to store the periodograms for each light curve
    float* periodograms = new float[N_curves * N_freqs * N_samples];

    //copy y and ivar to device
    // float* y_host = y_mat.data();
    // float* dy_host = dy_mat.data();
    float* ivar_host = new float[N_dat * N_curves];
    for (int i = 0; i < N_curves * N_dat; i++) {
        ivar_host[i] = 1.0 / (dy_host[i] * dy_host[i]);
    }

    //diagnostics
    cout << "periodogram size " << N_curves * N_freqs * N_samples << endl;

    cudaEventRecord(start);
    // Run kernels for each batch of frequencies over all light curves
    for (int i=0; i < N_freqs; i += N_freqs_batch) {
        int N_freqs_current = min(N_freqs_batch, N_freqs - i);

        cudaMemcpyAsync(freqs, freqs_host + i, N_freqs_current * sizeof(float), cudaMemcpyHostToDevice);

        dim3 gridSizeInd_current((N_dat + blockSizeInd.x - 1) / blockSizeInd.x, (N_freqs_current + blockSizeInd.y - 1) / blockSizeInd.y);

        cudaMakeIndices<<<gridSizeInd_current, blockSizeInd>>>(t, freqs, indices, N_bins[0], N_dat, N_freqs_current);

        for (int j = 0; j < N_curves; j += N_curves_batch) {
            int N_curves_current = min(N_curves_batch, N_curves - j);

            cudaMemsetAsync(deltaChiArr, 0, N_freqs_current * N_curves_current * N_samples * sizeof(float));

            cudaMemcpy(y, y_host + j * N_dat, N_dat * N_curves_current * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(ivar, ivar_host + j * N_dat, N_dat * N_curves_current * sizeof(float), cudaMemcpyHostToDevice);

            cudaDeltaChi2Multibin<<<gridSizeChi, blockSizeChi, sharedMemoryChi>>>(
                                    y, ivar, indices, deltaChiArr, N_bins_dev, N_dat, N_freqs_current, N_curves_current, N_samples, N_bins_total);

            cudaMemcpyAsync(deltaChi_host, deltaChiArr, N_freqs_current * N_curves_current * N_samples * sizeof(float), cudaMemcpyDeviceToHost);

            for (int k = 0; k < N_samples; k++) {
                for (int l = 0; l < N_curves_current; l++) {
                    memcpy(
                        periodograms + k * N_curves * N_freqs + (j + l) * N_freqs + i,  // Destination
                        deltaChi_host + k * N_curves_current * N_freqs_current + l * N_freqs_current,  // Source
                        N_freqs_current * sizeof(float)  // Number of bytes to copy
                    );
                }
            }
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    totalTime += milliseconds;
    
    cout << "total time: " << totalTime << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("execution time per LC per bin sample: %f ms\n", totalTime/ N_curves / N_samples); // Print the total time

    // Free device memory
    cudaFree(t);
    cudaFree(y);
    cudaFree(ivar);
    cudaFree(freqs);
    cudaFree(indices);
    cudaFree(deltaChiArr);

    // Free host memory
    delete[] ivar_host;
    delete[] deltaChi_host;

    return periodograms;
}