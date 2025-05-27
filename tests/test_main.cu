// #define CATCH_CONFIG_MAIN  // This generates the main() function
#include "test_utils.cuh"

TEST_CASE("FPW GPU vs CPU 20 Bin Periodogram Comparison", "[FPW]") {
    try {
        // Read the data file
        std::string dataFile = "../tests/data/tydy.csv";
        std::vector<std::vector<double>> data = readCSV(dataFile);

        // Extract columns
        std::vector<double> times = extractColumn(data, 0);
        std::vector<double> flux = extractColumn(data, 1);
        std::vector<double> fluxErr = extractColumn(data, 2);

        // Read the frequency grid file
        std::string fgridFile = "../tests/data/fgrid.csv";
        std::vector<std::vector<double>> fgridData = readCSV(fgridFile);
        std::vector<double> fgrid = extractColumn(fgridData, 0);

        // Convert to floats
        std::vector<float> timesFloat(times.size());
        std::vector<float> fluxFloat(flux.size());
        std::vector<float> fluxErrFloat(fluxErr.size());
        std::vector<float> fgridFloat(fgrid.size());
        std::transform(times.begin(), times.end(), timesFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(flux.begin(), flux.end(), fluxFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(fluxErr.begin(), fluxErr.end(), fluxErrFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(fgrid.begin(), fgrid.end(), fgridFloat.begin(), [](double val) { return static_cast<float>(val); });

        // Convert the float vectors to pointer arrays
        float* timesPtr = timesFloat.data();
        float* fluxPtr = fluxFloat.data();
        float* fluxErrPtr = fluxErrFloat.data();
        float* fgridPtr = fgridFloat.data();

        // Extract constants
        int N_samples = 3;
        int* N_bins = new int[N_samples]{20, 10, 5};
        int N_dat = timesFloat.size();
        unsigned long N_curves = 1;
        unsigned long N_freqs = fgridFloat.size();
        unsigned long N_freqs_batch = 100000;
        unsigned long N_curves_batch = 1;

        // Run FPW on the data
        std::vector<float> periodograms = cudaRunFPW(timesPtr, fluxPtr, fluxErrPtr, fgridPtr,
                                                     N_dat, N_curves, N_freqs, N_bins, N_freqs_batch, N_curves_batch, N_samples);

        std::vector<float> gpuPeriodogram(periodograms.begin(), periodograms.begin() + N_freqs);

        // Load in the expected periodogram
        std::string expectedFile = "../tests/data/CPU_periodogram.csv";
        std::vector<std::vector<double>> expectedData = readCSV(expectedFile);
        std::vector<double> cpuPeriodogram = extractColumn(expectedData, 0);
        // Convert to float
        std::vector<float> cpuPeriodogramFloat(cpuPeriodogram.size());
        std::transform(cpuPeriodogram.begin(), cpuPeriodogram.end(), cpuPeriodogramFloat.begin(),
                       [](double val) { return static_cast<float>(val); });

        // Element-wise subtract the two periodograms
        std::vector<float> diff(gpuPeriodogram.size());
        for (size_t i = 0; i < gpuPeriodogram.size(); ++i) {
            diff[i] = gpuPeriodogram[i] - cpuPeriodogramFloat[i];
        }

        // Compute the mean and standard deviation of the CPU periodogram
        auto [meanCPU, stdevCPU] = computeMeanAndStdDev(cpuPeriodogramFloat);

        auto [meanGPU, stdevGPU] = computeMeanAndStdDev(gpuPeriodogram);

        // Compute the mean and standard deviation of the difference
        auto [meanDiff, stdevDiff] = computeMeanAndStdDev(diff);

        // Compute the maximum of the periodogram
        float maxPeriodogram = *std::max_element(cpuPeriodogramFloat.begin(), cpuPeriodogramFloat.end());

        //  Print the mean and standard deviation of both periodograms
        std::cout << "mean of the CPU periodogram: " << meanCPU << std::endl;
        std::cout << "std of the CPU periodogram: " << stdevCPU << std::endl;

        std::cout << "mean of the GPU periodogram: " << meanGPU << std::endl;
        std::cout << "std of the GPU periodogram: " << stdevGPU << std::endl;

        // Print the maximum of the periodogram
        std::cout << "max of the CPU periodogram: " << maxPeriodogram << std::endl;

        //Print the mean and standard deviation of the difference
        std::cout << "mean of the difference: " << meanDiff << std::endl;
        std::cout << "std of the difference: " << stdevDiff << std::endl;

        // Assertions
        REQUIRE(meanDiff < 1.0f); // Mean of the difference should be less than 1
        REQUIRE(stdevDiff < stdevCPU); // Standard deviation of the difference should be less than that of the CPU periodogram

        // Clean up
        delete[] N_bins;

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}


TEST_CASE("Check Outputs of Peak Finding", "[Statistics]") {
    try {
        // Read the data file
        std::string dataFile = "../tests/data/tydy.csv";
        std::vector<std::vector<double>> data = readCSV(dataFile);

        // Extract columns
        std::vector<double> times = extractColumn(data, 0);
        std::vector<double> flux = extractColumn(data, 1);
        std::vector<double> fluxErr = extractColumn(data, 2);

        // Read the frequency grid file
        std::string fgridFile = "../tests/data/fgrid.csv";
        std::vector<std::vector<double>> fgridData = readCSV(fgridFile);
        std::vector<double> fgrid = extractColumn(fgridData, 0);

        // Convert to floats
        std::vector<float> timesFloat(times.size());
        std::vector<float> fluxFloat(flux.size());
        std::vector<float> fluxErrFloat(fluxErr.size());
        std::vector<float> fgridFloat(fgrid.size());
        std::transform(times.begin(), times.end(), timesFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(flux.begin(), flux.end(), fluxFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(fluxErr.begin(), fluxErr.end(), fluxErrFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(fgrid.begin(), fgrid.end(), fgridFloat.begin(), [](double val) { return static_cast<float>(val); });

        // Convert the float vectors to pointer arrays
        float* timesPtr = timesFloat.data();
        float* fluxPtr = fluxFloat.data();
        float* fluxErrPtr = fluxErrFloat.data();
        float* fgridPtr = fgridFloat.data();

        // Extract constants
        int N_samples = 3;
        int* N_bins = new int[N_samples]{20, 10, 5};
        int N_dat = timesFloat.size();
        unsigned long N_curves = 1;
        unsigned long N_freqs = fgridFloat.size();
        unsigned long N_freqs_batch = 100000;
        unsigned long N_curves_batch = 1;

        // Run FPW on the data
        std::vector<float> periodograms = cudaRunFPW(timesPtr, fluxPtr, fluxErrPtr, fgridPtr,
                                                     N_dat, N_curves, N_freqs, N_bins, N_freqs_batch, N_curves_batch, N_samples);

        // Reshape the periodogram to 2D
        std::vector<std::vector<float>> reshapedPeriodogram(N_samples, std::vector<float>(N_freqs));
        for (size_t i = 0; i < N_samples; ++i) {
            for (size_t j = 0; j < N_freqs; ++j) {
                reshapedPeriodogram[i][j] = periodograms[i * N_freqs + j];
            }
        }

        // Find the peaks in each periodogram
        int Ntop = 10;
        int distance = 300;
        float threshold = 7.0f;

        for (size_t i = 0; i < N_samples; ++i) {
            std::vector<float> periodogram = reshapedPeriodogram[i];
            auto [mean, stdev] = computeMeanAndStdDev(periodogram);
            std::vector<int> peakIndices = find_peaks(periodogram, Ntop, distance, mean, stdev, threshold);
            std::vector<float> peaks(Ntop);
            for (int j = 0; j < Ntop; ++j) {
                peaks[j] = (periodogram[peakIndices[j]] - mean) / stdev;
            }

            // Log the sample index and peaks for debugging
            INFO("Sample index: " << i);
            INFO("Peaks: " << vectorToString(peaks));

            // Require that peaks are found in this periodogram
            REQUIRE(peakIndices[0] > 0);

            // Require that all peaks are above the threshold
            for (int j = 0; j < Ntop; ++j) {
                CAPTURE(i, j, peaks[j]);
                REQUIRE(peaks[j] > threshold);
            }

            // Require that the peaks are sorted
            for (int j = 1; j < Ntop; ++j) {
                CAPTURE(i, j, peaks[j], peaks[j - 1]);
                REQUIRE(peaks[j] <= peaks[j - 1]);
            }
        }

        // Clean up
        delete[] N_bins;

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}


TEST_CASE("Check GPU Median and MAD Calculations", "[Statistics]") {
    try {

        int seed = Catch::getSeed(); // Get the seed used by Catch2
        std::mt19937 gen(seed);      // Use the same seed for your random generator

        // Make fake light curves
        int N_dat = 1000000;
        int N_curves = 512;
        float* data = new float[N_dat * N_curves];

        // Generate random data uniformly between -10 and 10
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        for (int i = 0; i < N_dat * N_curves; ++i) {
            data[i] = dist(gen);
        }

        // Run the CPU median and MAD calculations
        std::vector<float> CPUmedians(N_curves);
        std::vector<float> CPUmads(N_curves);

        auto cpuStart = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N_curves; ++i) {
            std::vector<float> curve(data + i * N_dat, data + (i + 1) * N_dat);
            auto [median, mad] = computeMedianAndMAD(curve);
            CPUmedians[i] = median;
            CPUmads[i] = mad;
        }

        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuElapsed = cpuEnd - cpuStart;

        // Print CPU timing
        std::cout << "CPU execution time: " << cpuElapsed.count() / N_curves << " ms per light curve" << std::endl;

        // Run the GPU median and MAD calculations
        // Define the number of threads and blocks
        int threadsPerBlock = 256;
        dim3 blockSizeMed(threadsPerBlock);
        dim3 gridSizeMed((N_curves + threadsPerBlock - 1) / threadsPerBlock);
        // size_t sharedMemMed = N_dat * threadsPerBlock * sizeof(float);

        // Copy light curves to device
        float* d_data;
        cudaMalloc(&d_data, N_dat * N_curves * sizeof(float));
        cudaMemcpy(d_data, data, N_dat * N_curves * sizeof(float), cudaMemcpyHostToDevice);

        // Create output arrays for the median and MAD
        float* GPUmedians = new float[N_curves];
        float* GPUmads = new float[N_curves];
        cudaMalloc(&GPUmedians, N_curves * sizeof(float)); // Allocate memory for GPU medians
        cudaMalloc(&GPUmads, N_curves * sizeof(float));    // Allocate memory for GPU MADs

        // Check for errors in memory copy
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in memory copy: " << cudaGetErrorString(error) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record the start event
        cudaEventRecord(start, 0);

        // Launch the kernel
        CudaComputeMedianandMAD<<<gridSizeMed, blockSizeMed>>>(d_data, GPUmedians, GPUmads, N_dat, N_curves);
        // CudaComputeMedianandMAD<<<gridSizeMed, blockSizeMed, sharedMemMed>>>(d_data, GPUmedians, GPUmads, N_dat, N_curves);

        // Record the stop event
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        // Print the timing per light curve
        std::cout << "Kernel execution time: " << elapsedTime / N_curves << " ms per light curve" << std::endl;

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Check for errors in kernel launch
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in kernel launch: " << cudaGetErrorString(error) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the results back to the host
        float* h_GPUmedians = new float[N_curves];
        float* h_GPUmads = new float[N_curves];
        cudaMemcpy(h_GPUmedians, GPUmedians, N_curves * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_GPUmads, GPUmads, N_curves * sizeof(float), cudaMemcpyDeviceToHost);
        // Check for errors in memory copy
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in memory copy: " << cudaGetErrorString(error) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Compare the results
        for (int i = 0; i < N_curves; ++i) {
            CAPTURE(i);
            CAPTURE(CPUmedians[i], h_GPUmedians[i]);
            CAPTURE(CPUmads[i], h_GPUmads[i]);
            REQUIRE(std::abs(CPUmedians[i] - h_GPUmedians[i]) < 1e-3f);
            REQUIRE(std::abs(CPUmads[i] - h_GPUmads[i]) < 1e-3f);
        }

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}

//Set Ntop at compile time
#define Ntop 32

TEST_CASE("Check GPU Peak Finding vs CPU", "[Statistics]") {
    // Read the frequency grid file
    std::string fgridFile = "../tests/data/fgrid.csv";
    std::vector<std::vector<double>> fgridData = readCSV(fgridFile);
    std::vector<double> fgrid = extractColumn(fgridData, 0);
    //convert to float
    std::vector<float> fgridFloat(fgrid.size());
    std::transform(fgrid.begin(), fgrid.end(), fgridFloat.begin(),
                   [](double val) { return static_cast<float>(val); });


    std::string expectedFile = "../tests/data/CPU_periodogram.csv";
    std::vector<std::vector<double>> expectedData = readCSV(expectedFile);
    std::vector<double> cpuPeriodogram = extractColumn(expectedData, 0);
    // Convert to float
    std::vector<float> cpuPeriodogramFloat(cpuPeriodogram.size());
    std::transform(cpuPeriodogram.begin(), cpuPeriodogram.end(), cpuPeriodogramFloat.begin(),
                   [](double val) { return static_cast<float>(val); });

    const int N_dat = fgrid.size();
    const int N_curves = 1; // Test with a single curve for simplicity

    // --- Parameters ---
    const int distance = 300;
    const float threshold = 1.5f; // Significance threshold

    // --- Calculate Median/MAD ---
    float pgramMedian = median(cpuPeriodogramFloat.data(), N_dat);
    float pgramMAD = mad(cpuPeriodogramFloat.data(), N_dat, pgramMedian);

    // --- Run CPU Version ---
    // Use the significance array as input, with mean=0, stdev=1
    std::vector<int> cpu_peaks_indices = find_peaks(cpuPeriodogramFloat, Ntop, distance, pgramMedian, pgramMAD, threshold);

    // --- Prepare GPU Data ---
    float* d_data;
    float* d_median;
    float* d_mad;
    int* d_peakIndices;
    float* d_peakValues; // Although not strictly compared, the kernel needs it

    cudaMalloc(&d_data, N_curves * N_dat * sizeof(float));
    cudaMalloc(&d_median, N_curves * sizeof(float));
    cudaMalloc(&d_mad, N_curves * sizeof(float));
    cudaMalloc(&d_peakIndices, N_curves * Ntop * sizeof(int));
    cudaMalloc(&d_peakValues, N_curves * Ntop * sizeof(float));

    // Copy periodogram data to GPU
    cudaMemcpy(d_data, cpuPeriodogramFloat.data(), N_dat * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_median, &pgramMedian, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mad, &pgramMAD, sizeof(float), cudaMemcpyHostToDevice);

    // --- Run GPU Version ---
    dim3 blockSize(256); // Example block size
    dim3 gridSize((N_curves + blockSize.x - 1) / blockSize.x);

    // IMPORTANT: The GPU kernel modifies d_data in-place.
    // Since we copied significance data, this is okay for this test structure.
    cudaFindPeaks<<<gridSize, blockSize>>>(d_data, d_median, d_mad,
                                           d_peakIndices, d_peakValues,
                                           N_dat, N_curves, distance, threshold);
    cudaGetLastError(); // Check for kernel launch errors
    cudaDeviceSynchronize(); // Wait for kernel to finish

    // --- Copy GPU Results ---
    std::vector<int> gpu_peaks_indices(Ntop);
    // Copy results for the first (and only) curve
    cudaMemcpy(gpu_peaks_indices.data(), d_peakIndices, Ntop * sizeof(int), cudaMemcpyDeviceToHost);

    // // --- Compare Results ---
    // // Sort both vectors to compare contents regardless of order (ignoring -1 placeholders)
    // auto sort_and_filter = [](std::vector<int>& v) {
    //     v.erase(std::remove(v.begin(), v.end(), -1), v.end());
    //     std::sort(v.begin(), v.end());
    // };

    // sort_and_filter(cpu_peaks_indices);
    // sort_and_filter(gpu_peaks_indices);

    // Use Catch2 assertions
    REQUIRE(cpu_peaks_indices.size() == gpu_peaks_indices.size());
    CHECK(cpu_peaks_indices == gpu_peaks_indices);

    // Optional: Print for debugging if CHECK fails
    if (cpu_peaks_indices != gpu_peaks_indices) {
        std::cout << "CPU Peaks: ";
        for(int idx : cpu_peaks_indices) std::cout << idx << " ";
        std::cout << std::endl;
        std::cout << "GPU Peaks: ";
        for(int idx : gpu_peaks_indices) std::cout << idx << " ";
        std::cout << std::endl;
    }


    // --- Clean Up GPU Memory ---
    cudaFree(d_data);
    cudaFree(d_median);
    cudaFree(d_mad);
    cudaFree(d_peakIndices);
    cudaFree(d_peakValues);
}


TEST_CASE("Check GPU Peak Finding vs CPU Timing", "[Statistics][Timing]") {
    try {
        // --- Parameters ---
        const int N_dat = 1000000; // Length of each curve
        const int N_curves = 1000; // Number of curves to process
        const int distance = 300;
        const float threshold = 1.5f; // Significance threshold
        // Ntop is assumed to be defined elsewhere

        // --- Generate Random Data ---
        std::mt19937 gen(Catch::getSeed());
        std::normal_distribution<float> dist(0.0f, 10.0f); // Mean 0, Stddev 10

        std::vector<float> h_data(static_cast<size_t>(N_curves) * N_dat);
        for (size_t i = 0; i < h_data.size(); ++i) {
            h_data[i] = dist(gen);
        }

        // --- GPU Median/MAD Calculation ---
        float* d_data;
        float* d_median;
        float* d_mad;
        int* d_peakIndices;
        float* d_peakValues;

        cudaMalloc(&d_data, static_cast<size_t>(N_curves) * N_dat * sizeof(float));
        cudaMalloc(&d_median, N_curves * sizeof(float));
        cudaMalloc(&d_mad, N_curves * sizeof(float));
        cudaMalloc(&d_peakIndices, static_cast<size_t>(N_curves) * Ntop * sizeof(int));
        cudaMalloc(&d_peakValues, static_cast<size_t>(N_curves) * Ntop * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_data, h_data.data(), static_cast<size_t>(N_curves) * N_dat * sizeof(float), cudaMemcpyHostToDevice);

        // Run GPU Median/MAD kernel
        dim3 blockSizeMed(256);
        dim3 gridSizeMed((N_curves + blockSizeMed.x - 1) / blockSizeMed.x);
        CudaComputeMedianandMAD<<<gridSizeMed, blockSizeMed>>>(d_data, d_median, d_mad, N_dat, N_curves);
        cudaDeviceSynchronize(); // Sync after median/mad kernel
        cudaGetLastError(); // Check for errors

        // Copy GPU-computed medians/MADs back to host for CPU processing
        std::vector<float> h_gpu_medians(N_curves);
        std::vector<float> h_gpu_mads(N_curves);
        cudaMemcpy(h_gpu_medians.data(), d_median, N_curves * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gpu_mads.data(), d_mad, N_curves * sizeof(float), cudaMemcpyDeviceToHost);


        // --- CPU Peak Finding (using GPU Medians/MADs) ---
        std::vector<std::vector<int>> h_cpu_peak_indices(N_curves, std::vector<int>(Ntop, -1));
        auto cpuStart = std::chrono::high_resolution_clock::now();

        for (int c = 0; c < N_curves; ++c) {
            // Get pointer to current curve data on host
            float* current_curve_data = h_data.data() + static_cast<size_t>(c) * N_dat;
            // Create a *non-owning* vector view or copy if find_peaks modifies it
            std::vector<float> curve_vec(current_curve_data, current_curve_data + N_dat);

            // Run CPU peak finding using the medians/MADs calculated by the GPU
            std::vector<int> peaks = find_peaks(curve_vec, Ntop, distance, h_gpu_medians[c], h_gpu_mads[c], threshold);
            // Ensure peaks vector size doesn't exceed Ntop before copying
             size_t num_peaks_to_copy = std::min(peaks.size(), static_cast<size_t>(Ntop));
            std::copy(peaks.begin(), peaks.begin() + num_peaks_to_copy, h_cpu_peak_indices[c].begin());
        }

        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuElapsed = cpuEnd - cpuStart;
        std::cout << "CPU Peak Finding Total Time (using GPU Med/MAD): " << cpuElapsed.count() << " ms" << std::endl;
        std::cout << "CPU Peak Finding Avg Time per Curve: " << cpuElapsed.count() / N_curves << " ms" << std::endl;


        // --- GPU Peak Finding ---
        dim3 blockSizePeaks(256);
        dim3 gridSizePeaks((N_curves + blockSizePeaks.x - 1) / blockSizePeaks.x);

        cudaEvent_t startGPU, stopGPU;
        cudaEventCreate(&startGPU);
        cudaEventCreate(&stopGPU);

        cudaEventRecord(startGPU);

        // IMPORTANT: The GPU kernel modifies d_data in-place (calculates S/N).
        // It uses d_median and d_mad which were computed by CudaComputeMedianandMAD.
        cudaFindPeaks<<<gridSizePeaks, blockSizePeaks>>>(d_data, d_median, d_mad,
                                                        d_peakIndices, d_peakValues,
                                                        N_dat, N_curves, distance, threshold);
        cudaGetLastError(); // Check for kernel launch errors

        cudaEventRecord(stopGPU);
        cudaEventSynchronize(stopGPU); // Wait for kernel to finish

        float gpuElapsedMs;
        cudaEventElapsedTime(&gpuElapsedMs, startGPU, stopGPU);
        std::cout << "GPU Peak Finding Kernel Time: " << gpuElapsedMs << " ms" << std::endl;
        std::cout << "GPU Peak Finding Avg Time per Curve: " << gpuElapsedMs / N_curves << " ms" << std::endl;

        // --- Copy GPU Results ---
        std::vector<int> h_gpu_peak_indices_flat(static_cast<size_t>(N_curves) * Ntop);
        cudaMemcpy(h_gpu_peak_indices_flat.data(), d_peakIndices, static_cast<size_t>(N_curves) * Ntop * sizeof(int), cudaMemcpyDeviceToHost);

        bool all_match = true;
        for(int c = 0; c < N_curves; ++c) {
            std::vector<int> current_cpu_peaks = h_cpu_peak_indices[c];
            std::vector<int> current_gpu_peaks(h_gpu_peak_indices_flat.begin() + static_cast<size_t>(c) * Ntop,
                                               h_gpu_peak_indices_flat.begin() + static_cast<size_t>(c + 1) * Ntop);

            // sort_and_filter(current_cpu_peaks);
            // sort_and_filter(current_gpu_peaks);
            for (int p = 0; p < 1; ++p) {
                if (current_cpu_peaks[p] != current_gpu_peaks[p]) {
                    all_match = false;
                    std::cout << "Mismatch found for curve " << c << std::endl;
                }
            }

        }
        CHECK(all_match); // Check if all curves produced matching peaks


        // --- Clean Up GPU Memory ---
        cudaFree(d_data);
        cudaFree(d_median);
        cudaFree(d_mad);
        cudaFree(d_peakIndices);
        cudaFree(d_peakValues);
        cudaEventDestroy(startGPU);
        cudaEventDestroy(stopGPU);

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}

TEST_CASE("Check Chunk Peak Finding VS Global", "[Statistics]") {
    try {
        // Read the frequency grid file
        std::string fgridFile = "../tests/data/fgrid.csv";
        std::vector<std::vector<double>> fgridData = readCSV(fgridFile);
        std::vector<double> fgrid = extractColumn(fgridData, 0);
        //convert to float
        std::vector<float> fgridFloat(fgrid.size());
        std::transform(fgrid.begin(), fgrid.end(), fgridFloat.begin(),
                    [](double val) { return static_cast<float>(val); });

        std::string expectedFile = "../tests/data/CPU_periodogram.csv";
        std::vector<std::vector<double>> expectedData = readCSV(expectedFile);
        std::vector<double> cpuPeriodogram = extractColumn(expectedData, 0);
        // Convert to float
        std::vector<float> cpuPeriodogramFloat(cpuPeriodogram.size());
        std::transform(cpuPeriodogram.begin(), cpuPeriodogram.end(), cpuPeriodogramFloat.begin(),
                    [](double val) { return static_cast<float>(val); });

        const int N_dat = fgrid.size();
        const int N_curves = 1; // Test with a single curve for simplicity

        // --- Parameters ---
        const int distance = 300;
        const float threshold = 1.5f; // Significance threshold

        // --- Calculate Median/MAD ---
        float pgramMedian = median(cpuPeriodogramFloat.data(), N_dat);
        float pgramMAD = mad(cpuPeriodogramFloat.data(), N_dat, pgramMedian);

        // --- Allocate device memory ---
        float *d_data, *d_median, *d_mad, *d_peakValues, *d_peakValues_chunk;
        int *d_peakIndices, *d_peakIndices_chunk;
        cudaMalloc(&d_data, N_dat * sizeof(float));
        cudaMalloc(&d_median, sizeof(float));
        cudaMalloc(&d_mad, sizeof(float));
        cudaMalloc(&d_peakIndices, N_peaks * sizeof(int));
        cudaMalloc(&d_peakValues, N_peaks * sizeof(float));
        cudaMalloc(&d_peakIndices_chunk, N_peaks * sizeof(int));
        cudaMalloc(&d_peakValues_chunk, N_peaks * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_data, cpuPeriodogramFloat.data(), N_dat * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_median, &pgramMedian, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mad, &pgramMAD, sizeof(float), cudaMemcpyHostToDevice);

        // --- Run global peak finding ---
        cudaMemset(d_peakIndices, -1, N_peaks * sizeof(int));
        cudaMemset(d_peakValues, 0, N_peaks * sizeof(float));
        cudaFindPeaks<<<1, 1>>>(d_data, d_median, d_mad, d_peakIndices, d_peakValues, N_dat, N_curves, distance, threshold);

        std::vector<int> global_peak_indices(N_peaks);
        std::vector<float> global_peak_values(N_peaks);
        cudaMemcpy(global_peak_indices.data(), d_peakIndices, N_peaks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(global_peak_values.data(), d_peakValues, N_peaks * sizeof(float), cudaMemcpyDeviceToHost);

        // --- Prepare for chunked peak finding ---
        const int N_chunks = 5;
        int chunk_size = (N_dat + N_chunks - 1) / N_chunks;

        // Allocate persistent state arrays
        int *d_state_candidate_idx, *d_state_distLastPeak;
        float *d_state_candidate_val, *d_state_currentThreshold;
        cudaMalloc(&d_state_candidate_idx, sizeof(int));
        cudaMalloc(&d_state_candidate_val, sizeof(float));
        cudaMalloc(&d_state_distLastPeak, sizeof(int));
        cudaMalloc(&d_state_currentThreshold, sizeof(float));

        // Initialize persistent state
        int init_idx = -1;
        float init_val = -INFINITY, init_thresh = threshold;
        cudaMemcpy(d_state_candidate_idx, &init_idx, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_state_candidate_val, &init_val, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_state_currentThreshold, &init_thresh, sizeof(float), cudaMemcpyHostToDevice);

        // Zero out chunk peak arrays
        cudaMemset(d_peakIndices_chunk, -1, N_peaks * sizeof(int));
        cudaMemset(d_peakValues_chunk, 0, N_peaks * sizeof(float));

        // Copy data again for chunked version (since global version normalizes in-place)
        cudaMemcpy(d_data, cpuPeriodogramFloat.data(), N_dat * sizeof(float), cudaMemcpyHostToDevice);

        // Run chunked peak finding
        for (int c = 0; c < N_chunks; ++c) {
            int chunk_start = c * chunk_size;
            int chunk_end = std::min((c + 1) * chunk_size, N_dat);
            cudaFindPeaksChunk<<<1, 1>>>(
                d_data, d_median, d_mad,
                d_peakIndices_chunk, d_peakValues_chunk,
                N_dat, N_curves, distance, threshold,
                d_state_candidate_idx, d_state_candidate_val,
                d_state_distLastPeak, d_state_currentThreshold,
                chunk_start, chunk_end
            );
        }

        std::vector<int> chunk_peak_indices(N_peaks);
        std::vector<float> chunk_peak_values(N_peaks);
        cudaMemcpy(chunk_peak_indices.data(), d_peakIndices_chunk, N_peaks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(chunk_peak_values.data(), d_peakValues_chunk, N_peaks * sizeof(float), cudaMemcpyDeviceToHost);

        // --- Compare results ---
        for (int i = 0; i < N_peaks; ++i) {
            REQUIRE(chunk_peak_indices[i] == global_peak_indices[i]);
            REQUIRE(chunk_peak_values[i] == global_peak_values[i]);
        }

        // Free device memory
        cudaFree(d_data);
        cudaFree(d_median);
        cudaFree(d_mad);
        cudaFree(d_peakIndices);
        cudaFree(d_peakValues);
        cudaFree(d_peakIndices_chunk);
        cudaFree(d_peakValues_chunk);
        cudaFree(d_state_candidate_idx);
        cudaFree(d_state_candidate_val);
        cudaFree(d_state_distLastPeak);
        cudaFree(d_state_currentThreshold);
    }
    catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}

TEST_CASE("Time Pre-Computing Vs Recursive Indices", "[FPW][Timing]") {
    try {
        std::string dataFile = "../tests/data/tydy.csv";
        std::vector<std::vector<double>> data = readCSV(dataFile);

        // Extract columns
        std::vector<double> times = extractColumn(data, 0);

        // Read the frequency grid file
        std::string fgridFile = "../tests/data/fgrid.csv";
        std::vector<std::vector<double>> fgridData = readCSV(fgridFile);
        std::vector<double> fgrid = extractColumn(fgridData, 0);

        // Convert to floats
        std::vector<float> timesFloat(times.size());
        std::vector<float> fgridFloat(fgrid.size());
        std::transform(times.begin(), times.end(), timesFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(fgrid.begin(), fgrid.end(), fgridFloat.begin(), [](double val) { return static_cast<float>(val); });

        // Extract constants
        int N_bins = 20;
        int N_dat = timesFloat.size();
        int N_freqs_batch = 100000;
        int N_freqs = std::min((int)fgridFloat.size(), N_freqs_batch);

        // --- Precompute indices on CPU and time transfer ---
        std::vector<int> h_indices(N_freqs * N_dat);
        unsigned char* d_indices = createIndices(timesFloat.data(), fgridFloat.data(), N_bins, N_dat, N_freqs);
        cudaMemcpy(h_indices.data(), d_indices, N_freqs * N_dat * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaFree(d_indices);

        auto start_transfer = std::chrono::high_resolution_clock::now();
        unsigned char* d_indices_transfer;
        cudaMalloc(&d_indices_transfer, N_freqs * N_dat * sizeof(unsigned char));
        cudaMemcpy(d_indices_transfer, h_indices.data(), N_freqs * N_dat * sizeof(unsigned char), cudaMemcpyHostToDevice);
        auto end_transfer = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_transfer = end_transfer - start_transfer;
        std::cout << "Time to transfer precomputed indices to device: " << elapsed_transfer.count() / N_freqs << " ms per frequency" << std::endl;
        cudaFree(d_indices_transfer);

        // --- Time computing indices directly on the GPU ---
        auto start_compute = std::chrono::high_resolution_clock::now();
        unsigned char* d_indices2 = createIndices(timesFloat.data(), fgridFloat.data(), N_bins, N_dat, N_freqs);
        cudaFree(d_indices2);
        auto end_compute = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_compute = end_compute - start_compute;
        std::cout << "Time to compute indices on device: " << elapsed_compute.count() / N_freqs << " ms per frequency" << std::endl;

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}

TEST_CASE("Check NVIDIA Device Parameters", "[Memory]") {
    int device = 0;
    int maxSharedMem = 0;
    cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Print the maximum shared memory size
    std::cout << "Maximum shared memory size per block: " << maxSharedMem / 1024 << " KB" << std::endl;
}


TEST_CASE("Check Processing Output", "[Processing]") {
    std::string hdf_filename = "/data/swhitebook/zvar_testing/data/data_0642_01_2_zr.h5";
    std::string datatype = "zvar";

    std::string id_field_name = "psid";
    std::string ra_field_name = "ra";
    std::string dec_field_name = "dec";
    std::string time_field_name = "bjd";
    std::string flux_field_name = "flux";
    std::string fluxErr_field_name = "flux_err";
    // if (datatype == "zforce") {
    //     time_field_name = "mjd";
    // }
    std::string logpath = "/data/swhitebook/zvar_testing/logs";
    std::string output_dir = "/data/swhitebook/zvar_testing/data";

    ProcessingExternalParams params(
        0,
        1, // num_omp_threads
        0.5f, // percent_indices
        hdf_filename,
        logpath,
        output_dir,
        id_field_name,
        ra_field_name,
        dec_field_name,
        time_field_name,
        flux_field_name,
        fluxErr_field_name
    );

    processTEST(params);
}

/*
TEST_CASE("Check GPUFit's Output", "[Statistics]") {
    try {
        int seed = Catch::getSeed(); // Get the seed used by Catch2
        // int seed = 42; // Use a fixed seed for reproducibility
        std::mt19937 gen(seed);      // Use the same seed for your random generator

        //Make 10 fake polynomial light curves
        int N_dat = 1000;
        int N_curves = 1;
        int modelID = 13;
        float* times = new float[N_dat];
        float* y = new float[N_dat * N_curves];
        float* dy = new float[N_dat * N_curves];

        //Generate random times uniformly between 0 and 10
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);
        for (int i = 0; i < N_dat; ++i) {
            times[i] = dist(gen);
        }
        std::sort(times, times + N_dat);

        //Generate random polynomial coefficients for 10 7th degree polynomials
        std::vector<float> coeffs(N_curves * 8);
        for (int i = 0; i < N_curves * 8; ++i) {
            coeffs[i] = dist(gen);
            std::cout << "coeffs[" << i << "] = " << coeffs[i] << std::endl;
        }

        //Generate light curves from the polynomial coefficients
        for (int i = 0; i < N_curves; ++i) {
            for (int j = 0; j < N_dat; ++j) {
                y[i * N_dat + j] = coeffs[i * 8] * std::pow(times[j], 7) +
                                   coeffs[i * 8 + 1] * std::pow(times[j], 6) +
                                   coeffs[i * 8 + 2] * std::pow(times[j], 5) +
                                   coeffs[i * 8 + 3] * std::pow(times[j], 4) +
                                   coeffs[i * 8 + 4] * std::pow(times[j], 3) +
                                   coeffs[i * 8 + 5] * std::pow(times[j], 2) +
                                   coeffs[i * 8 + 6] * times[j] +
                                   coeffs[i * 8 + 7];
            }
        }

        //Set all errors to unity
        std::fill(dy, dy + N_dat * N_curves, 1.0f);

        //mean subtract the light curves
        for (int i = 0; i < N_curves; ++i) {
            auto [mean, stdev] = computeMeanAndStdDev(std::vector<float>(y + i * N_dat, y + (i + 1) * N_dat));
            for (int j = 0; j < N_dat; ++j) {
                y[i * N_dat + j] -= mean;
            }
        }

        float* flux_no_baseline = removeBaseline(times, y, dy, N_dat, N_curves, 15);

        //Print the mean of the first light curve
        auto [mean, stdev] = computeMeanAndStdDev(std::vector<float>(y, y + N_dat));

        //print the first 10 values of the first light curve
        for (int i = 0; i < 10; ++i) {
            std::cout << "y[0][" << i << "] = " << y[i] << std::endl;
        }

        //print the first 10 values of the first light curve with the baseline removed
        for (int i = 0; i < 10; ++i) {
            std::cout << "flux_no_baseline_chunk[0][" << i << "] = " << flux_no_baseline[i] << std::endl;
        }

        // Clean up
        delete[] times;
        delete[] y;
        delete[] dy;

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}

*/
