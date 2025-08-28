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

TEST_CASE("Check Uniqueness of Frequency Grid", "[FPW]") {
    try {
        // Read the data file
        std::string dataFile = "../tests/data/tydy.csv";
        std::vector<std::vector<double>> data = readCSV(dataFile);
        std::vector<double> times = extractColumn(data, 0);

        // Convert times to MJD
        for (size_t i = 0; i < times.size(); ++i) {
            times[i] = times[i] / 86400.0; // Convert to days
        }

        double f_min = 1.0 / 10.0; // Default: 10 day max period
        double f_max = 1440.0 / 5.0; // Default: 5 min min period
        int freq_oversampling = 3; // Default: 3x oversampling

        // Make a frequency grid
        vector<double> fgrid = freq_grid(times, f_min, f_max, freq_oversampling);

        int duplicateCount = 0;
        //Check if the frequency grid contains any duplicate values
        for (size_t i = 1; i < fgrid.size(); ++i) {
            if (fgrid[i] == fgrid[i - 1]) {
                duplicateCount++;
            }
        }
        REQUIRE(duplicateCount == 0); // There should be no duplicates

        // Convert all frequencies to floats
        std::vector<float> fgridFloat(fgrid.size());
        std::transform(fgrid.begin(), fgrid.end(), fgridFloat.begin(),
                       [](double val) { return static_cast<float>(val); });

        // Check for duplicates in the float vector
        duplicateCount = 0;
        for (size_t i = 1; i < fgridFloat.size(); ++i) {
            if (fgridFloat[i] == fgridFloat[i - 1]) {
                duplicateCount++;
            }
        }
        REQUIRE(duplicateCount == 0); // There should be no duplicates in the float vector

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
        int N_dat = 1500000;
        int N_curves = 623;
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
    try {
        const int N_dat = 100000; // Number of data points in the curve
        const int N_curves = 1;    // Test with a single curve
        const int distance_param = 300;
        const float threshold_param = 1.5f; // S/N threshold
        const int N_PEAKS_CONST = 32; // Matching N_peaks macro in peak_utils.cu

        // --- Generate Random Data (Absolute Gaussian) ---
        std::vector<float> h_raw_data(N_dat);
        int seed = Catch::getSeed();
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 5.0f); // Mean 0, Stddev 5
        for (int i = 0; i < N_dat; ++i) {
            h_raw_data[i] = std::abs(dist(gen));
        }

        // --- Calculate Median/MAD and Normalize Data on CPU ---
        float cpu_median_val = median(h_raw_data.data(), N_dat);
        float cpu_mad_val = mad(h_raw_data.data(), N_dat, cpu_median_val);
        float safe_mad = (std::abs(cpu_mad_val) > 1e-12f) ? cpu_mad_val : 1e-12f;
        if (std::abs(cpu_mad_val) <= 1e-12f) {
            std::cout << "Warning: MAD is very small or zero, using epsilon. MAD: " << cpu_mad_val << std::endl;
        }


        std::vector<float> h_normalized_data(N_dat);
        for (int i = 0; i < N_dat; ++i) {
            h_normalized_data[i] = (h_raw_data[i] - cpu_median_val) / safe_mad;
        }

        // --- CPU Peak Finding ---
        // find_peaks expects data, N_peaks, distance, median, mad, threshold.
        // Since h_normalized_data is already S/N, pass median=0, mad=1.
        std::vector<int> cpu_indices_from_func = find_peaks(h_normalized_data, N_PEAKS_CONST, distance_param, 0.0f, 1.0f, threshold_param);
        
        std::vector<int> cpu_final_indices(N_PEAKS_CONST, -1);
        std::vector<float> cpu_final_values(N_PEAKS_CONST, 0.0f);

        for (size_t i = 0; i < cpu_indices_from_func.size(); ++i) {
            if (i < N_PEAKS_CONST) { 
                int peak_idx = cpu_indices_from_func[i];
                if (peak_idx != -1) {
                    cpu_final_indices[i] = peak_idx;
                    cpu_final_values[i] = h_normalized_data[peak_idx];
                }
            }
        }

        // --- GPU Peak Finding ---
        float *d_data_gpu, *d_peakValues_gpu;
        int *d_peakIndices_gpu;

        cudaMalloc(&d_data_gpu, N_dat * sizeof(float));
        cudaMalloc(&d_peakIndices_gpu, N_PEAKS_CONST * sizeof(int)); 
        cudaMalloc(&d_peakValues_gpu, N_PEAKS_CONST * sizeof(float)); 

        cudaMemcpy(d_data_gpu, h_normalized_data.data(), N_dat * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(d_peakIndices_gpu, -1, N_PEAKS_CONST * sizeof(int));
        // Initialize with a value that helps distinguish from valid small positive peaks if necessary,
        // though the kernel itself initializes its local top_peak_values to -INFINITY and output to 0.0f for invalid.
        // Setting to 0.0f to match the kernel's padding for invalid peaks.
        cudaMemset(d_peakValues_gpu, 0, N_PEAKS_CONST * sizeof(float)); 

        dim3 blockSize(1); 
        dim3 gridSize(1); 
        cudaFindPeaks<<<gridSize, blockSize>>>(d_data_gpu, d_peakIndices_gpu, d_peakValues_gpu, N_dat, N_curves, distance_param, threshold_param);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after cudaFindPeaks kernel: " << cudaGetErrorString(err) << std::endl;
            FAIL("CUDA kernel launch failed");
        }
        cudaDeviceSynchronize(); // Ensure kernel completion

        std::vector<int> gpu_peak_indices_vec(N_PEAKS_CONST);
        std::vector<float> gpu_peak_values_vec(N_PEAKS_CONST);
        cudaMemcpy(gpu_peak_indices_vec.data(), d_peakIndices_gpu, N_PEAKS_CONST * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_peak_values_vec.data(), d_peakValues_gpu, N_PEAKS_CONST * sizeof(float), cudaMemcpyDeviceToHost);

        // --- Compare Results ---
        for (int i = 0; i < N_PEAKS_CONST; ++i) {
            CAPTURE("Peak rank", i, "CPU Index", cpu_final_indices[i], "GPU Index", gpu_peak_indices_vec[i], "CPU Value", cpu_final_values[i], "GPU Value", gpu_peak_values_vec[i]);
            REQUIRE(cpu_final_indices[i] == gpu_peak_indices_vec[i]);
            // Values from cudaFindPeaks are actual values from input data, not S/N if it were to do internal normalization.
            // Since we pass normalized data, values should match.
            // The kernel sets value to 0.0f if index is -1.
            const float epsilon = 1e-5f;
            if (cpu_final_indices[i] != -1) {
                 REQUIRE(std::abs(cpu_final_values[i] - gpu_peak_values_vec[i]) < epsilon);
            } else {
                 REQUIRE(std::abs(gpu_peak_values_vec[i]) < epsilon); // Check if GPU value is close to 0.0f
            }
        }

        // --- Free device memory ---
        cudaFree(d_data_gpu);
        cudaFree(d_peakIndices_gpu);
        cudaFree(d_peakValues_gpu);

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}


TEST_CASE("Check GPU Peak Finding vs CPU Timing", "[Statistics][Timing]") {
    try {
        // --- Parameters ---
        const int N_dat = 1000000; // Length of each curve
        const int N_curves = 100; // Number of curves to process
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
        cudaFindPeaks<<<gridSizePeaks, blockSizePeaks>>>(d_data,
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

TEST_CASE("Check Peak Finding Algorithms", "[Statistics]") {
    try {

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            FAIL("CUDA kernel launch failed");
        }

        const int N_curves = 100;
        const int N_dat = 100033;
        const int N_chunks = 5;
        const int chunk_size = (N_dat + N_chunks - 1) / N_chunks; // Max data points in a chunk for one curve
        const int distance = 300;
        const float threshold = 1.0f; // Significance threshold

        int seed = Catch::getSeed();
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Generate random data
        std::vector<float> all_data(N_curves * N_dat);
        for (auto& v : all_data) v = dist(gen);

        // Allocate device memory for all curves
        float *d_data, *d_peakValues, *d_peakValues_chunk;
        int *d_peakIndices, *d_peakIndices_chunk;
        cudaMalloc(&d_data, N_curves * N_dat * sizeof(float));
        cudaMalloc(&d_peakIndices, N_curves * N_peaks * sizeof(int));
        cudaMalloc(&d_peakValues, N_curves * N_peaks * sizeof(float));
        cudaMalloc(&d_peakIndices_chunk, N_curves * N_peaks * sizeof(int));
        cudaMalloc(&d_peakValues_chunk, N_curves * N_peaks * sizeof(float));

        // Compute median and MAD for each curve on CPU
        std::vector<float> medians(N_curves), mads(N_curves);
        for (int i = 0; i < N_curves; ++i) {
            float* curve_ptr = all_data.data() + i * N_dat;
            medians[i] = median(curve_ptr, N_dat);
            mads[i] = mad(curve_ptr, N_dat, medians[i]);
        }

        // Normalize the data using the computed medians and MADs
        for (int i = 0; i < N_curves; ++i) {
            for (int j = 0; j < N_dat; ++j) {
                all_data[i * N_dat + j] = (all_data[i * N_dat + j] - medians[i]) / mads[i];
            }
        }

        std::cout << "Data generated and normalized." << std::endl;

        // Copy data to device
        cudaMemcpy(d_data, all_data.data(), N_curves * N_dat * sizeof(float), cudaMemcpyHostToDevice);

        std::cout << "Data copied to device." << std::endl;

        // --- Run global peak finding ---
        cudaMemset(d_peakIndices, -1, N_curves * N_peaks * sizeof(int));
        cudaMemset(d_peakValues, 0, N_curves * N_peaks * sizeof(float));
        dim3 blockSize(32);
        dim3 gridSize((N_curves + blockSize.x - 1) / blockSize.x);
        cudaFindPeaks<<<gridSize, blockSize>>>(d_data, d_peakIndices, d_peakValues, N_dat, N_curves, distance, threshold);
        cudaDeviceSynchronize();

        std::cout << "Global peak finding completed." << std::endl;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in global peak finding: " << cudaGetErrorString(err) << std::endl;
            FAIL("CUDA kernel launch failed");
        }

        std::vector<int> global_peak_indices(N_curves * N_peaks);
        std::vector<float> global_peak_values(N_curves * N_peaks);
        cudaMemcpy(global_peak_indices.data(), d_peakIndices, N_curves * N_peaks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(global_peak_values.data(), d_peakValues, N_curves * N_peaks * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Global peak indices and values copied to host." << std::endl;

        // --- Prepare for chunked peak finding ---
        // Allocate memory for a single chunk's data on device
        float *d_data_current_chunk, *d_last_chunk_data;
        cudaMalloc(&d_data_current_chunk, N_curves * chunk_size * sizeof(float)); // For one curve
        cudaMalloc(&d_last_chunk_data, N_curves * 2 * sizeof(float)); // For the last chunk

        int *d_state_candidate_idx, *d_state_distLastPeak;
        float *d_state_candidate_val, *d_state_currentThreshold;
        cudaMalloc(&d_state_candidate_idx, N_curves * sizeof(int));
        cudaMalloc(&d_state_candidate_val, N_curves * sizeof(float));
        cudaMalloc(&d_state_distLastPeak, N_curves * sizeof(int));
        cudaMalloc(&d_state_currentThreshold, N_curves * sizeof(float));

        // Initialize persistent state
        std::vector<int> init_idx_vec(N_curves, -1);
        std::vector<float> init_val_vec(N_curves, -INFINITY);
        std::vector<int> init_dist_vec(N_curves, distance); // Initial distLastPeak for chunk processing
        std::vector<float> init_thresh_vec(N_curves, threshold);
        cudaMemcpy(d_state_candidate_idx, init_idx_vec.data(), N_curves * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_state_candidate_val, init_val_vec.data(), N_curves * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_state_distLastPeak, init_dist_vec.data(), N_curves * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_state_currentThreshold, init_thresh_vec.data(), N_curves * sizeof(float), cudaMemcpyHostToDevice);

        std::cout << "Persistent state initialized." << std::endl;

        // Allocate memory for chunk peak arrays
        cudaMalloc(&d_peakIndices_chunk, N_curves * N_peaks * sizeof(int));
        cudaMalloc(&d_peakValues_chunk, N_curves * N_peaks * sizeof(float));

        // Zero out chunk peak arrays
        cudaMemset(d_peakIndices_chunk, -1, N_curves * N_peaks * sizeof(int));
        cudaMemset(d_peakValues_chunk, 0, N_curves * N_peaks * sizeof(float));

        cudaMemset(d_data_current_chunk, 0, N_curves * chunk_size * sizeof(float));
        cudaMemset(d_last_chunk_data, 0, N_curves * 2 * sizeof(float)); // For the last chunk

        std::cout << "Chunk peak arrays initialized." << std::endl;

        // Run chunked peak finding
        for (int c = 0; c < N_chunks; ++c) {
            int chunk_global_start_idx = c * chunk_size;
            // N_dat here is the total number of data points for a single, full curve (e.g., 100000)
            int N_dat_in_this_chunk = std::min(chunk_size, N_dat - chunk_global_start_idx);

            // Correctly copy data for each curve's current chunk
            for (int curve_idx_loop = 0; curve_idx_loop < N_curves; ++curve_idx_loop) {
                // Pointer to the start of the current curve's data in the full device dataset
                float* d_full_curve_start = d_data + (static_cast<size_t>(curve_idx_loop) * N_dat);
                // Pointer to the start of the current chunk for this curve in the full device dataset
                float* d_source_ptr_for_chunk = d_full_curve_start + chunk_global_start_idx;

                // Pointer to where this curve's chunk data should go in d_data_current_chunk
                float* d_dest_ptr_in_chunk_buffer = d_data_current_chunk + (static_cast<size_t>(curve_idx_loop) * N_dat_in_this_chunk);

                cudaError_t memcpy_err = cudaMemcpy(d_dest_ptr_in_chunk_buffer,
                                                    d_source_ptr_for_chunk,
                                                    N_dat_in_this_chunk * sizeof(float),
                                                    cudaMemcpyDeviceToDevice);
                if (memcpy_err != cudaSuccess) {
                    std::cerr << "CUDA Memcpy error during chunk data preparation: " << cudaGetErrorString(memcpy_err) << std::endl;
                    FAIL("CUDA Memcpy failed for chunk data.");
                }
            }

            std::cout << "Processing chunk " << c + 1 << "/" << N_chunks << " with "
                      << N_dat_in_this_chunk << " data points." << std::endl;

            cudaFindPeaksChunk<<<gridSize, blockSize>>>(
                d_data_current_chunk,      // Pointer to the data for the current chunk
                d_last_chunk_data,         // Pointer to the last chunk's data (if needed)
                d_peakIndices_chunk,       // Output global peak indices
                d_peakValues_chunk,        // Output global peak values
                N_dat_in_this_chunk,       // Number of data points per curve IN THIS CHUNK
                N_curves,                  // N_curves is 1 for this test
                distance,
                threshold,
                chunk_global_start_idx, // Global starting index of this chunk
                d_state_candidate_idx,     // Persistent state: global candidate peak index
                d_state_candidate_val,     // Persistent state: candidate peak value
                d_state_distLastPeak,      // Persistent state: distance from last
                d_state_currentThreshold  // Persistent state: current dynamic threshold
            );
            cudaDeviceSynchronize(); // Ensure kernel completes before next chunk processing

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in chunk peak finding: " << cudaGetErrorString(err) << std::endl;
                FAIL("CUDA kernel launch failed");
            }

            // Copy the last two values of the current chunk to d_last_chunk_data
            for (int i = 0; i < N_curves; ++i) {
                int last_idx = (i * N_dat) + chunk_global_start_idx + N_dat_in_this_chunk - 1;
                int second_last_idx = (i * N_dat) + chunk_global_start_idx + N_dat_in_this_chunk - 2;
                float last_value = all_data[last_idx];
                float second_last_value = all_data[second_last_idx];
                cudaMemcpy(d_last_chunk_data + i * 2, &second_last_value, sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_last_chunk_data + i * 2 + 1, &last_value, sizeof(float), cudaMemcpyHostToDevice);
            }
        }

        // Download persistent state after last chunk
        std::vector<int> final_candidate_idx(N_curves);
        std::vector<float> final_candidate_val(N_curves);
        std::vector<int> final_dist_last_peak(N_curves);
        std::vector<float> final_current_threshold(N_curves);
        cudaMemcpy(final_candidate_idx.data(), d_state_candidate_idx, N_curves * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_candidate_val.data(), d_state_candidate_val, N_curves * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_dist_last_peak.data(), d_state_distLastPeak, N_curves * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_current_threshold.data(), d_state_currentThreshold, N_curves * sizeof(float), cudaMemcpyDeviceToHost);

        // Download chunk peaks for update
        std::vector<int> peakIndices_chunk(N_curves * N_peaks);
        std::vector<float> peakValues_chunk(N_curves * N_peaks);
        cudaMemcpy(peakIndices_chunk.data(), d_peakIndices_chunk, N_curves * N_peaks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(peakValues_chunk.data(), d_peakValues_chunk, N_curves * N_peaks * sizeof(float), cudaMemcpyDeviceToHost);

        // Now compare as before
        for (int i = 0; i < N_curves; ++i) {
            for (int j = 0; j < N_peaks; ++j) {
                CAPTURE("Curve", i, "Peak", j);
                int global_idx = i * N_peaks + j;
                int chunk_idx = i * N_peaks + j;
                REQUIRE(global_peak_indices[global_idx] == peakIndices_chunk[chunk_idx]);
                REQUIRE(global_peak_values[global_idx] == peakValues_chunk[chunk_idx]);
            }
        }

        // Free device memory
        cudaFree(d_data);
        cudaFree(d_peakIndices);
        cudaFree(d_peakValues);
        cudaFree(d_data_current_chunk);
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

    // processTEST(params);
    process(params);
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

TEST_CASE("Check Simulated Periodograms", "[FPW]") {
    try {
        int N_dat = 500;
        int N_curves = 1000;
        int N_freqs = 537130;

        // Load simulated data
        float* sim_times = new float[N_dat];
        float* sim_flux = new float[N_dat * N_curves];
        float* sim_ivar = new float[N_dat * N_curves];
        float* sim_fgrid = new float[N_freqs];

        // Load sim_times (single column)
        std::string timesFile = "../tests/data/sim_times.csv";
        std::vector<std::vector<double>> timesData = readCSV(timesFile);
        std::vector<double> timesVec = extractColumn(timesData, 0);
        for (int i = 0; i < N_dat; ++i) {
            sim_times[i] = static_cast<float>(timesVec[i]);
        }

        // Load sim_fgrid (single column)
        std::string fgridFile = "../tests/data/sim_freqs.csv";
        std::vector<std::vector<double>> fgridData = readCSV(fgridFile);
        std::vector<double> fgridVec = extractColumn(fgridData, 0);
        for (int i = 0; i < N_freqs; ++i) {
            sim_fgrid[i] = static_cast<float>(fgridVec[i]);
        }

        // Load sim_flux (multiple columns)
        std::string fluxFile = "../tests/data/sim_flux.csv";
        std::vector<std::vector<double>> fluxData = readCSV(fluxFile);
        for (int curve = 0; curve < N_curves; ++curve) {
            std::vector<double> fluxCol = extractColumn(fluxData, curve);
            for (int i = 0; i < N_dat; ++i) {
                sim_flux[curve * N_dat + i] = static_cast<float>(fluxCol[i]);
            }
        }

        // Load sim_flux_errs (multiple columns)
        std::string fluxErrsFile = "../tests/data/sim_flux_errs.csv";
        std::vector<std::vector<double>> fluxErrsData = readCSV(fluxErrsFile);
        for (int curve = 0; curve < N_curves; ++curve) {
            std::vector<double> fluxErrCol = extractColumn(fluxErrsData, curve);
            for (int i = 0; i < N_dat; ++i) {
                sim_ivar[curve * N_dat + i] = 1/ (static_cast<float>(fluxErrCol[i]) * static_cast<float>(fluxErrCol[i]));
            }
        }

        //Print the first 10 flux values of curve 18
        for (int i = 0; i < 10; ++i) {
            std::cout << "sim_flux[18][" << i << "] = " << sim_flux[18 * N_dat + i] << std::endl;
        }

        std::vector<int> N_bins = {20, 10, 5}; // Default: 20, 10, 5 bins

        float* t_device = nullptr;
        cudaMalloc(&t_device, N_dat * sizeof(float));

        float* y_device = nullptr;
        cudaMalloc(&y_device, N_curves * N_dat * sizeof(float));

        float* ivar_device = nullptr;
        cudaMalloc(&ivar_device, N_curves * N_dat * sizeof(float));

        float* freqs_device = nullptr;
        cudaMalloc(&freqs_device, N_freqs * sizeof(float));

        unsigned char* indices_device = nullptr;
        cudaMalloc(&indices_device, N_freqs * N_dat * sizeof(unsigned char));

        int* N_bins_device = nullptr;
        cudaMalloc(&N_bins_device, 3 * sizeof(int));

        float* periodograms_device = nullptr;
        cudaMalloc(&periodograms_device, N_curves * N_freqs * 3 * sizeof(float));

        float* periodograms_medians_device = nullptr;
        cudaMalloc(&periodograms_medians_device, N_curves * 3 * sizeof(float));

        float* periodograms_mads_device = nullptr;
        cudaMalloc(&periodograms_mads_device, N_curves * 3 * sizeof(float));

        cudaMemcpy(t_device, sim_times, N_dat * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y_device, sim_flux, N_curves * N_dat * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ivar_device, sim_ivar, N_curves * N_dat * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(freqs_device, sim_fgrid, N_freqs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(N_bins_device, N_bins.data(), 3 * sizeof(int), cudaMemcpyHostToDevice);

        dim3 block_size_indices(32, 32);
        dim3 grid_size_indices((N_freqs + block_size_indices.x - 1) / block_size_indices.x,
                                (N_dat + block_size_indices.y - 1) / block_size_indices.y);

        cudaMakeIndices<<<grid_size_indices,
                          block_size_indices>>>(t_device,
                                                freqs_device,
                                                indices_device,
                                                20,
                                                N_dat,
                                                N_freqs);
        
        cudaDeviceSynchronize();

          dim3 block_size_deltaChi2(32, 8);
          dim3 grid_size_deltaChi2((N_freqs + block_size_deltaChi2.x - 1) / block_size_deltaChi2.x,
                                (N_curves + block_size_deltaChi2.y - 1) / block_size_deltaChi2.y);
          size_t shared_memory_deltaChi2 = 2 * 20 * block_size_deltaChi2.x * block_size_deltaChi2.y * sizeof(float);

        cudaDeltaChi2Multibin<<<grid_size_deltaChi2,
                                block_size_deltaChi2,
                                shared_memory_deltaChi2>>>(y_device,
                                                            ivar_device,
                                                            indices_device,
                                                            periodograms_device,
                                                            N_bins_device,
                                                            N_dat,
                                                            N_freqs,
                                                            N_curves,
                                                            3);
        cudaDeviceSynchronize();

        std::vector<float> debug_periodogram(N_freqs);
        cudaMemcpy(debug_periodogram.data(),
                periodograms_device + 30 * N_freqs * 3, // 20th curve, sample 0
                N_freqs * sizeof(float),
                cudaMemcpyDeviceToHost);
        for (int i = 100; i < 110; ++i) std::cout << debug_periodogram[i] << std::endl;

        dim3 block_size_postprocess(512);
        dim3 grid_size_postprocess((N_curves * 3 + block_size_postprocess.x - 1) / block_size_postprocess.x);
        CudaComputeMedianandMAD<<<grid_size_postprocess,
                            block_size_postprocess>>>(periodograms_device,
                                                    periodograms_medians_device,
                                                    periodograms_mads_device,
                                                    N_freqs,
                                                    N_curves * 3);
        cudaDeviceSynchronize();

        //Copy medians and mads back to the host
        float* periodogram_medians_host = new float[N_curves * 3];
        float* periodogram_mads_host = new float[N_curves * 3];
        cudaMemcpy(periodogram_medians_host, periodograms_medians_device, N_curves * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(periodogram_mads_host, periodograms_mads_device, N_curves * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        //Print the median of the 20th curve

        std::vector<float> periodogram_medians_vec(periodogram_medians_host, periodogram_medians_host + N_curves * 3);
        std::vector<float> periodogram_mads_vec(periodogram_mads_host, periodogram_mads_host + N_curves * 3);

        //Save the results
        std::string sim_periodogram_stats_file = "sim_periodogram_stats";
        writePeriodogramStatisticsToCSV(sim_periodogram_stats_file,
                                        periodogram_medians_vec,
                                        periodogram_mads_vec,
                                        0,
                                        N_curves,
                                        3,
                                        true);


        // Clean up
        delete[] sim_times;
        delete[] sim_fgrid;
        delete[] sim_flux;
        delete[] sim_ivar;

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}
