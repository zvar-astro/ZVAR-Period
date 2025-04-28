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

        // Compute the mean and standard deviation of the difference
        auto [meanDiff, stdevDiff] = computeMeanAndStdDev(diff);

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
