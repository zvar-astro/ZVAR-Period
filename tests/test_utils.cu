#include "test_utils.cuh"

// Function to read a CSV file into a 2D vector
std::vector<std::vector<double>> readCSV(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to extract a column from a 2D vector
std::vector<double> extractColumn(const std::vector<std::vector<double>>& data, size_t colIndex) {
    std::vector<double> column;
    for (const auto& row : data) {
        if (colIndex < row.size()) {
            column.push_back(row[colIndex]);
        } else {
            throw std::out_of_range("Column index out of range");
        }
    }
    return column;
}

//Function to compute the mean and standard deviation of a vector
std::pair<float, float> computeMeanAndStdDev(const std::vector<float>& data) {
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    float sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f);
    float stdev = std::sqrt(sq_sum / data.size() - mean * mean);
    return {mean, stdev};
}

// Helper function: Partition the array around a pivot
size_t partitionArray(std::vector<float>& data, size_t left, size_t right, size_t pivotIndex) {
    float pivotValue = data[pivotIndex];
    std::swap(data[pivotIndex], data[right]); // Move pivot to the end
    size_t storeIndex = left;

    for (size_t i = left; i < right; ++i) {
        if (data[i] < pivotValue) {
            std::swap(data[i], data[storeIndex]);
            ++storeIndex;
        }
    }

    std::swap(data[storeIndex], data[right]); // Move pivot to its final place
    return storeIndex;
}

// Helper function: Quickselect to find the k-th smallest element
float quickselect(std::vector<float>& data, size_t k) {
    size_t left = 0, right = data.size() - 1;

    while (true) {
        if (left == right) {
            return data[left];
        }

        size_t pivotIndex = left + (right - left) / 2;
        pivotIndex = partitionArray(data, left, right, pivotIndex);

        if (k == pivotIndex) {
            return data[k];
        } else if (k < pivotIndex) {
            right = pivotIndex - 1;
        } else {
            left = pivotIndex + 1;
        }
    }
}

// Function to compute the median and MAD using Quickselect
std::pair<float, float> computeMedianAndMAD(std::vector<float>& data) {
    size_t n = data.size();
    if (n == 0) {
        throw std::invalid_argument("The input array is empty.");
    }

    // Compute the median using Quickselect
    float median;
    if (n % 2 == 0) {
        float mid1 = quickselect(data, n / 2 - 1);
        float mid2 = quickselect(data, n / 2);
        median = (mid1 + mid2) / 2.0f;
    } else {
        median = quickselect(data, n / 2);
    }

    // Compute the absolute deviations from the median
    std::vector<float> absDev(n);
    std::transform(data.begin(), data.end(), absDev.begin(), [median](float x) { return std::abs(x - median); });

    // Compute the MAD using Quickselect
    float mad;
    if (n % 2 == 0) {
        float mid1 = quickselect(absDev, n / 2 - 1);
        float mid2 = quickselect(absDev, n / 2);
        mad = (mid1 + mid2) / 2.0f;
    } else {
        mad = quickselect(absDev, n / 2);
    }

    return {median, mad};
}

// Function to convert a vector to a string for debugging
std::string vectorToString(const std::vector<float>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

void processTEST(const ProcessingExternalParams& params) {

    auto start_full = chrono::high_resolution_clock::now(); //Time full run
    auto start_preprocess = chrono::high_resolution_clock::now(); //Time loading and pre-processing

    logMessage(params.log_path, "Running on file: " + params.input_hdf_filename);

    // --- Initialize Frame Data ---
    FieldData field_data = loadAndPrepareFieldData(params);

    if (!field_data.success) {
        logMessage(params.log_path, "Failed to load field data. Exiting.");
        exit(EXIT_FAILURE);
    }

    // If there are no exposures, skip the file
    if (field_data.N_exp == 0) {
        logMessage(params.log_path, "No exposures found in file, skipping");
        return;
    }

    // --- Create Frequency Grid ---
    createAndFilterFrequencyGrid(params, field_data); //loads frequency grid into field_data

    if(!field_data.success) {
        logMessage(params.log_path, "Failed to create frequency grid. Exiting.");
        exit(EXIT_FAILURE);
    }

    // --- Initialize Chunking Parameters ---
    initializeChunkParameters(params, field_data); //loads chunk parameters into field_data

    if (!field_data.success) {
        logMessage(params.log_path, "Failed to initialize chunk parameters. Exiting.");
        exit(EXIT_FAILURE);
    }
    
    // --- Initialize Chunk Source Data ---
    int status;
    unsigned long objstart_periodograms, objnum_periodograms;
    SourcesData sources_data(params, field_data); // Initialize sources data with field data

    // --- Initialize Kernel Launch Parameters ---
    KernelLaunchParams kernel_params(params, field_data);

    // --- Initialize Device Memory ---
    DeviceArrays device_arrays(params, field_data);
    PeakPersistentArrays peak_arrays(params, field_data);

    auto stop_preprocess = chrono::high_resolution_clock::now();
    auto duration_preprocess = chrono::duration_cast<chrono::milliseconds>(stop_preprocess - start_preprocess);
    logMessage(params.log_path, "Time to load and pre-process: " + to_string(duration_preprocess.count()) + " milliseconds");

    // print how many sources we have in total, how many chunks we will run, how many sources per chunk
    // and how many exposures per source
    logMessage(params.log_path, 
        "Total sources: " + to_string(field_data.N_src) + 
        ", Total source chunks: " + to_string(field_data.N_src_batch) + 
        ", Sources per chunk: " + to_string(field_data.N_src_per_batch) + 
        ", Exposures per source: " + to_string(field_data.N_exp) + 
        ", Total frequency chunks: " + to_string(field_data.N_freqs_batch) +
        ", Frequencies per chunk: " + to_string(field_data.N_freqs_per_batch));

    // for (unsigned long k = 0; k < field_data.N_src_batch; k++) {
    for (unsigned long k = 0; k < 1; k++) {
        auto start_loadchunk = chrono::high_resolution_clock::now(); // Time loading chunk

        // Get the range of objects to analyze
        objstart_periodograms = k * field_data.N_src_per_batch;
        unsigned long objnum_periodograms = (k < field_data.N_src_batch - 1)
            ? field_data.N_src_per_batch
            : (field_data.N_src - objstart_periodograms);

        // Load the flux and fluxerr values
        loadChunkData(params, field_data, sources_data, objstart_periodograms, objnum_periodograms);

        auto stop_loadchunk = chrono::high_resolution_clock::now();
        auto duration_loadchunk = chrono::duration_cast<chrono::milliseconds>(stop_loadchunk - start_loadchunk);
        logMessage(params.log_path, "Time to load chunk: " + to_string(duration_loadchunk.count()) + " milliseconds");

        auto start_removeOutliers = chrono::high_resolution_clock::now(); // Time removing outliers

        // Call removeOutliers here
        removeOutliers(field_data, sources_data, device_arrays, kernel_params, objnum_periodograms);

        auto stop_removeOutliers = chrono::high_resolution_clock::now();
        auto duration_removeOutliers = chrono::duration_cast<chrono::milliseconds>(stop_removeOutliers - start_removeOutliers);
        logMessage(params.log_path, "Time to remove outliers: " + to_string(duration_removeOutliers.count()) + " milliseconds");

        auto start_removebaseline = chrono::high_resolution_clock::now(); // Time removing baseline

        processBaselineRemoval(params, field_data, sources_data, objstart_periodograms, objnum_periodograms);

        if (!sources_data.success) {
            logMessage(params.log_path, "Failed to remove baseline. Exiting.");
            exit(EXIT_FAILURE);
        }

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error cudaDeviceSynchronize 1: " << cudaGetErrorString(cudaStatus) << std::endl;
            exit(EXIT_FAILURE);
        }

        auto stop_removebaseline = chrono::high_resolution_clock::now();
        auto duration_removebaseline = chrono::duration_cast<chrono::milliseconds>(stop_removebaseline - start_removebaseline);
        logMessage(params.log_path, "Time to remove baseline: " + to_string(duration_removebaseline.count()) + " milliseconds");

        // --- DEBUGGING
        if (k == 0) { // or (k == field_data.N_src_batch - 1) if you use all batches
            unsigned long last_source_idx = objstart_periodograms + objnum_periodograms - 10;
            std::string csv_filename = "last_source_lc.csv";
            std::ofstream csv(csv_filename);
            for (unsigned int e = 0; e < field_data.N_exp; ++e) {
                float t = field_data.times[e];
                float flux = sources_data.fluxes[last_source_idx * field_data.N_exp + e];
                float flux_err = sources_data.fluxErrs[last_source_idx * field_data.N_exp + e];
                float flux_no_baseline = sources_data.fluxes_no_baseline[last_source_idx * field_data.N_exp + e];
                csv << std::setprecision(10) << t << "," << flux << "," << flux_err << "," << flux_no_baseline << "\n";
            }
            csv.close();
            logMessage(params.log_path, "Saved last source of last chunk to " + csv_filename);
        }

        // Copy host fluxes and flux errors to device
        copyPhotToDevice(field_data, sources_data, device_arrays, objnum_periodograms);

        for (unsigned long i = 0; i < field_data.N_freqs; i += field_data.N_freqs_per_batch) {
            unsigned long N_freqs_current = std::min(field_data.N_freqs_per_batch, field_data.N_freqs - i);

            //Time the periodogram computation
            auto start_periodogram = chrono::high_resolution_clock::now(); // Time computing periodograms

            cudaMemcpy(device_arrays.freqs_device, field_data.fgrid.data() + i, N_freqs_current * sizeof(float), cudaMemcpyHostToDevice);

            // Compute indices for the current chunk
            cudaMakeIndices<<<kernel_params.grid_size_indices,
                              kernel_params.block_size_indices>>>(device_arrays.t_device,
                                                                  device_arrays.freqs_device,
                                                                  device_arrays.indices_device,
                                                                  params.N_bins[0],
                                                                  field_data.N_exp,
                                                                  static_cast<int>(N_freqs_current));    
            cudaDeviceSynchronize();

            // Compute the periodograms for the current chunk
            cudaDeltaChi2Multibin<<<kernel_params.grid_size_deltaChi2,
                                    kernel_params.block_size_deltaChi2,
                                    kernel_params.shared_memory_deltaChi2>>>(device_arrays.y_device,
                                                                             device_arrays.ivar_device,
                                                                             device_arrays.indices_device,
                                                                             device_arrays.periodograms_device,
                                                                             device_arrays.N_bins_device,
                                                                             field_data.N_exp,
                                                                             static_cast<int>(N_freqs_current),
                                                                             static_cast<int>(objnum_periodograms),
                                                                             params.N_samples);
            cudaDeviceSynchronize();

            if (k == 0) {
                unsigned long last_source_idx = objstart_periodograms + objnum_periodograms - 10;
                std::string csv_filename = "last_source_periodogram.csv";
                std::ofstream csv(csv_filename);

                // Copy periodogram for this source from device to host
                std::vector<float> periodogram_host(N_freqs_current);
                cudaMemcpy(periodogram_host.data(),
                        // device_arrays.periodograms_device + last_source_idx * N_freqs_current * params.N_samples + N_freqs_current * 2,
                        device_arrays.periodograms_device + last_source_idx * N_freqs_current * params.N_samples,
                        N_freqs_current * sizeof(float),
                        cudaMemcpyDeviceToHost);

                // Save frequency and periodogram value
                for (unsigned int f = 0; f < N_freqs_current; ++f) {
                    float freq = field_data.fgrid[i + f];
                    float periodogram_val = periodogram_host[f];
                    csv << std::setprecision(10) << freq << "," << periodogram_val << "\n";
                }
                csv.close();
                logMessage(params.log_path, "Saved last source periodogram of first chunk to " + csv_filename);
            }

            

            auto stop_periodogram = chrono::high_resolution_clock::now();
            auto duration_periodogram = chrono::duration_cast<chrono::milliseconds>(stop_periodogram - start_periodogram);
            logMessage(params.log_path, "Time to compute periodograms: " + to_string(duration_periodogram.count()) + " milliseconds");

            // If this is the first chunk, compute the median and MAD
            if (i == 0) {

                // Time the median and MAD computation
                auto start_median = chrono::high_resolution_clock::now(); // Time computing median and MAD

                //Check the sizing of the periodograms
                std::cout << "We have : " << static_cast<int>(objnum_periodograms) * params.N_samples << " periodograms of size " << static_cast<int>(N_freqs_current) << std::endl;

                CudaComputeMedianandMAD<<<kernel_params.grid_size_postprocess,
                                          kernel_params.block_size_postprocess>>>(device_arrays.periodograms_device,
                                                                                  device_arrays.periodogram_medians_device,
                                                                                  device_arrays.periodogram_mads_device,
                                                                                  static_cast<int>(N_freqs_current),
                                                                                  static_cast<int>(objnum_periodograms) * params.N_samples);
                
                cudaDeviceSynchronize();
                auto stop_median = chrono::high_resolution_clock::now();
                auto duration_median = chrono::duration_cast<chrono::milliseconds>(stop_median - start_median);
                logMessage(params.log_path, "Time to compute median and MAD: " + to_string(duration_median.count()) + " milliseconds");

                // Transfer medians and MADs to host for the first chunk
                std::vector<float> periodogram_medians_host(static_cast<int>(objnum_periodograms) * params.N_samples);
                std::vector<float> periodogram_mads_host(static_cast<int>(objnum_periodograms) * params.N_samples);

                cudaMemcpy(periodogram_medians_host.data(),
                           device_arrays.periodogram_medians_device,
                           static_cast<int>(objnum_periodograms) * params.N_samples * sizeof(float),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(periodogram_mads_host.data(),
                           device_arrays.periodogram_mads_device,
                           static_cast<int>(objnum_periodograms) * params.N_samples * sizeof(float),
                           cudaMemcpyDeviceToHost);

                // Print the median and MAD for the 10th last source
                unsigned long last_source_idx = objstart_periodograms + objnum_periodograms - 10;
                float median = periodogram_medians_host[last_source_idx * params.N_samples];
                float mad = periodogram_mads_host[last_source_idx * params.N_samples];
                std::cout << "Last source " << last_source_idx << ": Median = " << median << ", MAD = " << mad << std::endl;
                
            }

            // Time the peak finding
            auto start_findpeaks = chrono::high_resolution_clock::now(); // Time finding peaks

            std::cout << "peak threshold: " << params.peak_threshold << std::endl;
            std::cout << "peak distance: " << params.peak_distance << std::endl;

            // Print the first 10 values of the periodogram for the 10th last source
            unsigned long last_source_idx = objstart_periodograms + objnum_periodograms - 10;
            std::vector<float> periodogram_host(N_freqs_current);
            cudaMemcpy(periodogram_host.data(),
                       device_arrays.periodograms_device + last_source_idx * N_freqs_current * params.N_samples,
                       N_freqs_current * sizeof(float),
                       cudaMemcpyDeviceToHost);
            std::cout << "Last source " << last_source_idx << " periodogram values BEFORE FindPeaks: ";
            for (unsigned int f = 0; f < 10; ++f) {
                std::cout << periodogram_host[f] << " ";
            }
            std::cout << std::endl;

            // Find peaks in this chunk
            // cudaFindPeaksChunk<<<kernel_params.grid_size_postprocess,
            //             kernel_params.block_size_postprocess>>>(device_arrays.periodograms_device,
            //                                                     device_arrays.periodogram_medians_device,
            //                                                     device_arrays.periodogram_mads_device,
            //                                                     device_arrays.peak_indices_device,
            //                                                     device_arrays.peak_values_device,
            //                                                     static_cast<int>(N_freqs_current),
            //                                                     static_cast<int>(objnum_periodograms) * params.N_samples,
            //                                                     params.peak_distance,
            //                                                     params.peak_threshold,
            //                                                     peak_arrays.state_candidate_idx,
            //                                                     peak_arrays.state_candidate_vals,
            //                                                     peak_arrays.state_dist_last_peak,
            //                                                     peak_arrays.state_current_threshold,
            //                                                     static_cast<int>(i),
            //                                                     static_cast<int>(i + N_freqs_current));

            cudaDeviceSynchronize();
            auto stop_findpeaks = chrono::high_resolution_clock::now();
            auto duration_findpeaks = chrono::duration_cast<chrono::milliseconds>(stop_findpeaks - start_findpeaks);
            logMessage(params.log_path, "Time to find peaks: " + to_string(duration_findpeaks.count()) + " milliseconds");

            cudaMemcpy(periodogram_host.data(),
                       device_arrays.periodograms_device + last_source_idx * N_freqs_current * params.N_samples,
                       N_freqs_current * sizeof(float),
                       cudaMemcpyDeviceToHost);
            std::cout << "Last source " << last_source_idx << " periodogram values AFTER FindPeaks: ";
            for (unsigned int f = 0; f < 10; ++f) {
                std::cout << periodogram_host[f] << " ";
            }
            std::cout << std::endl;
        }

        // Copy the peak indices and values back to the host
        copyPeaksToHost(params, field_data, sources_data, device_arrays, objnum_periodograms);

        // Check if ANY peak_indices are not -1
        for (unsigned long i = 0; i < objnum_periodograms * params.N_samples; ++i) {
            for (unsigned int f = 0; f < N_peaks; ++f) {
                if (sources_data.peak_indices[i * N_peaks + f] != -1) {
                    std::cout << "Found peaks for source " << (objstart_periodograms + i) << " at frequency index " << f << std::endl;
                    break; // Exit the loop if we found at least one peak
                }
            }
        }

        if (k == 0) {
            unsigned long last_source_idx = objstart_periodograms + objnum_periodograms - 10;
            std::string csv_filename = "last_source_peaks.csv";
            std::ofstream csv(csv_filename);

            // Save peak indices and values for the last source
            for (unsigned int f = 0; f < N_peaks; ++f) {
                int peak_index = sources_data.peak_indices[last_source_idx * N_peaks * params.N_samples + f];
                float peak_value = sources_data.peak_values[last_source_idx * N_peaks * params.N_samples + f];
                std::cout << "Peak index: " << peak_index << ", Peak value: " << peak_value << std::endl;
                csv << std::setprecision(10) << peak_index << "," << peak_value << "\n";
            }
            csv.close();
            logMessage(params.log_path, "Saved last source peaks of first chunk to " + csv_filename);
        }

        // Reset the persistent arrays for the next chunk
        resetStateArrays(peak_arrays, params, field_data);
    }

    auto stop_full = chrono::high_resolution_clock::now();
    auto duration_full = chrono::duration_cast<chrono::seconds>(stop_full - start_full);
    logMessage(params.log_path, "matchfile " + params.input_hdf_filename + " finished in " + to_string(duration_full.count()) + " seconds");
}