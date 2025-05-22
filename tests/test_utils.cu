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

    // --- Initialize Devices ---
    // omp_set_num_threads(params.num_omp_threads); // Set the number of threads for OpenMP

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

    // --- Initialize Memory ---
    initializeChunkParameters(params, field_data); //loads chunk parameters into field_data

    if (!field_data.success) {
        logMessage(params.log_path, "Failed to initialize chunk parameters. Exiting.");
        exit(EXIT_FAILURE);
    }
    
    // --- Initialize Chunk Source Data ---
    int status;
    unsigned long objstart_periodograms, objnum_periodograms;
    SourcesData sources_data(field_data); // Initialize sources data with field data

    // ---Create Output File---
    // std::string results_filename = createOutputFilepath(params.output_dir, params.input_hdf_filename);
    // createResults(results_filename, field_data.N_src, params.N_samples, N_peaks);

    auto stop_preprocess = chrono::high_resolution_clock::now();
    auto duration_preprocess = chrono::duration_cast<chrono::milliseconds>(stop_preprocess - start_preprocess);
    logMessage(params.log_path, "Time to load and pre-process: " + to_string(duration_preprocess.count()) + " milliseconds");

    // print how many sources we have in total, how many chunks we will run, how many sources per chunk
    // and how many exposures per source
    logMessage(params.log_path, 
        "Total sources: " + to_string(field_data.N_src) + 
        ", Total source chunks: " + to_string(field_data.N_src_batch) + 
        ", Sources per chunk: " + to_string(field_data.N_src_per_batch) + 
        ", Exposures per source: " + to_string(field_data.N_exp));

    // Readout for diagnostics
    std::vector<float> periodogram_full(field_data.N_freqs, 0.0f);

    for (unsigned long k = 1; k < 2; k++) {

        std::cout << "Processing chunk " << k + 1 << " of " << field_data.N_src_batch << std::endl;

        auto start_loadchunk = chrono::high_resolution_clock::now(); // Time loading chunk

        // Get the range of objects to analyze
        objstart_periodograms = k * field_data.N_src_per_batch;
        unsigned long objnum_periodograms = (k < field_data.N_src_batch - 1)
            ? field_data.N_src_per_batch
            : (field_data.N_src - objstart_periodograms);

        // Load the flux and fluxerr values
        // TODO: pass the name of the flux and fluxerr columns as arguments (??? - Sam)
        loadChunkData(params, field_data, sources_data, objstart_periodograms, objnum_periodograms);

        std::cout << "Loaded chunk data for " << objnum_periodograms << " sources." << std::endl;

        auto stop_loadchunk = chrono::high_resolution_clock::now();
        auto duration_loadchunk = chrono::duration_cast<chrono::milliseconds>(stop_loadchunk - start_loadchunk);

        auto start_removebaseline = chrono::high_resolution_clock::now(); // Time removing baseline


        processBaselineRemoval(params, field_data, sources_data, objstart_periodograms, objnum_periodograms);

        if (!sources_data.success) {
            logMessage(params.log_path, "Failed to remove baseline. Exiting.");
            exit(EXIT_FAILURE);
        }

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error cudaDeviceSynchronize: " << cudaGetErrorString(cudaStatus) << std::endl;
            exit(EXIT_FAILURE);
        }


        auto stop_removebaseline = chrono::high_resolution_clock::now();
        auto duration_removebaseline = chrono::duration_cast<chrono::milliseconds>(stop_removebaseline - start_removebaseline);
        logMessage(params.log_path, "Time to remove baseline: " + to_string(duration_removebaseline.count()) + " milliseconds");

        // Run FPW

        // Allocate persistent state arrays
        int *d_state_candidate_idx, *d_state_distLastPeak, *d_peak_indices;
        float *d_state_candidate_val, *d_state_currentThreshold, *d_peak_values;
        cudaMalloc(&d_state_candidate_idx, objnum_periodograms * params.N_samples * sizeof(int));
        cudaMalloc(&d_state_candidate_val, objnum_periodograms * params.N_samples * sizeof(float));
        cudaMalloc(&d_state_distLastPeak, objnum_periodograms * params.N_samples * sizeof(int));
        cudaMalloc(&d_state_currentThreshold, objnum_periodograms * params.N_samples * sizeof(float));
        cudaMalloc(&d_peak_indices, objnum_periodograms * params.N_samples * N_peaks * sizeof(int));
        cudaMalloc(&d_peak_values, objnum_periodograms * params.N_samples * N_peaks * sizeof(float));

        // Initialize persistent state
        cudaMemsetAsync(d_state_candidate_idx, -1, objnum_periodograms * params.N_samples * sizeof(int));
        cudaMemsetAsync(d_state_candidate_val, -INFINITY, objnum_periodograms * params.N_samples * sizeof(float));
        cudaMemsetAsync(d_state_distLastPeak, 0, objnum_periodograms * params.N_samples * sizeof(int));
        cudaMemsetAsync(d_state_currentThreshold, params.peak_threshold, objnum_periodograms * params.N_samples * sizeof(float));
        cudaMemsetAsync(d_peak_indices, -1, objnum_periodograms * params.N_samples * N_peaks * sizeof(int));
        cudaMemsetAsync(d_peak_values, -INFINITY, objnum_periodograms * params.N_samples * N_peaks * sizeof(float));

        std::cout << "Allocated device memory for persistent state." << std::endl;

        int threadsPerBlock = 256;
        dim3 blockSizePostprocess(threadsPerBlock);
        dim3 gridSizePostprocess((objnum_periodograms * params.N_samples + threadsPerBlock - 1) / threadsPerBlock);

        float* device_medians;
        float* device_mads;
        cudaMalloc(&device_medians, objnum_periodograms * sizeof(float)); // Allocate memory for GPU medians
        cudaMalloc(&device_mads, objnum_periodograms * sizeof(float));    // Allocate memory for GPU MADs

        // SAVE LIGHTCURVES FOR DIAGNOSTICS ---
        unsigned long last_src_global = objstart_periodograms + objnum_periodograms - 1;
        unsigned long last_src_local = objnum_periodograms - 1;

        std::vector<float> lightcurve(field_data.N_exp, 0.0f);
        std::vector<float> lightcurve_no_baseline(field_data.N_exp, 0.0f);
        std::vector<float> lightcurve_errors(field_data.N_exp, 0.0f);

        if (last_src_local < objnum_periodograms) {
            std::copy(
                sources_data.fluxes.begin() + last_src_local * field_data.N_exp,
                sources_data.fluxes.begin() + (last_src_local + 1) * field_data.N_exp,
                lightcurve.begin()
            );
            std::copy(
                sources_data.fluxes_no_baseline.begin() + last_src_local * field_data.N_exp,
                sources_data.fluxes_no_baseline.begin() + (last_src_local + 1) * field_data.N_exp,
                lightcurve_no_baseline.begin()
            );
            std::copy(
                sources_data.fluxErrs.begin() + last_src_local * field_data.N_exp,
                sources_data.fluxErrs.begin() + (last_src_local + 1) * field_data.N_exp,
                lightcurve_errors.begin()
            );

        std::cout << "Saved lightcurve for last source." << std::endl;
        } else {
            std::cerr << "last_src_local (" << last_src_local << ") is out of bounds for objnum_periodograms (" << objnum_periodograms << ")" << std::endl;
        }

        // Time the frequency loop

        auto start_freqloop = chrono::high_resolution_clock::now(); // Time frequency loop

        unsigned long freq_offset = 0;
        for (unsigned long i = 0; i < field_data.N_freqs; i += field_data.N_freqs_per_batch) {
            unsigned long N_freqs_current = std::min(field_data.N_freqs_per_batch, field_data.N_freqs - i);

            // Compute indices for the current chunk
            unsigned char* device_indices_chunk = createIndices(field_data.times.data(), field_data.fgrid.data() + i, params.N_bins[0], field_data.N_exp, N_freqs_current);
            cudaDeviceSynchronize();

            //Print the length of sources_data.fluxes_no_baseline
            std::cout << "Length of sources_data.fluxes_no_baseline: " << sources_data.fluxes_no_baseline.size() << std::endl;
            std::cout << "Length of sources_data.fluxErrs: " << sources_data.fluxErrs.size() << std::endl;
            std::cout << "objnum_periodograms: " << objnum_periodograms << std::endl;
            std::cout << "N_freqs_current: " << N_freqs_current << std::endl;
            std::cout << "N_src_per_batch: " << field_data.N_src_per_batch << std::endl;
            std::cout << "N_samples: " << params.N_samples << std::endl;

            //Compute the periodograms for the current chunk
            float* device_periodogram_chunk = runFPWChunk(device_indices_chunk,
                                                          sources_data.fluxes_no_baseline.data(),
                                                          sources_data.fluxErrs.data(),
                                                          field_data.N_exp,
                                                          objnum_periodograms,
                                                          N_freqs_current,
                                                          params.N_bins.data(),
                                                          params.N_samples);
            cudaDeviceSynchronize();

            std::vector<float> chunk_periodogram(N_freqs_current * params.N_samples, 0.0f);
            cudaMemcpy(
                chunk_periodogram.data(),
                device_periodogram_chunk + last_src_local * N_freqs_current * params.N_samples,
                N_freqs_current * params.N_samples * sizeof(float),
                cudaMemcpyDeviceToHost
            );

            // Place into the full periodogram array
            for (unsigned long f = 0; f < N_freqs_current; ++f) {
                periodogram_full[freq_offset + f] = chunk_periodogram[f];
            }
            freq_offset += N_freqs_current;

            //Delete the device indices
            cudaFree(device_indices_chunk);

            // If this is the first chunk, compute the median and MAD
            if (i == 0) {
                CudaComputeMedianandMAD<<<gridSizePostprocess, blockSizePostprocess>>>(device_periodogram_chunk, device_medians, device_mads, N_freqs_current, objnum_periodograms);
            }

            // Find peaks in this chunk
            cudaFindPeaksChunk<<<gridSizePostprocess, blockSizePostprocess>>>(device_periodogram_chunk,
                                                             device_medians,
                                                             device_mads,
                                                             d_peak_indices,
                                                             d_peak_values,
                                                             N_freqs_current,
                                                             objnum_periodograms * params.N_samples,
                                                             params.peak_distance,
                                                             params.peak_threshold,
                                                             d_state_candidate_idx,
                                                             d_state_candidate_val,
                                                             d_state_distLastPeak,
                                                             d_state_currentThreshold,
                                                             i,
                                                             i + N_freqs_current
                                                            );

            // Clean up device memory
            cudaFree(device_periodogram_chunk);
        }

        // Finish timing the frequency loop
        auto stop_freqloop = chrono::high_resolution_clock::now();
        auto duration_freqloop = chrono::duration_cast<chrono::milliseconds>(stop_freqloop - start_freqloop);
        logMessage(params.log_path, "Time for frequency loop: " + to_string(duration_freqloop.count()) + " milliseconds");

        std::ofstream pgram_out("periodogram_last_source.csv");
        for (unsigned long f = 0; f < field_data.N_freqs; ++f) {
            pgram_out << field_data.fgrid[f] << periodogram_full[f] << "\n";
        }
        pgram_out.close();

        std::ofstream lc_out("lightcurve_last_source.csv");
        for (unsigned long e = 0; e < field_data.N_exp; ++e) {
            lc_out << field_data.times[e] << ", " << lightcurve[e] << ", " << lightcurve_no_baseline[e] << ", " << lightcurve_errors[e] <<  "\n";
        }
        lc_out.close();

        // Copy the peak indices and values back to the host
        vector<int> peak_indices(objnum_periodograms * params.N_samples * N_peaks, -1);
        vector<float> peak_values(objnum_periodograms * params.N_samples * N_peaks, -INFINITY);
        vector<float> periodogram_medians(objnum_periodograms * params.N_samples, 0.0f);
        vector<float> periodogram_mads(objnum_periodograms * params.N_samples, 0.0f);

        cudaMemcpy(peak_indices.data(), d_peak_indices, objnum_periodograms * params.N_samples * N_peaks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(peak_values.data(), d_peak_values, objnum_periodograms * params.N_samples * N_peaks * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(periodogram_medians.data(), device_medians, objnum_periodograms * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(periodogram_mads.data(), device_mads, objnum_periodograms * sizeof(float), cudaMemcpyDeviceToHost);

        // Find the peak frequencies from the indices
        vector<float> peak_freqs(objnum_periodograms * params.N_samples * N_peaks, -1.0f);
        for (unsigned long j = 0; j < objnum_periodograms * params.N_samples; j++) {
            for (unsigned long k = 0; k < N_peaks; k++) {
                int peakIndex = peak_indices[j * N_peaks + k];
                if (peakIndex >= 0) {
                    peak_freqs[j * N_peaks + k] = field_data.fgrid[peakIndex];
                } else {
                    peak_freqs[j * N_peaks + k] = -1.0f; // Set to -1 if no peak found
                }
            }
        }

        // // Write results to file
        // appendResults(results_filename,
        //               sources_data.psids.data(),
        //               sources_data.ras.data(), sources_data.decs.data(),
        //               sources_data.ratioValid.data(),
        //               peak_freqs.data(),
        //               peak_values.data(),
        //               periodogram_medians.data(),
        //               periodogram_mads.data(),
        //               objnum_periodograms,
        //               params.N_samples,
        //               N_peaks,
        //               objstart_periodograms);

        // Clean up device memory
        cudaFree(d_state_candidate_idx);
        cudaFree(d_state_candidate_val);
        cudaFree(d_state_distLastPeak);
        cudaFree(d_state_currentThreshold);
        cudaFree(d_peak_indices);
        cudaFree(d_peak_values);
        cudaFree(device_medians);
        cudaFree(device_mads);

    }

    auto stop_full = chrono::high_resolution_clock::now();
    auto duration_full = chrono::duration_cast<chrono::seconds>(stop_full - start_full);
    logMessage(params.log_path, "Time for full run: " + to_string(duration_full.count()) + " seconds");
    cout << "matchfile " << params.input_hdf_filename << " finished in " << duration_full.count() << " seconds" << endl;

    // rename the results file to remove the .temp extension
    // string final_results_filename = results_filename;
    // // remove the last 5 characters (.temp) from the filename
    // final_results_filename = final_results_filename.substr(0, final_results_filename.size() - 5);
    // string rename_command = "mv " + results_filename + " " + final_results_filename;
    // int rename_status = system(rename_command.c_str());
    // if (rename_status != 0) {
    //     logMessage(params.log_path, "Error renaming results file: " + std::to_string(rename_status));
    // } else {
    //     logMessage(params.log_path, "Results written to: " + final_results_filename);
    // }
}