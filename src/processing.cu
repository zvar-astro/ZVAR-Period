#include "processing.cuh"

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Loads and prepares field data from the HDF5 file
 * 
 * This function loads the field data from the HDF5 file and prepares it for processing.
 * 
 * @param params The processing parameters
 * @return FieldData containing the loaded data
 */
FieldData loadAndPrepareFieldData(const ProcessingExternalParams& params) {
    FieldData field_data;
    logMessage(params.log_path, params.input_hdf_filename, "Loading frame data from: " + params.input_hdf_filename);
    try {
        std::tie(field_data.N_exp, field_data.N_src, field_data.times_double, field_data.times_float, field_data.dd_indices) =
            initializeFrameData(params.input_hdf_filename,
                                params.id_field_name.c_str(),
                                params.time_field_name.c_str(),
                                params.deep_drilling);
        field_data.success = true;
        logMessage(params.log_path, params.input_hdf_filename, "Frame data loaded. N_exp: " + std::to_string(field_data.N_exp) + ", N_src: " + std::to_string(field_data.N_src));
    } catch (const std::exception& e) {
        logMessage(params.log_path, params.input_hdf_filename, "Error loading frame data: " + std::string(e.what()));
        field_data.success = false;
    }
    return field_data;
}

/**
 * @brief Creates a frequency grid and filters out terrestrial frequencies
 * 
 * This function creates a frequency grid based on the provided parameters and filters out terrestrial frequencies.
 * 
 * @param params The processing parameters
 * @param data The field data to add the frequency grid to
 * @return true if the frequency grid was created and filtered successfully, false otherwise
 */
void createAndFilterFrequencyGrid(
    const ProcessingExternalParams& params,
    FieldData& field_data) { // Takes FieldData by reference

    logMessage(params.log_path, params.input_hdf_filename, "Creating and filtering frequency grid...");
    try {
        // std::vector<double> times_double(field_data.times_float.begin(), field_data.times_float.end());
        std::vector<double> fgrid_double = freq_grid(field_data.times_double, params.f_min, params.f_max, params.freq_oversampling);
        std::vector<float> fgrid(fgrid_double.begin(), fgrid_double.end());
        unsigned long N_freqs_initial = fgrid.size();

        if (N_freqs_initial == 0) {
            logMessage(params.log_path, params.input_hdf_filename, "Frequency grid is empty after creation.");
            field_data.success = false; // Grid creation resulted in empty, likely a failure
            return;
        }

        // Erase terrestrial frequencies from vector
        auto start_erase_freq = chrono::high_resolution_clock::now();
        std::vector<int> freq_flags = flagTerrestrialFreq(fgrid.data(), N_freqs_initial); // Get indices to exclude terrestrial frequencies

        // Remove terrestrial frequencies
        auto badElements = std::remove_if(fgrid.begin(), fgrid.end(), [&](float freq) {
            static int index = 0;
            return freq_flags[index++] == 0;
        });
        fgrid.erase(badElements, fgrid.end());

        unsigned long N_freqs = fgrid.size(); // Update the NFREQ value

        field_data.fgrid = fgrid; // Store the filtered frequency grid 
        field_data.N_freqs = N_freqs; // Save the number of frequencies

        auto duration_erase_freq = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_erase_freq);
        logMessage(params.log_path, params.input_hdf_filename, "Time to filter terrestrial frequencies: " + std::to_string(duration_erase_freq.count()) + " ms");
        logMessage(params.log_path, params.input_hdf_filename, "Filtered frequency grid size: " + std::to_string(field_data.fgrid.size()));
    } catch (const std::exception& e) {
        logMessage(params.log_path, params.input_hdf_filename, "Error creating or filtering frequency grid: " + std::string(e.what()));
        field_data.success = false; // Grid creation resulted in empty, likely a failure
        return;
    }
}

/**
 * @brief Initializes chunk parameters for processing
 * 
 * This function calculates the chunk parameters based on the provided device and field data.
 * 
 * @param params The processing parameters
 * @param field_data The field data to initialize chunk parameters for
 */
void initializeChunkParameters(const ProcessingExternalParams& params, FieldData& field_data) {
    try {
        std::pair<size_t, size_t> gpu_batches = calculateGPUBatches(params.device, field_data.N_exp, params.N_samples, field_data.N_src, field_data.N_freqs, N_peaks, params.percent_indices, 32, 8);

        unsigned long N_freqs_per_batch = static_cast<unsigned long>(gpu_batches.first);
        unsigned long N_src_per_batch = static_cast<unsigned long>(gpu_batches.second);

        unsigned long N_freqs_batch = (field_data.N_freqs + N_freqs_per_batch - 1) / N_freqs_per_batch;
        unsigned long N_src_batch = (field_data.N_src + N_src_per_batch - 1) / N_src_per_batch;

        unsigned long N_chunk_baseline = calculateNChunkBaseline(params.device, N_src_per_batch, field_data.N_exp, params.model_id);
        unsigned long N_src_per_chunk_baseline = 1 + ((N_src_per_batch-1) / N_chunk_baseline);

        field_data.N_freqs_batch = N_freqs_batch;
        field_data.N_src_batch = N_src_batch;
        field_data.N_freqs_per_batch = N_freqs_per_batch;
        field_data.N_src_per_batch = N_src_per_batch;

        field_data.N_src_batch_gpu_baseline = N_src_per_chunk_baseline;
        field_data.N_batch_gpu_baseline = N_chunk_baseline;

        // std::cout << "Number of frequency batches: " << field_data.N_freqs_batch << std::endl;
        // std::cout << "Number of source batches: " << field_data.N_src_batch << std::endl;
    } catch (const std::exception& e) {
        logMessage(params.log_path, params.input_hdf_filename, "Error initializing chunk parameters: " + std::string(e.what()));
        field_data.success = false;
        return;
    }
}

/**
 * @brief Loads chunk data from the HDF5 file
 * 
 * This function loads the flux and flux error values from the HDF5 file for a specific chunk of data.
 * 
 * @param params The processing parameters
 * @param field_data The field data to load into
 * @param sources_data The sources data to load into
 * @param objstart_periodograms The starting index for the periodograms
 * @param objnum_periodograms The number of periodograms to load
 */
void loadChunkData(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data, unsigned long objstart_periodograms, unsigned long objnum_periodograms) {

    auto& dd_indices = field_data.dd_indices; // C++ requires we initialize dd_indices as an lvalue for some god damn reason
    // Load the flux and fluxerr values
    int status = get_sourcedata_chunk(
        params.input_hdf_filename, sources_data.fluxes.data(), sources_data.fluxErrs.data(), sources_data.ratioValid.data(),
        dd_indices, objstart_periodograms, objnum_periodograms, field_data.N_exp,
        params.flux_field_name.c_str(), params.fluxErr_field_name.c_str()
    );
    check_status(status);

    // Load the ids, ra, dec values
    status = get_sources_chunk(
        params.input_hdf_filename, sources_data.psids.data(), sources_data.ras.data(), sources_data.decs.data(),
        objstart_periodograms, objnum_periodograms,
        params.id_field_name.c_str(), params.ra_field_name.c_str(), params.dec_field_name.c_str()
    );
    check_status(status);

    // Compute and fill in ivars
    for (unsigned long i = 0; i< objnum_periodograms * field_data.N_exp; i++) {
        sources_data.flux_ivars[i] = 1.0f / (sources_data.fluxErrs[i] * sources_data.fluxErrs[i]);
    }
}


void removeOutliers(FieldData& field_data, SourcesData& sources_data, DeviceArrays& device_arrays, KernelLaunchParams& kernel_params, unsigned long objnum_periodograms) {

    cudaError_t cudaStatus;
    // Find the medians and MADS of the fluxes
    // Use the device arrays to compute the medians and MADS
    cudaMemcpy(device_arrays.y_device, sources_data.fluxes.data(), objnum_periodograms * field_data.N_exp * sizeof(float), cudaMemcpyHostToDevice);

    CudaComputeMedianandMAD<<<kernel_params.grid_size_preprocess,
                            kernel_params.block_size_preprocess>>>(device_arrays.y_device,
                                                                    device_arrays.y_medians_device,
                                                                    device_arrays.y_mads_device,
                                                                    field_data.N_exp,
                                                                    static_cast<int>(objnum_periodograms));

    cudaDeviceSynchronize(); // Ensure all operations are complete before proceeding

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error in CudaComputeMedianandMAD: " << cudaGetErrorString(cudaStatus) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy the medians and MADS back to the host
    cudaMemcpy(sources_data.fluxes_medians.data(), device_arrays.y_medians_device, objnum_periodograms * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sources_data.fluxes_mads.data(), device_arrays.y_mads_device, objnum_periodograms * sizeof(float), cudaMemcpyDeviceToHost);

    // Median subtract the data and remove outliers based on the medians and MADS
    for (unsigned long i = 0; i < objnum_periodograms; i++) {
        float median = sources_data.fluxes_medians[i];
        float mad = sources_data.fluxes_mads[i];

        // print out the median and MAD for debugging
        float std = 1.483 * mad; // MAD to STD conversion factor
        for (unsigned long j = 0; j < field_data.N_exp; j++) {
            float flux = sources_data.fluxes[i * field_data.N_exp + j] - median; // Subtract the median from the flux

            // If the median subtracted flux is more than 5 std ABOVE from the median, set it to 0, error to infinity, and ivar to 0
            if (flux > 5 * std) {
                sources_data.fluxes[i * field_data.N_exp + j] = 0.0f; // Set flux to 0
                sources_data.fluxErrs[i * field_data.N_exp + j] = INFINITY; // Set error to infinity
                sources_data.flux_ivars[i * field_data.N_exp + j] = 0.0f; // Set ivar to 0
            }
            // If the median subtracted flux is more than 10 std BELOW the median, set it to 0, error to infinity, and ivar to 0
            if (flux < -10 * std) {
                sources_data.fluxes[i * field_data.N_exp + j] = 0.0f; // Set flux to 0
                sources_data.fluxErrs[i * field_data.N_exp + j] = INFINITY; // Set error to infinity
                sources_data.flux_ivars[i * field_data.N_exp + j] = 0.0f; // Set ivar to 0
            } 
        }
    }
    // Reset y_device since we use it again later.
    size_t N_exp_total = objnum_periodograms * field_data.N_exp;
    size_t threads = 256;
    size_t blocks = (N_exp_total + threads - 1) / threads;
    setFloatArrayKernel<<<blocks, threads>>>(device_arrays.y_device, 0.0f, N_exp_total);
    cudaDeviceSynchronize(); // Ensure all operations are complete before proceeding
}


/**
 * @brief Processes baseline removal for the sources data
 * 
 * This function processes the baseline removal for the sources data.
 * 
 * @param params The processing parameters
 * @param field_data The field data to process
 * @param sources_data The sources data to process
 * @param objstart_periodograms The starting index for the periodograms
 * @param objnum_periodograms The number of periodograms to process
 */
void processBaselineRemoval(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data, unsigned long objstart_periodograms, unsigned long objnum_periodograms) {
    try {
        auto result = removeBaseline(
            field_data.times_float.data(),
            sources_data.fluxes.data(),
            sources_data.fluxErrs.data(),
            field_data.N_exp,
            objnum_periodograms,
            params.model_id);

        float* temp_chunk = result.first; // Get the pointer to the fluxes with baseline removed
        float fraction_failed = result.second; // Get the fraction of failures

        // Copy directly into the correct place in sources_data.fluxes_no_baseline
        std::copy(
            temp_chunk,
            temp_chunk + (objnum_periodograms * field_data.N_exp),
            sources_data.fluxes_no_baseline.begin()
        );
        delete[] temp_chunk; // Free the temporary pointer returned by removeBaseline

        logMessage(params.log_path, params.input_hdf_filename, "Baseline removal completed for chunk starting at index " + std::to_string(objstart_periodograms) +
            ". Fraction of curves failed: " + std::to_string(fraction_failed));

        field_data.success = true; // Set success to true if baseline removal was successful
    } catch (const std::exception& e) {
        logMessage(params.log_path, params.input_hdf_filename, "Error in baseline removal: " + std::string(e.what()));
        field_data.success = false;
        return;
    }
    sources_data.success = true; // Set success to true if baseline removal was successful
}


/**
 * @brief copies the photometry data to the device
 * 
 * This function copies the photometry data for each chunk from the host to the device.
 * 
 * @param field_data The field data to copy
 * @param sources_data The sources data to copy from
 * @param device_arrays The device arrays to copy to
 * @param objnum_periodograms The number of periodograms to copy
 */
void copyPhotToDevice(FieldData& field_data, SourcesData& sources_data, DeviceArrays& device_arrays, unsigned long objnum_periodograms) {
    cudaMemcpy(device_arrays.y_device, sources_data.fluxes_no_baseline.data(), objnum_periodograms * field_data.N_exp * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_arrays.ivar_device, sources_data.flux_ivars.data(), objnum_periodograms * field_data.N_exp * sizeof(float), cudaMemcpyHostToDevice);
}

void copyLastChunkData(PeakPersistentArrays& peak_arrays, DeviceArrays& device_arrays, int num_periodogram_curves, unsigned long N_freqs_current) {
    cudaError_t cudaStatus;
    // Copy the last two periodogram values from the last chunk to the persistent arrays
    for (int curve_idx = 0; curve_idx < num_periodogram_curves; ++curve_idx) {
        float* src_base_ptr = device_arrays.periodograms_device + (static_cast<size_t>(curve_idx) * N_freqs_current);
        float* dest_base_ptr = peak_arrays.last_chunk_data_device + (static_cast<size_t>(curve_idx) * 2);

        cudaMemcpy(dest_base_ptr, src_base_ptr + (N_freqs_current - 2), sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dest_base_ptr + 1, src_base_ptr + (N_freqs_current - 1), sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

/**
 * @brief Copies the peak data back to the host and finds the peak frequencies
 * 
 * This function copies the peak indices and values back to the host and finds the peak frequencies from the indices.
 * 
 * @param params The processing parameters
 * @param field_data The field data to copy from
 * @param sources_data The sources data to copy to
 * @param device_arrays The device arrays to copy from
 * @param objnum_periodograms The number of periodograms to copy
 */
void copyPeaksToHost(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data, DeviceArrays& device_arrays, unsigned long objnum_periodograms) {
    // Copy the peak indices and values back to the host

    cudaMemcpy(sources_data.peak_indices.data(), device_arrays.peak_indices_device, objnum_periodograms * params.N_samples * N_peaks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sources_data.peak_values.data(), device_arrays.peak_values_device, objnum_periodograms * params.N_samples * N_peaks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sources_data.periodogram_means.data(), device_arrays.periodogram_means_device, objnum_periodograms * params.N_samples * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sources_data.periodogram_stds.data(), device_arrays.periodogram_stds_device, objnum_periodograms * params.N_samples * sizeof(float), cudaMemcpyDeviceToHost);

    // Find the peak frequencies from the indices
    for (unsigned long j = 0; j < objnum_periodograms * params.N_samples; j++) {
        for (unsigned long k = 0; k < N_peaks; k++) {
            int peakIndex = sources_data.peak_indices[j * N_peaks + k];
            if (peakIndex >= 0) {
                sources_data.peak_freqs[j * N_peaks + k] = field_data.fgrid[peakIndex];
            } else {
                sources_data.peak_freqs[j * N_peaks + k] = -1.0f; // Set to -1 if no peak found
            }
        }
    }
}

/**
 * @brief Resets the state arrays for peak finding
 * 
 * This function resets the persistent arrays used for peak finding to their initial state for the next source chunk.
 * 
 * @param peak_arrays The peak persistent arrays to reset
 * @param params The processing parameters
 * @param field_data The field data to use for resetting
 */
void resetPersitentArrays(DeviceArrays& device_arrays, PeakPersistentArrays& peak_arrays, const ProcessingExternalParams& params, FieldData& field_data) {

    size_t threads = 256;

    // y_device
    size_t N_y = field_data.N_src_per_batch * field_data.N_exp;
    size_t blocks_y = (N_y + threads - 1) / threads;
    setFloatArrayKernel<<<blocks_y, threads>>>(device_arrays.y_device, 0.0f, N_y);

    // ivar_device
    setFloatArrayKernel<<<blocks_y, threads>>>(device_arrays.ivar_device, 0.0f, N_y);

    // y_medians_device
    size_t N_medians = field_data.N_src_per_batch;
    size_t blocks_medians = (N_medians + threads - 1) / threads;
    setFloatArrayKernel<<<blocks_medians, threads>>>(device_arrays.y_medians_device, 0.0f, N_medians);

    // y_mads_device
    setFloatArrayKernel<<<blocks_medians, threads>>>(device_arrays.y_mads_device, 0.0f, N_medians);

    // periodograms_device
    size_t N_periodograms = field_data.N_src_per_batch * params.N_samples * field_data.N_freqs_per_batch;
    size_t blocks_periodograms = (N_periodograms + threads - 1) / threads;
    setFloatArrayKernel<<<blocks_periodograms, threads>>>(device_arrays.periodograms_device, 0.0f, N_periodograms);

    // periodogram_means_device
    size_t N_periodogram_means = field_data.N_src_per_batch * params.N_samples;
    size_t blocks_periodogram_means = (N_periodogram_means + threads - 1) / threads;
    setFloatArrayKernel<<<blocks_periodogram_means, threads>>>(device_arrays.periodogram_means_device, 0.0f, N_periodogram_means);

    // periodogram_stds_device
    setFloatArrayKernel<<<blocks_periodogram_means, threads>>>(device_arrays.periodogram_stds_device, 0.0f, N_periodogram_means);

    // peak_indices_device
    size_t N_peak_indices = field_data.N_src_per_batch * params.N_samples * N_peaks;
    size_t blocks_peak_indices = (N_peak_indices + threads - 1) / threads;
    setIntArrayKernel<<<blocks_peak_indices, threads>>>(device_arrays.peak_indices_device, -1, N_peak_indices);

    // peak_values_device
    setFloatArrayKernel<<<blocks_peak_indices, threads>>>(device_arrays.peak_values_device, -INFINITY, N_peak_indices);

    cudaDeviceSynchronize(); // Ensure all memory operations are complete before proceeding


    // Reset the persistent arrays for peak finding
    size_t N_persist = field_data.N_src_per_batch * params.N_samples;
    size_t blocks_persist = (N_persist + threads - 1) / threads;

    setIntArrayKernel<<<blocks_persist, threads>>>(peak_arrays.state_candidate_idx, -1, N_persist);
    setFloatArrayKernel<<<blocks_persist, threads>>>(peak_arrays.state_candidate_vals, -INFINITY, N_persist);
    setIntArrayKernel<<<blocks_persist, threads>>>(peak_arrays.state_dist_last_peak, params.peak_distance, N_persist);
    setFloatArrayKernel<<<blocks_persist, threads>>>(peak_arrays.state_current_threshold, params.peak_threshold, N_persist);
    setFloatArrayKernel<<<blocks_persist, threads>>>(peak_arrays.last_chunk_data_device, 0.0f, N_persist * 2); // Initialize last chunk data to -1.0f
}


/**
 * @brief Removes the temporary status from the results file
 * 
 * This function renames the results file to remove the .temp extension.
 * 
 * @param params The processing parameters
 * @param results_filename The name of the results file
 */
void removeTempStatus(const ProcessingExternalParams& params, string results_filename) {
    // rename the results file to remove the .temp extension
    string final_results_filename = results_filename;
    // remove the last 5 characters (.temp) from the filename
    final_results_filename = final_results_filename.substr(0, final_results_filename.size() - 5);
    string rename_command = "mv " + results_filename + " " + final_results_filename;
    int rename_status = system(rename_command.c_str());
    if (rename_status != 0) {
        logMessage(params.log_path, params.input_hdf_filename, "Error renaming results file: " + std::to_string(rename_status));
    } else {
        logMessage(params.log_path, params.input_hdf_filename, "Results written to: " + final_results_filename);
    }
}


/**
 * @brief Runs a single periodogram algorithm on a single matchfile and writes the results to a file
 * 
 * Computes periodograms for every light curve in a matchfile. Finds the most significant independent peaks. Writes the results to an HDF5 file.
 * 
 * @param input_hdf_filename The path to the matchfile
 * @param logpath The path to the log file
 * 
 */
void process(const ProcessingExternalParams& params) {

    auto start_full = chrono::high_resolution_clock::now(); //Time full run
    auto start_preprocess = chrono::high_resolution_clock::now(); //Time loading and pre-processing

    logMessage(params.log_path, params.input_hdf_filename, "Running on file: " + params.input_hdf_filename);

    // --- Initialize Frame Data ---
    FieldData field_data = loadAndPrepareFieldData(params);

    if (!field_data.success) {
        logMessage(params.log_path, params.input_hdf_filename, "Failed to load field data. Exiting.");
        exit(EXIT_FAILURE);
    }

    // If there are less than 50 exposures, skip the file
    if (field_data.N_exp < 50) {
        logMessage(params.log_path, params.input_hdf_filename, "Less than 50 exposures found in file, skipping");
        return;
    }

    // If there are no sources, skip the file
    if (field_data.N_src <= 0) {
        logMessage(params.log_path, params.input_hdf_filename, "No sources found in file, skipping");
        return;
    }

    // --- Create Frequency Grid ---
    createAndFilterFrequencyGrid(params, field_data); //loads frequency grid into field_data

    if(!field_data.success) {
        logMessage(params.log_path, params.input_hdf_filename, "Failed to create frequency grid. Exiting.");
        exit(EXIT_FAILURE);
    }

    // --- Initialize Chunking Parameters ---
    initializeChunkParameters(params, field_data); //loads chunk parameters into field_data

    if (!field_data.success) {
        logMessage(params.log_path, params.input_hdf_filename, "Failed to initialize chunk parameters. Exiting.");
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

    // --- Create Output File ---
    std::string results_filename = createOutputFilepath(params.output_dir, params.input_hdf_filename);
    createResults(results_filename, field_data.N_src, params.N_samples, N_peaks);

    auto stop_preprocess = chrono::high_resolution_clock::now();
    auto duration_preprocess = chrono::duration_cast<chrono::milliseconds>(stop_preprocess - start_preprocess);
    logMessage(params.log_path, params.input_hdf_filename, "Time to load and pre-process: " + to_string(duration_preprocess.count()) + " milliseconds");

    // print how many sources we have in total, how many chunks we will run, how many sources per chunk
    // and how many exposures per source
    logMessage(params.log_path, params.input_hdf_filename, 
        "Total sources: " + to_string(field_data.N_src) + 
        ", Total source chunks: " + to_string(field_data.N_src_batch) + 
        ", Sources per chunk: " + to_string(field_data.N_src_per_batch) + 
        ", Exposures per source: " + to_string(field_data.N_exp) + 
        ", Total frequency chunks: " + to_string(field_data.N_freqs_batch) +
        ", Frequencies per chunk: " + to_string(field_data.N_freqs_per_batch));

    for (unsigned long k = 0; k < field_data.N_src_batch; k++) {
        auto start_loadchunk = chrono::high_resolution_clock::now(); // Time loading chunk

        // Get the range of objects to analyze
        objstart_periodograms = k * field_data.N_src_per_batch;
        unsigned long objnum_periodograms = (k < field_data.N_src_batch - 1)
            ? field_data.N_src_per_batch
            : (field_data.N_src - objstart_periodograms);
        int N_periodogram_curves = static_cast<int>(objnum_periodograms) * params.N_samples;

        // Load the flux and fluxerr values
        loadChunkData(params, field_data, sources_data, objstart_periodograms, objnum_periodograms);

        auto stop_loadchunk = chrono::high_resolution_clock::now();
        auto duration_loadchunk = chrono::duration_cast<chrono::milliseconds>(stop_loadchunk - start_loadchunk);
        logMessage(params.log_path, params.input_hdf_filename, "Time to load chunk: " + to_string(duration_loadchunk.count()) + " milliseconds");

        auto start_removeOutliers = chrono::high_resolution_clock::now(); // Time removing outliers

        //Remove outliers from the fluxes, also median subtracts the fluxes
        removeOutliers(field_data, sources_data, device_arrays, kernel_params, objnum_periodograms);

        auto stop_removeOutliers = chrono::high_resolution_clock::now();
        auto duration_removeOutliers = chrono::duration_cast<chrono::milliseconds>(stop_removeOutliers - start_removeOutliers);
        logMessage(params.log_path, params.input_hdf_filename, "Time to remove outliers: " + to_string(duration_removeOutliers.count()) + " milliseconds");

        auto start_removebaseline = chrono::high_resolution_clock::now(); // Time removing baseline

        if (params.remove_baseline) {
            processBaselineRemoval(params, field_data, sources_data, objstart_periodograms, objnum_periodograms);
        }
        else {
            // If not removing baseline, just copy the fluxes to the no baseline array
            sources_data.fluxes_no_baseline = sources_data.fluxes;
            sources_data.success = true;
            logMessage(params.log_path, params.input_hdf_filename, "Skipping baseline removal as per user request.");
        }

        if (!sources_data.success) {
            logMessage(params.log_path, params.input_hdf_filename, "Failed to remove baseline. Exiting.");
            exit(EXIT_FAILURE);
        }

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error cudaDeviceSynchronize: " << cudaGetErrorString(cudaStatus) << std::endl;
            exit(EXIT_FAILURE);
        }

        auto stop_removebaseline = chrono::high_resolution_clock::now();
        auto duration_removebaseline = chrono::duration_cast<chrono::milliseconds>(stop_removebaseline - start_removebaseline);
        logMessage(params.log_path, params.input_hdf_filename, "Time to remove baseline: " + to_string(duration_removebaseline.count()) + " milliseconds");

        // Copy host fluxes and flux errors to device
        copyPhotToDevice(field_data, sources_data, device_arrays, objnum_periodograms);
        cudaDeviceSynchronize(); // Ensure all operations are complete before proceeding

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error in copyPhotToDevice: " << cudaGetErrorString(cudaStatus) << std::endl;
            exit(EXIT_FAILURE);
        }

        auto start_processing = chrono::high_resolution_clock::now(); // Time processing periodograms

        for (unsigned long i = 0; i < field_data.N_freqs; i += field_data.N_freqs_per_batch) {
            unsigned long N_freqs_current = std::min(field_data.N_freqs_per_batch, field_data.N_freqs - i);

            cudaMemcpy(device_arrays.freqs_device, field_data.fgrid.data() + i, N_freqs_current * sizeof(float), cudaMemcpyHostToDevice);

            auto start_FPW = std::chrono::high_resolution_clock::now();
            // Compute indices for the current chunk
            cudaMakeIndices<<<kernel_params.grid_size_indices,
                              kernel_params.block_size_indices>>>(device_arrays.t_device,
                                                                  device_arrays.freqs_device,
                                                                  device_arrays.indices_device,
                                                                  params.N_bins[0],
                                                                  field_data.N_exp,
                                                                  static_cast<int>(N_freqs_current));    
            cudaDeviceSynchronize();

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                std::cerr << "CUDA error in cudaMakeIndices: " << cudaGetErrorString(cudaStatus) << std::endl;
                exit(EXIT_FAILURE);
            }

            //Compute the periodograms for the current chunk
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

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                std::cerr << "CUDA error in cudaDeltaChi2Multibin: " << cudaGetErrorString(cudaStatus) << std::endl;
                exit(EXIT_FAILURE);
            }

            auto stop_FPW = std::chrono::high_resolution_clock::now();
            auto duration_FPW = std::chrono::duration_cast<std::chrono::milliseconds>(stop_FPW - start_FPW);
            // std::cout << "Time taken for cudaDeltaChi2Multibin: " << duration_FPW.count() << " ms" << std::endl;

            // If this is the first chunk, compute the mean and STD of the periodograms
            if (i == 0) {
                auto start_means = std::chrono::high_resolution_clock::now();

                // If clip_pgram is set, use the clipping version
                if (params.clip_pgram) {
                    float N_sigma = 3.0;
                    CudaComputeMeanandSTDClip<<<kernel_params.grid_size_postprocess,
                                          kernel_params.block_size_postprocess>>>(device_arrays.periodograms_device,
                                                                            device_arrays.periodogram_means_device,
                                                                            device_arrays.periodogram_stds_device,
                                                                            static_cast<int>(N_freqs_current),
                                                                            N_periodogram_curves,
                                                                            N_sigma);
                } else {
                CudaComputeMeanandSTD<<<kernel_params.grid_size_postprocess,
                                          kernel_params.block_size_postprocess>>>(device_arrays.periodograms_device,
                                                                            device_arrays.periodogram_means_device,
                                                                            device_arrays.periodogram_stds_device,
                                                                            static_cast<int>(N_freqs_current),
                                                                            N_periodogram_curves);
                }
                cudaDeviceSynchronize();

                auto stop_means = std::chrono::high_resolution_clock::now();
                auto duration_means = std::chrono::duration_cast<std::chrono::milliseconds>(stop_means - start_means);
                logMessage(params.log_path, params.input_hdf_filename, "Time to compute means and STDs: " + to_string(duration_means.count()) + " milliseconds");

                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    std::cerr << "CUDA error in CudaComputeMeanandSTD: " << cudaGetErrorString(cudaStatus) << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            // Normalize the periodograms by the mean and STD
            cudaComputeSignificance<<<kernel_params.grid_size_postprocess,
                                          kernel_params.block_size_postprocess>>>(device_arrays.periodograms_device,
                                                                                  device_arrays.periodogram_means_device,
                                                                                  device_arrays.periodogram_stds_device,
                                                                                  static_cast<int>(N_freqs_current),
                                                                                  N_periodogram_curves);
            cudaDeviceSynchronize();

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                std::cerr << "CUDA error in cudaComputeSignificance: " << cudaGetErrorString(cudaStatus) << std::endl;
                exit(EXIT_FAILURE);
            }

            //Check the peak_arrays.state_current_threshold and print the first value
            float first_threshold;
            cudaMemcpy(&first_threshold, peak_arrays.state_current_threshold, sizeof(float), cudaMemcpyDeviceToHost);
            // Find peaks in this chunk
            cudaFindPeaksChunk<<<kernel_params.grid_size_postprocess,
                        kernel_params.block_size_postprocess>>>(device_arrays.periodograms_device,
                                                    peak_arrays.last_chunk_data_device,
                                                    device_arrays.peak_indices_device,
                                                    device_arrays.peak_values_device,
                                                    static_cast<int>(N_freqs_current),
                                                    N_periodogram_curves,
                                                    params.peak_distance,
                                                    params.peak_threshold,
                                                    static_cast<int>(i),
                                                    peak_arrays.state_candidate_idx,
                                                    peak_arrays.state_candidate_vals,
                                                    peak_arrays.state_dist_last_peak,
                                                    peak_arrays.state_current_threshold);
            cudaDeviceSynchronize();

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                std::cerr << "CUDA error in cudaFindPeaksChunk: " << cudaGetErrorString(cudaStatus) << std::endl;
                exit(EXIT_FAILURE);
            }

            // Update the last chunk data for the next iteration
            // This will store the last two periodogram values from the current frequency chunk (N_freqs_current)
            // to be used as context by cudaFindPeaksChunk in the next frequency chunk.
            copyLastChunkData(peak_arrays, device_arrays, N_periodogram_curves, N_freqs_current);
            cudaDeviceSynchronize();

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                std::cerr << "CUDA error in copyLastChunkData: " << cudaGetErrorString(cudaStatus) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // Copy the peak indices and values back to the host
        copyPeaksToHost(params, field_data, sources_data, device_arrays, objnum_periodograms);
        cudaDeviceSynchronize();

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error in copyPeaksToHost: " << cudaGetErrorString(cudaStatus) << std::endl;
            exit(EXIT_FAILURE);
        }

        auto stop_processing = chrono::high_resolution_clock::now();
        auto duration_processing = chrono::duration_cast<chrono::milliseconds>(stop_processing - start_processing);

        logMessage(params.log_path, params.input_hdf_filename, "Time to process periodograms for chunk " + to_string(k) +
            "/" + to_string(field_data.N_src_batch) +
            ": " + to_string(duration_processing.count()) + " milliseconds");

        // Write results to file
        appendResults(results_filename,
                      sources_data.psids.data(),
                      sources_data.ras.data(), sources_data.decs.data(),
                      sources_data.ratioValid.data(),
                      sources_data.peak_freqs.data(),
                      sources_data.peak_values.data(),
                      sources_data.periodogram_means.data(),
                      sources_data.periodogram_stds.data(),
                      objnum_periodograms,
                      params.N_samples,
                      N_peaks,
                      objstart_periodograms);

        // Reset the persistent arrays for the next chunk
        resetPersitentArrays(device_arrays, peak_arrays, params, field_data);
        cudaDeviceSynchronize(); // Ensure all operations are complete before proceeding

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error in resetPersitentArrays: " << cudaGetErrorString(cudaStatus) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    //Remove the .temp extension at the end
    removeTempStatus(params, results_filename);

    logMessage(params.log_path, params.input_hdf_filename, "Finished processing file: " + params.input_hdf_filename + ", saved results to: " + results_filename);

    auto stop_full = chrono::high_resolution_clock::now();
    auto duration_full = chrono::duration_cast<chrono::seconds>(stop_full - start_full);
    logMessage(params.log_path, params.input_hdf_filename, "matchfile " + params.input_hdf_filename + " finished in " + to_string(duration_full.count()) + " seconds");
}