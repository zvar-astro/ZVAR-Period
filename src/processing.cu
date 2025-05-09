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
    logMessage(params.log_path, "Loading frame data from: " + params.input_hdf_filename);
    try {
        std::tie(field_data.N_exp, field_data.N_src, field_data.times, field_data.dd_indices) =
            initializeFrameData(params.input_hdf_filename,
                                params.id_field_name.c_str(),
                                params.time_field_name.c_str());
        field_data.success = true;
        logMessage(params.log_path, "Frame data loaded. N_exp: " + std::to_string(field_data.N_exp) + ", N_src: " + std::to_string(field_data.N_src));
    } catch (const std::exception& e) {
        logMessage(params.log_path, "Error loading frame data: " + std::string(e.what()));
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

    logMessage(params.log_path, "Creating and filtering frequency grid...");
    try {
        std::vector<double> times_double(field_data.times.begin(), field_data.times.end());
        std::vector<double> fgrid_double = freq_grid(times_double, params.f_min, params.f_max, params.freq_oversampling);
        std::vector<float> fgrid(fgrid_double.begin(), fgrid_double.end());
        unsigned long N_freqs_initial = fgrid.size();

        if (N_freqs_initial == 0) {
            logMessage(params.log_path, "Frequency grid is empty after creation.");
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
        logMessage(params.log_path, "Time to filter terrestrial frequencies: " + std::to_string(duration_erase_freq.count()) + " ms");
        logMessage(params.log_path, "Filtered frequency grid size: " + std::to_string(field_data.fgrid.size()));
    } catch (const std::exception& e) {
        logMessage(params.log_path, "Error creating or filtering frequency grid: " + std::string(e.what()));
        field_data.success = false; // Grid creation resulted in empty, likely a failure
        return;
    }
}

void initializeChunkParameters(const ProcessingExternalParams& params, FieldData& field_data) {
    try {
        //Decide how many CPU chunks to run based on memory
        unsigned long N_chunk = calculateMaxCPUBatch(field_data.N_freqs, params.N_samples, field_data.N_exp, field_data.N_src);
        unsigned long N_src_per_chunk = 1 + ((field_data.N_src-1) / N_chunk);

        // Decide how many GPU chunks to run based on memory
        const char* N_freq_batch_env = std::getenv("N_FREQ_BATCH");
        unsigned long N_freq_batch = (N_freq_batch_env != nullptr) ? std::stoul(N_freq_batch_env) : 100000;

        if (N_freq_batch == 0) {
            logMessage(params.log_path, "Error: N_FREQ_BATCH cannot be zero");
            field_data.success = false;
            return;
        }

        if (N_freq_batch > field_data.N_freqs) {
            // make it match the size of the frequency grid
            logMessage(params.log_path, "Warning: N_FREQ_BATCH is larger than the size of the frequency grid, setting it to " + std::to_string(field_data.N_freqs));
            N_freq_batch = field_data.N_freqs;
        }

        unsigned long N_curves_batch = calculateMaxGPUBatch(params.device, field_data.N_exp, N_freq_batch, params.N_samples, N_src_per_chunk);

        if (N_curves_batch == 0) {
            std::cerr << "Error: N_curves_batch is zero, exiting" << std::endl;
            logMessage(params.log_path, "Error: N_curves_batch is zero, exiting");
            field_data.success = false;
            return;
        }

        unsigned long N_chunk_baseline = calculateNChunkBaseline(params.device, N_src_per_chunk, field_data.N_exp, params.model_id);
        unsigned long N_src_per_chunk_baseline = 1 + ((N_src_per_chunk-1) / N_chunk_baseline); // ?

        field_data.N_freqs_batch_gpu_periodogram = N_freq_batch;
        field_data.N_src_batch_cpu = N_src_per_chunk;
        field_data.N_src_batch_gpu_periodogram = N_curves_batch;
        field_data.N_batch_cpu = N_chunk;
        field_data.N_src_batch_gpu_baseline = N_src_per_chunk_baseline;
        field_data.N_batch_gpu_baseline = N_chunk_baseline;

        std::cout << "N_freq_batch: " << N_freq_batch << std::endl;
        std::cout << "N_curves_batch: " << N_curves_batch << std::endl;
    } catch (const std::exception& e) {
        logMessage(params.log_path, "Error initializing chunk parameters: " + std::string(e.what()));
        field_data.success = false;
        return;
    }
}

void loadChunkData(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data, unsigned long objstart, unsigned long objnum) {

    auto& dd_indices = field_data.dd_indices; // C++ requires we initialize dd_indices as an lvalue for some god damn reason
    // Load the flux and fluxerr values
    int status = get_sourcedata_chunk(
        params.input_hdf_filename, sources_data.fluxes.data(), sources_data.fluxErrs.data(), sources_data.ratioValid.data(),
        dd_indices, objstart, objnum, field_data.N_exp,
        params.flux_field_name.c_str(), params.fluxErr_field_name.c_str()
    );
    check_status(status);

    // Load the ids, ra, dec values
    status = get_sources_chunk(
        params.input_hdf_filename, sources_data.psids.data(), sources_data.ras.data(), sources_data.decs.data(),
        objstart, objnum,
        params.id_field_name.c_str(), params.ra_field_name.c_str(), params.dec_field_name.c_str()
    );
    check_status(status);
}

void processBaselineRemoval(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data, unsigned long objstart, unsigned long objnum) {

    try{
        for (unsigned long l = 0; l < field_data.N_batch_gpu_baseline; l++) {
            unsigned long objstart_baseline = l * field_data.N_src_batch_gpu_baseline;
            unsigned long objnum_baseline = (l < field_data.N_batch_gpu_baseline - 1) ? field_data.N_src_batch_gpu_baseline : (field_data.N_src_batch_cpu - objstart_baseline);
        
            // Call removeBaseline and store the result in a temporary vector
            std::vector<float> flux_no_baseline_chunk(objnum_baseline * field_data.N_exp, 0.0f);
            float* temp_chunk = removeBaseline(field_data.times.data(), 
                                            sources_data.fluxes.data() + objstart_baseline * field_data.N_exp,
                                            sources_data.fluxErrs.data() + objstart_baseline * field_data.N_exp,
                                            field_data.N_exp, objnum_baseline,
                                            params.model_id);
        
            // Copy the results from the temporary pointer to the vector
            std::copy(temp_chunk, temp_chunk + (objnum_baseline * field_data.N_exp), flux_no_baseline_chunk.begin());
            delete[] temp_chunk; // Free the temporary pointer returned by removeBaseline
        
            // Copy the chunk results into the main flux_no_baseline vector
            for (unsigned long i = 0; i < objnum_baseline; i++) {
                for (unsigned long j = 0; j < field_data.N_exp; j++) {
                    sources_data.fluxes_no_baseline[(objstart_baseline + i) * field_data.N_exp + j] = flux_no_baseline_chunk[i * field_data.N_exp + j];
                }
            }
        }
        field_data.success = true; // Set success to true if baseline removal was successful
    } catch (const std::exception& e) {
        logMessage(params.log_path, "Error in baseline removal: " + std::string(e.what()));
        field_data.success = false;
        return;
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

    // --- Initialize Devices ---
    omp_set_num_threads(params.num_omp_threads); // Set the number of threads for OpenMP

    //Obtain the semaphore for the device
    std::string semaphore_name = "/device_" + std::to_string(params.device) + "_lock";
    const char* semaphore_name_char = semaphore_name.c_str();
    sem_t* semaphore = openSemaphore(semaphore_name_char); // Open the semaphore

    auto start_full = chrono::high_resolution_clock::now(); //Time full run
    auto start_preprocess = chrono::high_resolution_clock::now(); //Time loading and pre-processing

    logMessage(params.log_path, "Running on file: " + params.input_hdf_filename);

    // --- Initialize Frame Data ---
    FieldData field_data = loadAndPrepareFieldData(params);

    if (!field_data.success) {
        logMessage(params.log_path, "Failed to load field data. Exiting.");
        closeSemaphore(semaphore, semaphore_name_char);
        exit(EXIT_FAILURE);
    }

    // If there are no exposures, skip the file
    if (field_data.N_exp == 0) {
        logMessage(params.log_path, "No exposures found in file, skipping");
        closeSemaphore(semaphore, semaphore_name_char);
        return;
    }

    // --- Create Frequency Grid ---
    createAndFilterFrequencyGrid(params, field_data); //loads frequency grid into field_data

    if(!field_data.success) {
        logMessage(params.log_path, "Failed to create frequency grid. Exiting.");
        closeSemaphore(semaphore, semaphore_name_char);
        exit(EXIT_FAILURE);
    }

    // --- Initialize Memory ---
    initializeChunkParameters(params, field_data); //loads chunk parameters into field_data

    if (!field_data.success) {
        logMessage(params.log_path, "Failed to initialize chunk parameters. Exiting.");
        closeSemaphore(semaphore, semaphore_name_char);
        exit(EXIT_FAILURE);
    }
    
    // --- Initialize Chunk Source Data ---
    int status;
    unsigned long objstart, objnum;
    SourcesData sources_data(field_data); // Initialize sources data with field data

    // ---Create Output File---
    std::string results_filename = createOutputFilepath(params.output_dir, params.input_hdf_filename);
    createResults(results_filename, field_data.N_src, params.N_samples, N_peaks);

    auto stop_preprocess = chrono::high_resolution_clock::now();
    auto duration_preprocess = chrono::duration_cast<chrono::milliseconds>(stop_preprocess - start_preprocess);
    logMessage(params.log_path, "Time to load and pre-process: " + to_string(duration_preprocess.count()) + " milliseconds");


    // print how many sources we have in total, how many chunks we will run, how many sources per chunk
    // and how many exposures per source
    logMessage(params.log_path, 
        "Total sources: " + to_string(field_data.N_src) + 
        ", Total chunks: " + to_string(field_data.N_batch_cpu) + 
        ", Sources per chunk: " + to_string(field_data.N_src_batch_cpu) + 
        ", Exposures per source: " + to_string(field_data.N_exp));

    for (unsigned long k = 0; k < field_data.N_batch_cpu; k++) {
        auto start_loadchunk = chrono::high_resolution_clock::now(); // Time loading chunk

        // Get the range of objects to analyze
        objstart = k * field_data.N_src_batch_cpu;
        unsigned long objnum = (k < field_data.N_batch_cpu - 1) ? field_data.N_src_batch_cpu : (field_data.N_src - objstart);

        // Load the flux and fluxerr values
        // TODO: pass the name of the flux and fluxerr columns as arguments (??? - Sam)
        loadChunkData(params, field_data, sources_data, objstart, objnum);

        auto stop_loadchunk = chrono::high_resolution_clock::now();
        auto duration_loadchunk = chrono::duration_cast<chrono::milliseconds>(stop_loadchunk - start_loadchunk);

        auto start_removebaseline = chrono::high_resolution_clock::now(); // Time removing baseline

        //Utilize a semaphore to prevent multiple threads from accessing the GPU at the same time
        acquireSemaphore(semaphore);

        logMessage(params.log_path, "Semaphore acquired for baseline removal in chunk: " + to_string(k));

        processBaselineRemoval(params, field_data, sources_data, objstart, objnum);

        if (!sources_data.success) {
            logMessage(params.log_path, "Failed to remove baseline. Exiting.");
            releaseSemaphore(semaphore);
            closeSemaphore(semaphore, semaphore_name_char);
            exit(EXIT_FAILURE);
        }

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error cudaDeviceSynchronize: " << cudaGetErrorString(cudaStatus) << std::endl;
            exit(EXIT_FAILURE);
        }

        releaseSemaphore(semaphore);

        auto stop_removebaseline = chrono::high_resolution_clock::now();
        auto duration_removebaseline = chrono::duration_cast<chrono::milliseconds>(stop_removebaseline - start_removebaseline);
        logMessage(params.log_path, "Time to remove baseline: " + to_string(duration_removebaseline.count()) + " milliseconds");

        // Run FPW
        auto start_periodogram = chrono::high_resolution_clock::now(); // Time running periodogram
        //Open the semaphore for the periodogram
        acquireSemaphore(semaphore);
        logMessage(params.log_path, "Semaphore acquired for periodogram computation in chunk: " + to_string(k));

        float* t_host = field_data.times.data();
        float* flux_host = sources_data.fluxes_no_baseline.data();
        float* fluxErr_host = sources_data.fluxErrs.data();
        float* fgrid_host = field_data.fgrid.data();
        int N_data = static_cast<int>(field_data.N_exp);
        unsigned long N_curves = objnum;

        // Periodograms are stored light curve contiguous and then sample contiguous. I.E. [curve1_Nbins1, curve1_Nbins2, ...]
        vector<float> periodograms = cudaRunFPW(t_host,
                                                flux_host,
                                                fluxErr_host,
                                                fgrid_host,
                                                N_data,
                                                N_curves,
                                                field_data.N_freqs,
                                                params.N_bins.data(),
                                                field_data.N_freqs_batch_gpu_periodogram,
                                                field_data.N_src_batch_gpu_periodogram,
                                                params.N_samples);

        releaseSemaphore(semaphore);
        auto stop_periodogram = chrono::high_resolution_clock::now();
        auto duration_periodogram = chrono::duration_cast<chrono::milliseconds>(stop_periodogram - start_periodogram);
        logMessage(params.log_path, "Time to run periodogram: " + to_string(duration_periodogram.count()) +
                                    " ms (per light curve: " + to_string(duration_periodogram.count() / objnum) +
                                    " ms)");

        // Find the most significant peaks
        auto start_postprocess = chrono::high_resolution_clock::now(); // Time computing post-process

        vector<float> peakFrequencies(N_curves * params.N_samples * N_peaks, 0.0f);
        vector<float> peakSignificances(N_curves * params.N_samples * N_peaks, 0.0f);
        vector<float> periodogramMeans(N_curves * params.N_samples, 0.0f);
        vector<float> periodogramStdevs(N_curves * params.N_samples, 0.0f);

        #pragma omp parallel for schedule(dynamic)
        for (unsigned long i = 0; i < N_curves * params.N_samples; i++) {
            std::vector<float>::const_iterator first = periodograms.begin() + i * field_data.N_freqs;
            std::vector<float>::const_iterator last = periodograms.begin() + (i + 1) * field_data.N_freqs;
            std::vector<float> periodogramSlice(first, last);

            float mean, stdev;
            std::tie(mean, stdev) = pg_stats(periodogramSlice); // TODO: Replace with Median/MAD

            //TODO: Replace with the GPU version
            std::vector<int> peakIndices = find_peaks(periodogramSlice, N_peaks, params.peak_distance, mean, stdev, params.peak_threshold);

            if (peakIndices[0] > 0) { // If any peaks were found
                for (int j = 0; j < N_peaks; j++) {
                    peakFrequencies[i * N_peaks + j] = field_data.fgrid[peakIndices[j]];
                    peakSignificances[i * N_peaks + j] = (periodogramSlice[peakIndices[j]] - mean) / stdev;
                    // If a value of peakSignificance is less than threshold, set it and the corresponding frequency to -1
                    if (peakSignificances[i * N_peaks + j] < params.peak_threshold) {
                        peakFrequencies[i * N_peaks + j] = -1.0f;
                        peakSignificances[i * N_peaks + j] = -1.0f;
                    }
                }
                periodogramMeans[i] = mean;
                periodogramStdevs[i] = stdev;
            } else { // If periodogram is empty, fill with -1 values
                for (int j = 0; j < N_peaks; j++) { 
                    peakFrequencies[i * N_peaks + j] = -1.0f;
                    peakSignificances[i * N_peaks + j] = -1.0f;
                }
                periodogramMeans[i] = -1.0f;
                periodogramStdevs[i] = -1.0f;
            }
        }

        // Write results to file
        appendResults(results_filename,
                      sources_data.psids.data(),
                      sources_data.ras.data(), sources_data.decs.data(),
                      sources_data.ratioValid.data(),
                      peakFrequencies.data(),
                      peakSignificances.data(),
                      periodogramMeans.data(),
                      periodogramStdevs.data(),
                      N_curves,
                      params.N_samples,
                      N_peaks,
                      objstart);

        // Free memory
        vector<float>().swap(periodograms);
        vector<float>().swap(peakFrequencies);
        vector<float>().swap(peakSignificances);
        vector<float>().swap(periodogramMeans);
        vector<float>().swap(periodogramStdevs);

        auto stop_postprocess = chrono::high_resolution_clock::now();
        auto duration_postprocess = chrono::duration_cast<chrono::milliseconds>(stop_postprocess - start_postprocess);
        logMessage(params.log_path, "Time to compute post-processing: " + to_string(duration_postprocess.count()) + " milliseconds");
    }

    closeSemaphore(semaphore, semaphore_name_char);

    auto stop_full = chrono::high_resolution_clock::now();
    auto duration_full = chrono::duration_cast<chrono::seconds>(stop_full - start_full);
    logMessage(params.log_path, "Time for full run: " + to_string(duration_full.count()) + " seconds");
    cout << "matchfile " << params.input_hdf_filename << " finished in " << duration_full.count() << " seconds" << endl;

    // rename the results file to remove the .temp extension
    string final_results_filename = results_filename;
    // remove the last 5 characters (.temp) from the filename
    final_results_filename = final_results_filename.substr(0, final_results_filename.size() - 5);
    string rename_command = "mv " + results_filename + " " + final_results_filename;
    int rename_status = system(rename_command.c_str());
    if (rename_status != 0) {
        logMessage(params.log_path, "Error renaming results file: " + std::to_string(rename_status));
    } else {
        logMessage(params.log_path, "Results written to: " + final_results_filename);
    }
}