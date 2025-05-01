#include "processing.cuh"

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Runs a single periodogram algorithm on a single matchfile and writes the results to a file
 * 
 * Computes periodograms for every light curve in a matchfile. Finds the most significant independent peaks. Writes the results to an HDF5 file.
 * 
 * @param hdf_filename The path to the matchfile
 * @param logpath The path to the log file
 * 
 */
void process(
    std::string hdf_filename,
    int device,
    std::string logpath,
    const char *id_field_name,
    const char *ra_field_name,
    const char *dec_field_name,
    const char *time_field_name,
    const char *flux_field_name,
    const char *fluxErr_field_name
) {

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "unexpected CUDA error (process): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    omp_set_num_threads(4); // Set the number of threads for OpenMP

    //Obtain the semaphore for the device
    std::string semaphore_name = "/device_" + std::to_string(device) + "_lock";
    const char* SEMAPHORE_NAME = semaphore_name.c_str();
    sem_t* semaphore = openSemaphore(SEMAPHORE_NAME); // Open the semaphore

    auto start_full = chrono::high_resolution_clock::now(); //Time full run
    auto start_preprocess = chrono::high_resolution_clock::now(); //Time loading and pre-processing

    logMessage(logpath, "Running on file: " + hdf_filename);

    //FPW parameters
    int N_samples = 3;
    int* N_bins = new int[N_samples]{20, 10, 5};

    // Initialize frame data
    unsigned int NEXP, NSRC; // Number of exposures and sources
    vector<float> time; // Time values in days
    vector<unsigned char> dd_indices; // Indices of deep drilling data
    // TODO: pass the name of the time column as an argument
    std::cout << "Loading frame data..." << std::endl;
    tie(NEXP, NSRC, time, dd_indices) = initializeFrameData(hdf_filename, id_field_name, time_field_name);
    std::cout << "initializeFrameData finished" << std::endl;

    vector<double> time_double(time.begin(), time.end());

    // Make frequency grid
    // double fmin = 1.0; // 1 day
    double fmin = 1.0/10.0; // 10 day maximum period
    // double fmin = -1.0; // No max period
    double fmax = 1440.0 / 5; // 5 minute minimum period
    vector<double> fgrid_double = freq_grid(time_double, fmin, fmax, 3); //frequency grid in 1/days
    vector<float> fgrid(fgrid_double.begin(), fgrid_double.end()); //convert to floats
    unsigned long N_freqs = fgrid.size();

    // Erase terrestrial frequencies from vector
    auto start_erase_freq = chrono::high_resolution_clock::now();
    std::vector<int> freq_flags = flagTerrestrialFreq(fgrid.data(), N_freqs); // Get indices to exclude terrestrial frequencies

    // Remove terrestrial frequencies
    auto badElements = std::remove_if(fgrid.begin(), fgrid.end(), [&](float freq) {
        static int index = 0;
        return freq_flags[index++] == 0;
    });
    fgrid.erase(badElements, fgrid.end());

    N_freqs = fgrid.size(); // Update the NFREQ value
    logMessage(logpath, "Frequency grid size: " + std::to_string(N_freqs));

    //Decide how many CPU chunks to run based on memory
    unsigned long NCHUNK = calculateMaxCPUBatch(N_freqs, N_samples, NEXP, NSRC);
    unsigned long NSRC_PER_CHUNK = 1 + ((NSRC-1) / NCHUNK);

    // Decide how many GPU chunks to run based on memory
    const char* N_freq_batch_env = std::getenv("N_FREQ_BATCH");
    unsigned long N_freq_batch = (N_freq_batch_env != nullptr) ? std::stoul(N_freq_batch_env) : 100000;
    if (N_freq_batch == 0) {
        std::cerr << "Error: N_FREQ_BATCH cannot be zero" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (N_freq_batch > N_freqs) {
        // make it match the size of the frequency grid
        std::cerr << "Warning: N_FREQ_BATCH is larger than the size of the frequency grid, setting it to " << N_freqs << std::endl;
        N_freq_batch = N_freqs;
    }
    unsigned long N_curves_batch = calculateMaxGPUBatch(device, NEXP, N_freq_batch, N_samples, NSRC_PER_CHUNK);

    if (N_curves_batch == 0) {
        std::cerr << "Error: N_curves_batch is zero, exiting" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "N_freq_batch: " << N_freq_batch << std::endl;
    std::cout << "N_curves_batch: " << N_curves_batch << std::endl;

    const int distance = 300;
    const float threshold = 7.0f;
    const int N_peaks = 50;
    
    // Initialize flux vectors in chunks
    int status;
    unsigned long objstart, objnum;
    std::vector<float> flux(NSRC_PER_CHUNK * NEXP, 0.0f);
    std::vector<float> fluxErr(NSRC_PER_CHUNK * NEXP, 1.0f);
    std::vector<float> ratioValid(NSRC_PER_CHUNK, 0);
    std::vector<unsigned long> psids(NSRC_PER_CHUNK, 0);
    std::vector<float> ra(NSRC_PER_CHUNK, 0.0f);
    std::vector<float> dec(NSRC_PER_CHUNK, 0.0f);

    // Create output file, in OUTPUT_DIR if it exists, otherwise in /data/zvar/output
    const char* outputDirEnv = std::getenv("OUTPUT_DIR");
    std::string outputDir = outputDirEnv ? std::string(outputDirEnv) : "/data/output";
    std::string mkdir = "mkdir -p " + outputDir;
    system(mkdir.c_str());
    std::string hdf_filename_copy = hdf_filename;
    std::string::size_type pos = hdf_filename_copy.find_last_of("/");
    if (pos != std::string::npos) {
        hdf_filename_copy = hdf_filename_copy.substr(pos + 1);
    }
    std::string results_filename = outputDir + "/" + hdf_filename_copy + "_results.h5.temp";
    createResults(results_filename, NSRC, N_samples, N_peaks);

    auto stop_preprocess = chrono::high_resolution_clock::now();
    auto duration_preprocess = chrono::duration_cast<chrono::milliseconds>(stop_preprocess - start_preprocess);
    logMessage(logpath, "Time to load and pre-process: " + to_string(duration_preprocess.count()) + " milliseconds");

    // If there are no exposures, skip the file
    if (NEXP == 0) {
        logMessage(logpath, "No exposures found in file, skipping");
        std::vector<float>().swap(time);
        std::vector<float>().swap(flux);
        std::vector<float>().swap(fluxErr);
        std::vector<float>().swap(fgrid);
        std::vector<unsigned long>().swap(psids);
        std::vector<float>().swap(ra);
        std::vector<float>().swap(dec);
        std::vector<double>().swap(time_double);
        std::vector<double>().swap(fgrid_double);
        std::vector<int>().swap(freq_flags);
        delete[] N_bins;

        closeSemaphore(semaphore, SEMAPHORE_NAME);
        return;
    }

    // print how many sources we have in total, how many chunks we will run, how many sources per chunk
    // and how many exposures per source
    logMessage(logpath, "Total sources: " + to_string(NSRC) + ", Total chunks: " + to_string(NCHUNK) + ", Sources per chunk: " + to_string(NSRC_PER_CHUNK) + ", Exposures per source: " + to_string(NEXP));

    for (unsigned long k = 0; k < NCHUNK; k++) {
        auto start_loadchunk = chrono::high_resolution_clock::now(); // Time loading chunk
        // Get the range of objects to analyze
        objstart = k * NSRC_PER_CHUNK;
        unsigned long objnum = (k < NCHUNK - 1) ? NSRC_PER_CHUNK : (NSRC - objstart);

        // Load the flux and fluxerr values
        // TODO: pass the name of the flux and fluxerr columns as arguments
        status = get_sourcedata_chunk(
            hdf_filename, flux.data(), fluxErr.data(), ratioValid.data(),
            dd_indices, objstart, objnum, NEXP,
            flux_field_name, fluxErr_field_name
        );
        check_status(status);

        // Load the ids, ra, dec values
        // TODO: pass the name of the id, ra, dec columns as arguments
        status = get_sources_chunk(
            hdf_filename, psids.data(), ra.data(), dec.data(),
            objstart, objnum,
            id_field_name, ra_field_name, dec_field_name
        );
        check_status(status);

        //Check for any nans in the flux values
        for (unsigned long i = 0; i < objnum; i++) {
            for (unsigned long j = 0; j < NEXP; j++) {
                if (std::isnan(flux[i * NEXP + j])) {
                    flux[i * NEXP + j] = 0.0f;
                    fluxErr[i * NEXP + j] = INFINITY;
                }
            }
        }

        auto stop_loadchunk = chrono::high_resolution_clock::now();
        auto duration_loadchunk = chrono::duration_cast<chrono::milliseconds>(stop_loadchunk - start_loadchunk);

        auto start_removebaseline = chrono::high_resolution_clock::now(); // Time removing baseline

        //Utilize a semaphore to prevent multiple threads from accessing the GPU at the same time
        acquireSemaphore(semaphore);

        logMessage(logpath, "Semaphore acquired for baseline removal in chunk: " + to_string(k));

        // Remove baseline from flux values
        unsigned long NCHUNK_BASELINE = calculateNChunkBaseline(device, NSRC_PER_CHUNK, NEXP, 15);
        unsigned long NSRC_PER_CHUNK_BASELINE = 1 + ((NSRC_PER_CHUNK-1) / NCHUNK_BASELINE);

        float* flux_no_baseline = new float[NSRC_PER_CHUNK * NEXP];

        for (unsigned long l = 0; l < NCHUNK_BASELINE; l++) {
            unsigned long objstart_baseline = l * NSRC_PER_CHUNK_BASELINE;
            unsigned long objnum_baseline = (l < NCHUNK_BASELINE - 1) ? NSRC_PER_CHUNK_BASELINE : (NSRC_PER_CHUNK - objstart_baseline);

            float* flux_no_baseline_chunk = removeBaseline(time.data(), flux.data() + objstart_baseline * NEXP, fluxErr.data() + objstart_baseline * NEXP, NEXP, objnum_baseline, 15);
            for (unsigned long i = 0; i < objnum_baseline; i++) {
                for (unsigned long j = 0; j < NEXP; j++) {
                    flux_no_baseline[(objstart_baseline + i) * NEXP + j] = flux_no_baseline_chunk[i * NEXP + j];
                }
            }
            delete[] flux_no_baseline_chunk;
        }

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error cudaDeviceSynchronize: " << cudaGetErrorString(cudaStatus) << std::endl;
            exit(EXIT_FAILURE);
        }

        releaseSemaphore(semaphore);
        auto stop_removebaseline = chrono::high_resolution_clock::now();
        auto duration_removebaseline = chrono::duration_cast<chrono::milliseconds>(stop_removebaseline - start_removebaseline);
        logMessage(logpath, "Time to remove baseline: " + to_string(duration_removebaseline.count()) + " milliseconds");

        // Run FPW
        auto start_periodogram = chrono::high_resolution_clock::now(); // Time running periodogram
        //Open the semaphore for the periodogram
        acquireSemaphore(semaphore);
        logMessage(logpath, "Semaphore acquired for periodogram computation in chunk: " + to_string(k));

        float* t_host = time.data();
        float* flux_host = flux_no_baseline;
        float* fluxErr_host = fluxErr.data();
        float* fgrid_host = fgrid.data();
        int N_data = static_cast<int>(NEXP);
        unsigned long N_curves = objnum;

        // Periodograms are stored light curve contiguous and then sample contiguous. I.E. [curve1_Nbins1, curve1_Nbins2, ...]
        vector<float> periodograms = cudaRunFPW(t_host, flux_host, fluxErr_host, fgrid_host, N_data, N_curves, N_freqs, N_bins, N_freq_batch, N_curves_batch, N_samples);

        releaseSemaphore(semaphore);
        auto stop_periodogram = chrono::high_resolution_clock::now();
        auto duration_periodogram = chrono::duration_cast<chrono::milliseconds>(stop_periodogram - start_periodogram);
        logMessage(logpath, "Time to run periodogram: " + to_string(duration_periodogram.count()) + " ms (per light curve: " + to_string(duration_periodogram.count() / objnum) + " ms)");

        // Find the most significant peaks
        auto start_postprocess = chrono::high_resolution_clock::now(); // Time computing post-process

        vector<float> peakFrequencies(N_curves * N_samples * N_peaks, 0.0f);
        vector<float> peakSignificances(N_curves * N_samples * N_peaks, 0.0f);
        vector<float> periodogramMeans(N_curves * N_samples, 0.0f);
        vector<float> periodogramStdevs(N_curves * N_samples, 0.0f);

        #pragma omp parallel for schedule(dynamic)
        for (unsigned long i = 0; i < N_curves * N_samples; i++) {
            std::vector<float>::const_iterator first = periodograms.begin() + i * N_freqs;
            std::vector<float>::const_iterator last = periodograms.begin() + (i + 1) * N_freqs;
            std::vector<float> periodogramSlice(first, last);

            float mean, stdev;
            std::tie(mean, stdev) = pg_stats(periodogramSlice);
            std::vector<int> peakIndices = find_peaks(periodogramSlice, N_peaks, distance, mean, stdev, threshold);

            if (peakIndices[0] > 0) { // If any peaks were found
                for (int j = 0; j < N_peaks; j++) {
                    peakFrequencies[i * N_peaks + j] = fgrid[peakIndices[j]];
                    peakSignificances[i * N_peaks + j] = (periodogramSlice[peakIndices[j]] - mean) / stdev;
                    // If a value of peakSignificance is less than threshold, set it and the corresponding frequency to -1
                    if (peakSignificances[i * N_peaks + j] < threshold) {
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
        appendResults(results_filename, psids.data(), ra.data(), dec.data(), ratioValid.data(), peakFrequencies.data(), peakSignificances.data(), periodogramMeans.data(), periodogramStdevs.data(),
                N_curves, N_samples, N_peaks, objstart);

        // Free memory
        delete[] flux_no_baseline;
        vector<float>().swap(periodograms);
        vector<float>().swap(peakFrequencies);
        vector<float>().swap(peakSignificances);
        vector<float>().swap(periodogramMeans);
        vector<float>().swap(periodogramStdevs);

        auto stop_postprocess = chrono::high_resolution_clock::now();
        auto duration_postprocess = chrono::duration_cast<chrono::milliseconds>(stop_postprocess - start_postprocess);
        logMessage(logpath, "Time to compute post-processing: " + to_string(duration_postprocess.count()) + " milliseconds");
    }

    // Free vectors from memory
    std::vector<float>().swap(time);
    std::vector<float>().swap(flux);
    std::vector<float>().swap(fluxErr);
    std::vector<float>().swap(fgrid);
    std::vector<unsigned long>().swap(psids);
    std::vector<float>().swap(ra);
    std::vector<float>().swap(dec);
    std::vector<double>().swap(time_double);
    std::vector<double>().swap(fgrid_double);
    std::vector<int>().swap(freq_flags);
    delete[] N_bins;

    closeSemaphore(semaphore, SEMAPHORE_NAME);

    auto stop_full = chrono::high_resolution_clock::now();
    auto duration_full = chrono::duration_cast<chrono::seconds>(stop_full - start_full);
    logMessage(logpath, "Time for full run: " + to_string(duration_full.count()) + " seconds");
    cout << "matchfile " << hdf_filename << " finished in " << duration_full.count() << " seconds" << endl;

    // rename the results file to remove the .temp extension
    string final_results_filename = results_filename;
    // remove the last 5 characters (.temp) from the filename
    final_results_filename = final_results_filename.substr(0, final_results_filename.size() - 5);
    string rename_command = "mv " + results_filename + " " + final_results_filename;
    int rename_status = system(rename_command.c_str());
    if (rename_status != 0) {
        logMessage(logpath, "Error renaming results file: " + std::to_string(rename_status));
    } else {
        logMessage(logpath, "Results written to: " + final_results_filename);
    }
}