#include "runFile.cuh"

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
 * @param field The field number of the matchfile
 * @param ccd The ccd number of the matchfile 1-16
 * @param quad The quadrant number of the matchfile 1-4
 * @param filter The filter of the matchfile "g", "r", "i"
 * @param logpath The path to the log file
 * 
 */
void runFile(int field, int ccd, int quad, std::string filter, int device, std::string logpath) {
    omp_set_num_threads(4);
    std::string semaphore_name = "/device_" + std::to_string(device) + "_lock";
    const char* SEMAPHORE_NAME = semaphore_name.c_str();
    sem_t* semaphore = openSemaphore(SEMAPHORE_NAME);

    auto start_full = chrono::high_resolution_clock::now(); //Time full run
    auto start_preprocess = chrono::high_resolution_clock::now(); //Time loading and pre-processing

    // Write the file path
    std::string hdf_filename = writeInputFilePath(field, ccd, quad, filter);

    // cout << "Running on file: " << hdf_filename.c_str() << endl;
    logMessage(logpath, "Running on file: " + hdf_filename);

    // Initialize frame data
    unsigned int NEXP, NSRC;
    vector<float> time;
    vector<unsigned char> dd_indices;
    tie(NEXP, NSRC, time, dd_indices) = initializeFrameData(hdf_filename);
    vector<double> time_double(time.begin(), time.end());

    //FPW parameters
    int N_samples = 3;
    int* N_bins = new int[N_samples]{20, 10, 5};

    //Make frequency grid
    double fmin = 1.0; // 1 day
    double fmax = 1440.0 / 5; // 5 minutes
    vector<double> fgrid_double = freq_grid(time_double, fmin, fmax, 3);
    vector<float> fgrid(fgrid_double.begin(), fgrid_double.end()); //convert to floats
    int N_freqs = fgrid.size();

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

    // Load in flux values while iterating over chunks of sources
    // Define Chunk parameters
    int status;
    unsigned long NCHUNK = 50;
    unsigned long NSRC_PER_CHUNK = 1 + ((NSRC-1) / NCHUNK);
    unsigned long objstart, objnum;

    // Initialize flux vectors in chunks
    std::vector<float> flux(NSRC_PER_CHUNK * NEXP, 0.0f);
    std::vector<float> fluxErr(NSRC_PER_CHUNK * NEXP, 1.0f);
    std::vector<float> ratioValid(NSRC_PER_CHUNK, 0);
    std::vector<unsigned long> psids(NSRC_PER_CHUNK, 0);
    std::vector<float> ra(NSRC_PER_CHUNK, 0.0f);
    std::vector<float> dec(NSRC_PER_CHUNK, 0.0f);

    auto stop_preprocess = chrono::high_resolution_clock::now();
    auto duration_preprocess = chrono::duration_cast<chrono::milliseconds>(stop_preprocess - start_preprocess);
    logMessage(logpath, "Time to load and pre-process: " + to_string(duration_preprocess.count()) + " milliseconds");

    logMessage(logpath, "Loading in flux values over " + to_string(NCHUNK) + " chunks with " + to_string(NEXP) + " exposures per source");

    for (unsigned long k = 0; k < NCHUNK; k++) {
        auto start_loadchunk = chrono::high_resolution_clock::now(); // Time loading chunk
        // Get the range of objects to analyze
        objstart = k * NSRC_PER_CHUNK;
        if (k < NCHUNK-1) {
            objnum = NSRC_PER_CHUNK;
        } else {
            objnum = NSRC - (k * NSRC_PER_CHUNK);
        }
        logMessage(logpath, "Chunk " + to_string(k+1) + ": Starting at source " + to_string(objstart) + ", loading in " + to_string(objnum) + " sources");

        // Load the flux values
        status = get_sourcedata_chunk(hdf_filename, flux.data(), fluxErr.data(), ratioValid.data(), dd_indices, objstart, objnum, NEXP);
        check_status(status);

        // Load the psids
        status = get_sources_chunk(hdf_filename, psids.data(), ra.data(), dec.data(), objstart, objnum);

        auto stop_loadchunk = chrono::high_resolution_clock::now();
        auto duration_loadchunk = chrono::duration_cast<chrono::milliseconds>(stop_loadchunk - start_loadchunk);
        logMessage(logpath, "Time to load chunk: " + to_string(duration_loadchunk.count()) + " milliseconds");

        auto start_removebaseline = chrono::high_resolution_clock::now(); // Time removing baseline

        //Utilize a semaphore to prevent multiple threads from accessing the GPU at the same time
        acquireSemaphore(semaphore);
        // std::cout << "Semaphore acquired for baseline removal in quadrant: " << quad << " chunk: " << k << std::endl;

        // Remove baseline from flux values
        float* flux_no_baseline = removeBaseline(time.data(), flux.data(), fluxErr.data(), NEXP, NSRC_PER_CHUNK, 15);

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        checkCudaError(cudaStatus, "cudaDeviceSynchronize after removeBaseline");

        // std::cout << "Semaphore releasing for baseline removal in quadrant: " << quad << " chunk: " << k << std::endl;
        releaseSemaphore(semaphore);
        auto stop_removebaseline = chrono::high_resolution_clock::now();
        auto duration_removebaseline = chrono::duration_cast<chrono::milliseconds>(stop_removebaseline - start_removebaseline);
        logMessage(logpath, "Time to remove baseline: " + to_string(duration_removebaseline.count()) + " milliseconds");

        // Run FPW
        auto start_periodogram = chrono::high_resolution_clock::now(); // Time running periodogram
        //Open the semaphore for the periodogram
        acquireSemaphore(semaphore);
        // std::cout << "Semaphore acquired for periodogram computation in quadrant: " << quad << " chunk: " << k << std::endl;

        int N_freq_batch = 1000000;
        int N_curves_batch = 500;
        float* t_host = time.data();
        float* flux_host = flux_no_baseline;
        float* fluxErr_host = fluxErr.data();
        float* fgrid_host = fgrid.data();
        int N_data = static_cast<int>(NEXP);
        int N_curves = static_cast<int>(objnum);

        // Periodograms are stored light curve contiguous and then sample contiguous. I.E. [curve1_Nbins1, curve1_Nbins2, ...]
        vector<float> periodograms = cudaRunFPW(t_host, flux_host, fluxErr_host, fgrid_host, N_data, N_curves, N_freqs, N_bins, N_freq_batch, N_curves_batch, N_samples);

        // std::cout << "Semaphore releasing for periodogram computation in quadrant: " << quad << " chunk: " << k << std::endl;
        releaseSemaphore(semaphore);
        auto stop_periodogram = chrono::high_resolution_clock::now();
        auto duration_periodogram = chrono::duration_cast<chrono::milliseconds>(stop_periodogram - start_periodogram);
        logMessage(logpath, "Time to run periodogram: " + to_string(duration_periodogram.count()) + " milliseconds");
        logMessage(logpath, "Time per light curve: " + to_string(duration_periodogram.count() / objnum) + " milliseconds");

        // Find the most significant peaks
        auto start_postprocess = chrono::high_resolution_clock::now(); // Time computing post-process

        const int distance = 300;
        const float threshold = 7.0f;
        const int N_peaks = 50;

        vector<float> peakFrequencies(N_curves * N_samples * N_peaks, 0.0f);
        vector<float> peakSignificances(N_curves * N_samples * N_peaks, 0.0f);
        vector<float> periodogramMeans(N_curves * N_samples, 0.0f);
        vector<float> periodogramStdevs(N_curves * N_samples, 0.0f);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_curves * N_samples; i++) {
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
        string results_filename = writeOutputFilePath(field, ccd, quad, filter);
        
        if (k == 0) {
            writeResults(results_filename, N_curves, NSRC, N_samples, N_peaks);
        }
        
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
    cout << "matchfile " << field << " " << ccd << " " << quad << " " << filter 
          << " finished in " << duration_full.count() << " seconds" << endl;
}