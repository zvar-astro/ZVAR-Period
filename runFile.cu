#include "runFile.cuh"

/**
 * @brief Runs a single periodogram algorithm on a single matchfile and writes the results to a file
 * 
 * Computes periodograms for every light curve in a matchfile. Find the most significant independent peaks. Writes the results to an HDF5 file.
 * 
 * @param field The field number of the matchfile
 * @param ccd The ccd number of the matchfile 1-16
 * @param quad The quadrant number of the matchfile 1-4
 * @param filter The filter of the matchfile "g", "r", "i"
 * @param logpath The path to the log file
 * 
 */
void runFile(int field, int ccd, int quad, std::string filter, std::string logpath) {
    auto start_full = chrono::high_resolution_clock::now(); //Time full run
    auto start_preprocess = chrono::high_resolution_clock::now(); //Time loading and pre-processing

    // Write the file path
    std::string hdf_filename = writeInputFilePath(field, ccd, quad, filter);
    cout << "Running on file: " << hdf_filename.c_str() << endl;
    logMessage(logpath, "Running on file: " + hdf_filename);

    unsigned int NEXP, NSRC;
    vector<float> time;
    tie(NEXP, NSRC, time) = initializeFrameData(hdf_filename);
    vector<double> time_double(time.begin(), time.end());

    int N_samples = 4;
    int* N_bins = new int[N_samples]{20, 10, 5, 4};

    double fmin = 1.0 /(23 * 60 * 60); // A little less than 1 day
    double fmax = 1.0 /(5 * 60); // 5 minutes
    vector<double> fgrid_double = freq_grid(time_double, fmin, fmax, 3);
    vector<float> fgrid(fgrid_double.begin(), fgrid_double.end()); //convert to floats
    int N_freqs = fgrid.size();
    logMessage(logpath, "Frequency grid size: " + to_string(N_freqs));

    // Load in flux values while iterating over chunks of sources
    // Define Chunk parameters
    int status;
    unsigned long NCHUNK = 100;
    unsigned long NSRC_PER_CHUNK = 1 + ((NSRC-1) / NCHUNK);
    unsigned long objstart, objnum;

    // Initialize flux vectors in chunks
    std::vector<float> flux(NSRC_PER_CHUNK * NEXP, 0.0f);
    std::vector<float> fluxErr(NSRC_PER_CHUNK * NEXP, 1.0f);
    std::vector<unsigned long> psids(NSRC_PER_CHUNK, 0);

    auto stop_preprocess = chrono::high_resolution_clock::now();
    auto duration_preprocess = chrono::duration_cast<chrono::milliseconds>(stop_preprocess - start_preprocess);
    logMessage(logpath, "Time to load and pre-process: " + to_string(duration_preprocess.count()) + " milliseconds");

    logMessage(logpath, "Loading in flux values over " + to_string(NCHUNK) + " chunks with " + to_string(NEXP) + " exposures per source");
    for (unsigned long k = 0; k < NCHUNK; k++) {
        
        auto start_loadchunk = chrono::high_resolution_clock::now(); //Time loading chunk
        // Get the range of objects to analyze
        objstart = k * NSRC_PER_CHUNK;
        if (k < NCHUNK-1) {
            objnum = NSRC_PER_CHUNK;
        }
        else {
            objnum = NSRC - (k * NSRC_PER_CHUNK);
        }
        logMessage(logpath, "Chunk " + to_string(k+1) + ": Starting at source " + to_string(objstart) + ", loading in " + to_string(objnum) + " sources");

        // Load the flux values
        status = get_sourcedata_chunk(hdf_filename, flux.data(), fluxErr.data(), objstart, objnum, NEXP, NEXP);
        check_status(status);

        // Load the psids
        status = get_sources_chunk(hdf_filename, psids.data(), objstart, objnum);

        auto stop_loadchunk = chrono::high_resolution_clock::now();
        auto duration_loadchunk = chrono::duration_cast<chrono::milliseconds>(stop_loadchunk - start_loadchunk);
        logMessage(logpath, "Time to load chunk: " + to_string(duration_loadchunk.count()) + " milliseconds");

        auto start_removebaseline = chrono::high_resolution_clock::now(); //Time removing baseline

        // Remove baseline from flux values
        float* flux_no_baseline = removeBaseline(time.data(), flux.data(), fluxErr.data(), NEXP, NSRC_PER_CHUNK, 15);

        auto stop_removebaseline = chrono::high_resolution_clock::now();
        auto duration_removebaseline = chrono::duration_cast<chrono::milliseconds>(stop_removebaseline - start_removebaseline);
        logMessage(logpath, "Time to remove baseline: " + to_string(duration_removebaseline.count()) + " milliseconds");

        //Run FPW

        auto start_periodogram = chrono::high_resolution_clock::now(); //Time running periodogram

        int N_freq_batch = 1000000;
        int N_curves_batch = 100;
        float* t_host = time.data();
        float* flux_host = flux_no_baseline;
        float* fluxErr_host = fluxErr.data();
        float* fgrid_host = fgrid.data();
        int N_data = static_cast<int>(NEXP);
        int N_curves = static_cast<int>(objnum);

        //Periodograms are stored light curve contiguous and then sample contiguous. I.E. [curve1_Nbins1, curve1_Nbins2, ...]
        float* periodograms = cudaRunFPW(t_host, flux_host, fluxErr_host, fgrid_host, N_data, N_curves, N_freqs, N_bins, N_freq_batch, N_curves_batch, N_samples);

        auto stop_periodogram = chrono::high_resolution_clock::now();
        auto duration_periodogram = chrono::duration_cast<chrono::milliseconds>(stop_periodogram - start_periodogram);
        logMessage(logpath, "Time to run periodogram: " + to_string(duration_periodogram.count()) + " milliseconds");
        logMessage(logpath, "Time per light curve: " + to_string(duration_periodogram.count() / objnum) + " milliseconds");

        // Find the most significant peaks

        auto start_postprocess = chrono::high_resolution_clock::now(); //Time computing post-process

        int* indices = findArgMaxWrapper(periodograms, N_freqs, N_curves * N_samples);
        
        //loop each source and find the best frequency
        float* bestFreqs = new float[N_curves * N_samples];
        for (unsigned long i = 0; i < N_curves * N_samples; i++) {
            int idx = indices[i];
            bestFreqs[i] = fgrid[idx];
        }

        float* signifiance = findSignificanceWrapper(periodograms, indices, N_freqs, N_curves * N_samples);

        // Write results to file
        string results_filename = writeOutputFilePath(field, ccd, quad, filter);
        
        if (k == 0) {
            writeResults(results_filename, psids.data(), bestFreqs, signifiance, N_curves, NSRC, N_samples);
        }
        
        appendResults(results_filename, psids.data(), bestFreqs, signifiance, N_curves, N_samples, objstart);

        // Free memory
        delete[] flux_no_baseline;
        delete[] indices;
        delete[] bestFreqs;
        delete[] signifiance;
        delete[] periodograms;

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
    std::vector<double>().swap(time_double);
    std::vector<double>().swap(fgrid_double);
    delete[] N_bins;

    auto stop_full = chrono::high_resolution_clock::now();
    auto duration_full = chrono::duration_cast<chrono::seconds>(stop_full - start_full);
    logMessage(logpath, "Time for full run: " + to_string(duration_full.count()) + " seconds");
    cout << "matchfile " << field << " " << ccd << " " << quad << " " << filter 
          << " finished in " << duration_full.count() << " seconds" << endl;
}