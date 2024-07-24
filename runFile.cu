#include "runFile.cuh"

void runFile(int field, int ccd, int quad, std::string filter) {

    // Write the file path
    string hdf_filename = writeInputFilePath(field, ccd, quad, filter);

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

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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

    printf("  Loading in flux values over %2lu chunks...\n", NCHUNK);
    for (unsigned long k = 0; k < NCHUNK; k++) {

        // Get the range of objects to analyze
        objstart = k * NSRC_PER_CHUNK;
        if (k < NCHUNK-1) {
            objnum = NSRC_PER_CHUNK;
        }
        else {
            objnum = NSRC - (k * NSRC_PER_CHUNK);
        }
        printf("    Chunk %2lu: Starting at source %6lu, loading in %5lu sources\n",
                k+1, objstart, objnum);

        // Load the flux values
        status = get_sourcedata_chunk(hdf_filename, flux.data(), fluxErr.data(), objstart, objnum, NEXP, NEXP);
        check_status(status);

        // Load the psids
        status = get_sources_chunk(hdf_filename, psids.data(), objstart, objnum);

        //Run FPW
        int N_freq_batch = 1000000;
        int N_curves_batch = 100;
        float* t_host = time.data();
        float* flux_host = flux.data();
        float* fluxErr_host = fluxErr.data();
        float* fgrid_host = fgrid.data();
        int N_data = static_cast<int>(NEXP);
        int N_curves = static_cast<int>(objnum);

        //Periodograms are stored light curve contiguous and then sample contiguous. I.E. [curve1_Nbins1, curve1_Nbins2, ...]
        float* periodograms = cudaRunFPW(t_host, flux_host, fluxErr_host, fgrid_host, N_data, N_curves, N_freqs, N_bins, N_freq_batch, N_curves_batch, N_samples);
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
        cout << results_filename << endl;
        
        if (k == 0) {
            writeResults(results_filename, psids.data(), bestFreqs, signifiance, N_curves, NSRC, N_samples);
        }
        else {
            appendResults(results_filename, psids.data(), bestFreqs, signifiance, N_curves, N_samples, objstart);
        }

        // Free memory
        delete[] indices;
        delete[] bestFreqs;
        delete[] signifiance;
        delete[] periodograms;
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
}