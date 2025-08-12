#include "diagnostics.h"

/**
 * @brief Log a message to a file
 * 
 * @param logfile The file to log to
 * @param message The message to log
 */
void logMessage(const std::string& logfile, const std::string& message) {
    std::ofstream log_file(logfile, std::ios_base::out | std::ios_base::app);
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    log_file << std::ctime(&now_time) << message << std::endl;
    std::cout << message << std::endl;
}

void computeAmplitudes(const std::vector<float>& fluxes_no_baseline,
                       std::vector<float>& amplitudes,
                       unsigned long N_exp,
                       unsigned long N_src) {
    
    /*
    Computes an amplitude for every source by finding the maximum and minimum
    */
    for (unsigned long i = 0; i < N_src; ++i) {
        auto minmax = std::minmax_element(fluxes_no_baseline.begin() + i * N_exp,
                                           fluxes_no_baseline.begin() + (i + 1) * N_exp);
        amplitudes[i] = *minmax.second - *minmax.first;
    }
}

void saveObjectTimesAndFluxToCSV(const std::string& filename,
                                 const std::vector<float>& times,
                                 const std::vector<float>& fluxes_no_baseline,
                                 const std::vector<float>& flux_errs,
                                 unsigned long N_exp,
                                 unsigned long object_index) {
    /*
    Saves the times and baseline-removed fluxes for a specific object to a CSV file.

    filename            = Name of the output CSV file
    times               = Vector of times (length N_exp)
    fluxes_no_baseline  = Vector of all baseline-removed fluxes (length N_src * N_exp)
    flux_errs           = Vector of all flux errors (length N_src * N_exp)
    N_exp               = Number of exposures per source
    object_index        = Index of the object to save
    */

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    file << "Time,FluxNoBaseline,FluxErr\n";
    unsigned long offset = object_index * N_exp;
    for (unsigned long i = 0; i < N_exp; ++i) {
        file << times[i] << "," << fluxes_no_baseline[offset + i] << "," << flux_errs[offset + i] << "\n";
    }

    file.close();
}

void writeFluxStatisticsToCSV(const std::string& filename,
                              const std::vector<float>& flux_medians,
                              const std::vector<float>& flux_mads,
                              const std::vector<float>& amplitudes,
                              unsigned long objstart_periodograms,
                              bool is_first_chunk) {
    /*
    Function to write or append flux medians and MADs to a CSV file.

    filename            = Name of the output CSV file
    flux_medians        = Vector of flux medians
    flux_mads           = Vector of flux MADs
    amplitudes          = Vector of flux amplitudes
    objstart_periodograms = Starting index for the current chunk
    is_first_chunk      = Flag indicating if this is the first source chunk
    */

    std::ofstream file;

    // Open the file in append mode if not the first chunk, otherwise overwrite
    if (is_first_chunk) {
        file.open(filename, std::ios::out); // Overwrite file
        file << "SourceIndex,FluxMedian,FluxMAD,FluxAmplitude\n"; // Write header
    } else {
        file.open(filename, std::ios::app); // Append to file
    }

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Write data
    for (unsigned long i = 0; i < flux_medians.size(); ++i) {
        file << (objstart_periodograms + i) << "," << flux_medians[i] << "," << flux_mads[i] << "," << amplitudes[i] << "\n";
    }

    file.close();
}

void writePeriodogramStatisticsToCSV(const std::string& base_filename,
                                     const std::vector<float>& periodogram_medians,
                                     const std::vector<float>& periodogram_mads,
                                     unsigned long objstart_periodograms,
                                     unsigned long N_src_per_batch,
                                     unsigned long N_samples,
                                     bool is_first_chunk) {
    /*
    Function to write or append periodogram medians and MADs for each bin to separate CSV files.

    base_filename       = Base name for the output CSV files
    periodogram_medians = Vector of periodogram medians
    periodogram_mads    = Vector of periodogram MADs
    objstart_periodograms = Starting index for the current chunk
    N_src_per_batch     = Number of sources in the current batch
    N_samples           = Number of bins per source
    is_first_chunk      = Flag indicating if this is the first source chunk
    */

    for (unsigned long bin_idx = 0; bin_idx < N_samples; ++bin_idx) {
        // Construct the filename for the current bin
        std::string filename = base_filename + "_bin_" + std::to_string(bin_idx) + ".csv";

        std::ofstream file;

        // Open the file in append mode if not the first chunk, otherwise overwrite
        if (is_first_chunk) {
            file.open(filename, std::ios::out); // Overwrite file
            file << "SourceIndex,PeriodogramMedian,PeriodogramMAD\n"; // Write header
        } else {
            file.open(filename, std::ios::app); // Append to file
        }

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Write data for the current bin
        for (unsigned long src_idx = 0; src_idx < N_src_per_batch; ++src_idx) {
            unsigned long index = src_idx * N_samples + bin_idx;
            file << (objstart_periodograms + src_idx) << "," << periodogram_medians[index] << "," << periodogram_mads[index] << "\n";
        }

        file.close();
    }
}