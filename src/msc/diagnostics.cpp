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

// void savePeriodogram(const std::string& filename, const std::vector<double>& freqs, const std::vector<double>& powers) {
//         // Check if psids contains a specific value for debugging purposes
//         unsigned long psid_check = 157282011398486575; // all nans
//         // unsigned long psid_check = 157182014286022129; //Known periodogram
//         if (std::find(psids.begin(), psids.end(), psid_check) != psids.end()) {
//             std::cout << "Found psid " << psid_check << " in chunk " << k << std::endl;
//             //append the psid's periodograms to a file

//             //Find the index of the psid
//             auto it = std::find(psids.begin(), psids.end(), psid_check);
//             int index = std::distance(psids.begin(), it);

//             //Write the periodograms to a file
//             std::ofstream csvFile("test_periodogram.csv");

//             // //Write the header
//             // csvFile << "Frequency,Periodogram" << std::endl;

//             // //Write the periodograms
//             // for (unsigned long i = 0; i < N_freqs; i++) {
//             //     csvFile << fgrid[i] << "," << periodograms[index * N_samples * N_freqs + i] << std::endl;
//             // }

//             //write the header
//             csvFile << "Time,Flux,FluxErr" << std::endl;

//             //write the light curve
//             for (unsigned long i = 0; i < NEXP; i++) {
//                 csvFile << time[i] << "," << flux_host[index * NEXP + i] << "," << fluxErr_host[index * NEXP + i] << std::endl;
//             }

//             //Close the file
//             csvFile.close();
//             std::cout << "Wrote periodograms to test_periodogram.csv" << std::endl;
//         }
// }