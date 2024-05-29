#include "read_data.h"

std::string writeFilePath(const int field, const int ccd, const int quad, const std::string &filter) {
    std::stringstream ss;
    std::stringstream ssr;
    ss << "/home/swhitebook/data/";
    ss << std::setfill('0') << std::setw(4) << "data_";
    ss << std::setfill('0') << std::setw(4) << field << "_";
    ss << std::setfill('0') << std::setw(2) << ccd << "_";
    ss << quad << "_zg.h5";
    std::string filename = ss.str();
    return filename;
}

std::tuple<unsigned int, unsigned int, std::vector<float>> initializeFrameData(const std::string &filename) {
    // Variables to store number of exposures & sources
    int NEXP = get_number_of_rows_wrapper(filename, "/data/exposures");
    int NSRC = get_number_of_rows_wrapper(filename, "/data/sources");

    // Use NEXP and NSRC to initialize time and source vectors
    std::vector<double> bjd(NEXP,0.0);  // BJD must be read in with double precision
    std::vector<float> time(NEXP,0.0f); // Time values after subtracting mean BJD
    std::vector<unsigned long> sourceid(NSRC,0);  // Source IDs

    // Read HDF5 tables and load into vectors
    int status;
    status = get_exposures_wrapper(filename, bjd.data(), 0, NEXP);
    check_status(status);
    status = get_sources_wrapper(filename, sourceid.data(), 0, NSRC);
    check_status(status);

    // Subtract mean BJD and convert to float
    double bjdref = std::accumulate(bjd.begin(), bjd.end(), 0.0) / 
                    static_cast<double>(NEXP);
    for (int i = 0; i < NEXP; i++) {
    time[i] = static_cast<float>(bjd[i] - bjdref);
    }
    return std::make_tuple(NEXP, NSRC, time);
}