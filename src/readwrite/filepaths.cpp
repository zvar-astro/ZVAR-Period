#include "filepaths.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

/**
 * @brief Function to initialize frame data from an HDF5 file
 *
 * This function reads the exposure and source tables from an HDF5 file and initializes the time vector.
 *
 * @param filename The path to the HDF5 file.
 * @return A tuple containing the number of exposures, number of sources, and time vector.
 */
std::tuple<unsigned int, unsigned int, std::vector<float>, std::vector<unsigned char>> initializeFrameData(const std::string &filename)
{
    // Variables to store number of exposures & sources
    int NEXP = get_number_of_rows_wrapper(filename, "/data/exposures");
    int NSRC = get_number_of_rows_wrapper(filename, "/data/sources");

    if (NEXP <= 0 || NSRC <= 0)
    {
        throw std::runtime_error("Invalid number of exposures or sources.");
    }

    // Use NEXP and NSRC to initialize time and source vectors
    std::vector<double> h_bjd(NEXP, 0.0);         // BJD must be read in with double precision
    std::vector<unsigned long> sourceid(NSRC, 0); // Source IDs

    // Read HDF5 tables and load into vectors
    int status = get_exposures_wrapper(filename, h_bjd.data(), 0, NEXP);
    check_status(status);
    status = get_sources_wrapper(filename, sourceid.data(), 0, NSRC);
    check_status(status);

    // Get indices of non-deep drilling time stamps
    std::vector<unsigned char> deep_drilling_flags = flagDeepDrilling(h_bjd, NEXP);
    int non_dd = std::accumulate(deep_drilling_flags.begin(), deep_drilling_flags.end(), 0);

    // Filter out deep drilling BJDs
    auto new_end = std::remove_if(h_bjd.begin(), h_bjd.end(),
                                  [&deep_drilling_flags, index = 0](double) mutable
                                  {
                                      return deep_drilling_flags[index++] == 0;
                                  });
    h_bjd.erase(new_end, h_bjd.end());

    if (h_bjd.empty())
    {
        throw std::runtime_error("No valid BJDs after filtering.");
    }

    NEXP = non_dd; // Update number of exposures

    // Subtract minimum BJD and convert to float
    double bjdref = *std::min_element(h_bjd.begin(), h_bjd.end());
    std::vector<float> time(NEXP, 0.0f);
    std::transform(h_bjd.begin(), h_bjd.end(), time.begin(),
                   [bjdref](double bjd)
                   {
                       return static_cast<float>(bjd - bjdref);
                   });

    return std::make_tuple(NEXP, NSRC, time, deep_drilling_flags);
}