#include "../periodograms/FPW/wrappers.cuh"
#include "../readwrite/hdf5_utils.h"
#include "../readwrite/filepaths.h"
#include "../periodograms/frequencyGrid.h"
#include "../statistics/curve_statistics.cuh"
#include "../statistics/polyfit.cuh"
#include "../statistics/peak_utils.h"
#include "../execution/diagnostics.h"

#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <iomanip>
#include <tuple>

std::string writeMetaDataFilePath(const int field, const int ccd, const int quad, const std::string &filter) {
    std::stringstream ss;
    ss << "/data/zvar/tempfiles/file_meta_data/";
    ss << std::setfill('0') << std::setw(4) << field << "/";

    //Check that the directory exists
    std::string directory = ss.str();

    // Check if the directory exists
    if (!std::filesystem::exists(directory)) {
        throw std::runtime_error("Directory does not exist: " + directory);
    }

    ss << std::setfill('0') << std::setw(7) << "metadata_";
    ss << std::setfill('0') << std::setw(4) << field << "_";
    ss << std::setfill('0') << std::setw(2) << ccd << "_";
    ss << quad << "_z" << filter << ".h5";
    std::string filename = ss.str();
    return filename;
}

void writeMetaDataFile(const std::string &filename, unsigned int NEXP, unsigned int NSRC, const std::vector<float> &time, const std::vector<float> &fgrid) {
    // Create a new file using the default properties.
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create the data space for the datasets.
    hsize_t dims[1];
    
    // Write NEXP
    dims[0] = 1;
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate(file_id, "NEXP", H5T_NATIVE_UINT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &NEXP);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write NSRC
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "NSRC", H5T_NATIVE_UINT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &NSRC);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write time vector
    dims[0] = time.size();
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "time", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, time.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write fgrid vector
    dims[0] = fgrid.size();
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "fgrid", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fgrid.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Close the file.
    H5Fclose(file_id);
}

int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <field> <ccd> <quad> <filter> <logpath>" << endl;
        return 1;
    }

    // Parse command-line arguments
    int field = atoi(argv[1]);
    int ccd = atoi(argv[2]);
    int quad = atoi(argv[3]);
    string filter = argv[4];
    string logpath = argv[5];

    auto start_preprocess = chrono::high_resolution_clock::now(); //Time loading and pre-processing

    // Write the file path
    std::string hdf_filename = writeInputFilePath(field, ccd, quad, filter);

    cout << "Running on file: " << hdf_filename.c_str() << endl;
    logMessage(logpath, "Running on file: " + hdf_filename);

    // Initialize frame data
    unsigned int NEXP, NSRC;
    vector<float> time;
    tie(NEXP, NSRC, time) = initializeFrameData(hdf_filename);
    vector<double> time_double(time.begin(), time.end());

    //FPW parameters
    int N_samples = 4;
    int* N_bins = new int[N_samples]{20, 10, 5, 4};

    //Make frequency grid
    double fmin = 1.0 /(24 * 60 * 60); // 1 day
    double fmax = 1.0 /(5 * 60); // 5 minutes
    vector<double> fgrid_double = freq_grid(time_double, fmin, fmax, 3);
    vector<float> fgrid(fgrid_double.begin(), fgrid_double.end()); //convert to floats
    int N_freqs = fgrid.size();

    // Erase terrestrial frequencies from vector
    std::vector<int> freq_flags = flagTerrestrialFreq(fgrid.data(), N_freqs); // Get indices to exclude terrestrial frequencies

    // Remove terrestrial frequencies
    auto badElements = std::remove_if(fgrid.begin(), fgrid.end(), [&](float freq) {
        static int index = 0;
        return freq_flags[index++] == 0;
    });
    fgrid.erase(badElements, fgrid.end());

    N_freqs = fgrid.size(); // Update the NFREQ value
    logMessage(logpath, "Frequency grid size: " + std::to_string(N_freqs));

    //Make output file to store results
    std::string output_filename = writeMetaDataFilePath(field, ccd, quad, filter);
    writeMetaDataFile(output_filename, NEXP, NSRC, time, fgrid);

    std::cout << "Output file: " << output_filename << std::endl;

    auto stop_preprocess = chrono::high_resolution_clock::now();
    auto duration_preprocess = chrono::duration_cast<chrono::milliseconds>(stop_preprocess - start_preprocess);
    logMessage(logpath, "Time to load and pre-process: " + to_string(duration_preprocess.count()) + " milliseconds");

    // Free vectors from memory
    std::vector<float>().swap(time);
    std::vector<float>().swap(fgrid);
    std::vector<double>().swap(time_double);
    std::vector<double>().swap(fgrid_double);
    std::vector<int>().swap(freq_flags);
    delete[] N_bins;
    
    return 0;
}