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

std::string writePreProcessChunkFilePath(const int chunkNum, const int field, const int ccd, const int quad, const std::string &filter) {
    std::stringstream ss;
    ss << "/data/zvar/tempfiles/source_chunk_preprocess/";
    ss << std::setfill('0') << std::setw(4) << field << "/";

    //Check that the directory exists
    std::string directory = ss.str();

    // Check if the directory exists
    if (!std::filesystem::exists(directory)) {
        throw std::runtime_error("Directory does not exist: " + directory);
    }

    ss << std::setfill('0') << std::setw(7) << "photometry_";
    ss << std::setfill('0') << std::setw(4) << chunkNum << "_";
    ss << std::setfill('0') << std::setw(4) << field << "_";
    ss << std::setfill('0') << std::setw(2) << ccd << "_";
    ss << quad << "_z" << filter << ".h5";
    std::string filename = ss.str();
    return filename;
}

void readHDF5File(const std::string &filename, unsigned int &NEXP, unsigned int &NSRC, std::vector<float> &time, std::vector<float> &fgrid) {
    // Open an existing file.
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Read NEXP
    hid_t dataset_id = H5Dopen(file_id, "NEXP", H5P_DEFAULT);
    H5Dread(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &NEXP);
    H5Dclose(dataset_id);

    // Read NSRC
    dataset_id = H5Dopen(file_id, "NSRC", H5P_DEFAULT);
    H5Dread(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &NSRC);
    H5Dclose(dataset_id);

    // Read time vector
    dataset_id = H5Dopen(file_id, "time", H5P_DEFAULT);
    hid_t dataspace_id = H5Dget_space(dataset_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    time.resize(dims[0]);
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, time.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Read fgrid vector
    dataset_id = H5Dopen(file_id, "fgrid", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    fgrid.resize(dims[0]);
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fgrid.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Close the file.
    H5Fclose(file_id);
}

void writePreProcessOutputFile(const std::string &output_filename, 
                         const std::vector<unsigned long> &psids, 
                         const std::vector<unsigned short> &ratioValid, 
                         const std::vector<float> &ra, 
                         const std::vector<float> &dec, 
                         const std::vector<float> &fluxNoBaseline, 
                         const std::vector<float> &fluxErr) {
    // Create a new file using the default properties.
    hid_t file_id = H5Fcreate(output_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write psids
    hsize_t dims[1] = { psids.size() };
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate(file_id, "psids", H5T_NATIVE_ULONG, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, psids.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write ratioValid
    dims[0] = ratioValid.size();
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "ratioValid", H5T_NATIVE_USHORT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ratioValid.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write ra
    dims[0] = ra.size();
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "ra", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ra.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write dec
    dims[0] = dec.size();
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "dec", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dec.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write fluxNoBaseline
    dims[0] = fluxNoBaseline.size();
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "fluxNoBaseline", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fluxNoBaseline.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write fluxErr
    dims[0] = fluxErr.size();
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "fluxErr", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fluxErr.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Close the file.
    H5Fclose(file_id);
}

// int main(int argc, char* argv[]) {
//     // Check if the correct number of arguments is provided
//     if (argc != 7) {
//         cerr << "Usage: " << argv[0] << " <field> <ccd> <quad> <filter> <logpath>" << endl;
//         return 1;
//     }

//     // Parse command-line arguments
//     int field = atoi(argv[1]);
//     int ccd = atoi(argv[2]);
//     int quad = atoi(argv[3]);
//     string filter = argv[4];
//     int chunkNum = atoi(argv[6]);
//     string logpath = argv[5];

int main() {
    int field = 504;
    int ccd = 1;
    int quad = 1;
    std::string filter = "g";
    string logpath = "/data/zvar/tempfiles/logs/";
    int chunkNum = 0;

    std::string hdf_filename = writeInputFilePath(field, ccd, quad, filter);

    //FPW parameters
    int N_samples = 4;
    int* N_bins = new int[N_samples]{20, 10, 5, 4};

    //load metadata
    unsigned int NEXP, NSRC;
    std::vector<float> time, fgrid;
    string input_filename = writeMetaDataFilePath(field, ccd, quad, filter);
    readHDF5File(input_filename, NEXP, NSRC, time, fgrid);
    int N_freqs = fgrid.size();

    int status;
    unsigned long NCHUNK = 100;
    unsigned long NSRC_PER_CHUNK = 1 + ((NSRC-1) / NCHUNK);
    unsigned long objstart, objnum;
    cout << NSRC_PER_CHUNK << endl;

    std::vector<float> flux(NSRC_PER_CHUNK * NEXP, 0.0f);
    std::vector<float> fluxErr(NSRC_PER_CHUNK * NEXP, 1.0f);
    std::vector<unsigned short> ratioValid(NSRC_PER_CHUNK, 0);
    std::vector<unsigned long> psids(NSRC_PER_CHUNK, 0);
    std::vector<float> ra(NSRC_PER_CHUNK, 0.0f);
    std::vector<float> dec(NSRC_PER_CHUNK, 0.0f);

    auto start_loadchunk = chrono::high_resolution_clock::now(); //Time loading chunk
    objstart = chunkNum * NSRC_PER_CHUNK;

    if (chunkNum < NCHUNK-1) {
        objnum = NSRC_PER_CHUNK;
        }
    else {
        objnum = NSRC - (chunkNum * NSRC_PER_CHUNK);
        }
        logMessage(logpath, "Chunk " + to_string(chunkNum+1) + ": Starting at source " + to_string(objstart) + ", loading in " + to_string(objnum) + " sources"
                    + " with " + to_string(NEXP) + " exposures each");

        auto start_loadflux = chrono::high_resolution_clock::now(); //Time loading flux values
        // Load the flux values
        status = get_sourcedata_chunk(hdf_filename, flux.data(), fluxErr.data(), ratioValid.data(), objstart, objnum, NEXP, NEXP);
        check_status(status);

        // Load the psids
        status = get_sources_chunk(hdf_filename, psids.data(), ra.data(), dec.data(), objstart, objnum);

        auto stop_loadflux = chrono::high_resolution_clock::now();
        auto duration_loadflux = chrono::duration_cast<chrono::milliseconds>(stop_loadflux - start_loadflux);
        cout << "Time to load flux values: " << duration_loadflux.count() << " milliseconds" << endl;


        // Remove baseline from flux values
        auto start_removebaseline = chrono::high_resolution_clock::now(); //Time removing baseline
        vector<float> fluxNoBaseline = removeBaseline(time.data(), flux.data(), fluxErr.data(), NEXP, NSRC_PER_CHUNK, 15);
        auto stop_removebaseline = chrono::high_resolution_clock::now();
        auto duration_removebaseline = chrono::duration_cast<chrono::milliseconds>(stop_removebaseline - start_removebaseline);
        cout << "Time to remove baseline: " << duration_removebaseline.count() << " milliseconds" << endl;

        auto start_writechunk = chrono::high_resolution_clock::now(); //Time writing chunk
        // Write the output file
        std::string output_filename = writePreProcessChunkFilePath(chunkNum, field, ccd, quad, filter);
        writePreProcessOutputFile(output_filename, psids, ratioValid, ra, dec, fluxNoBaseline, fluxErr);

        auto stop_writechunk = chrono::high_resolution_clock::now();
        auto duration_writechunk = chrono::duration_cast<chrono::milliseconds>(stop_writechunk - start_writechunk);
        cout << "Time to write chunk: " << duration_writechunk.count() << " milliseconds" << endl;

        // logMessage(logpath, "Time to remove baseline: " + to_string(duration_removebaseline.count()) + " milliseconds");

        
    return 0;
}