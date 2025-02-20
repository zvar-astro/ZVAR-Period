#include "runFile.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // For atoi
#include <chrono>  // For timing

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <path_to_matchfile> <device>" << endl;
        return 1;
    }

    // Parse command-line arguments
    string hdf_filename = argv[1];
    int device = atoi(argv[2]);

    // get the list of available cuda devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout << "Number of available CUDA devices: " << deviceCount << endl;

    if (device < 0 || device >= 4) {
        cerr << "Invalid device number. Please provide a number between 0 and 3." << endl;
        return 1;
    }

    // Set the device to use
    cudaSetDevice(device);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "unexpected CUDA error (main): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Tell the device to use the maximum amount of shared memory.
    cudaFuncSetAttribute(cudaDeltaChi2, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
    cudaFuncSetAttribute(cudaDeltaChi2Multibin, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "unexpected CUDA error (main after allocation): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // check if the log directory exists, create it (and all the parent directories) if it doesn't
    string logdir = "/data/zvar/logs";
    string mkdir = "mkdir -p " + logdir;
    system(mkdir.c_str());

    string logpath = logdir + "/device_" + to_string(device) + "_log.txt";

    //check if the log file exists, create one if it doesn't
    ifstream infile(logpath);
    if (!infile.good()) {
        ofstream outfile(logpath);
        if (!outfile) {
            cerr << "Failed to create log file: " << logpath << endl;
            return 1;
        }
    }

    // Run the file with the provided arguments
    runFile(hdf_filename, device, logpath);

    return 0;
}