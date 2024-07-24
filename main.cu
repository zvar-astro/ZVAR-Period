#include "runFile.cuh"
#include <iostream>
#include <string>
#include <cstdlib> // For atoi
#include <chrono>  // For timing

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <field> <ccd> <quad> <filter> <device>" << endl;
        return 1;
    }

    // Parse command-line arguments
    int field = atoi(argv[1]);
    int ccd = atoi(argv[2]);
    int quad = atoi(argv[3]);
    string filter = argv[4];
    int device = atoi(argv[5]);

    if (device < 0 || device >= 4) {
        cerr << "Invalid device number. Please provide a number between 0 and 3." << endl;
        return 1;
    }

    // Set the device to use
    cudaSetDevice(device);

    // Tell the device to use the maximum amount of shared memory.
    cudaFuncSetAttribute(cudaDeltaChi2, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(cudaDeltaChi2Multibin, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);

    // Start the timer
    auto start = high_resolution_clock::now();

    // Run the file with the provided arguments
    runFile(field, ccd, quad, filter);

    // Stop the timer
    auto end = high_resolution_clock::now();

    // Calculate the duration
    auto duration = duration_cast<minutes>(end - start).count();

    // Print the duration
    cout << "Execution time: " << duration << " minutes" << endl;

    return 0;
}