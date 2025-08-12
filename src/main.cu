#include "processing.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // For atoi
#include <chrono>  // For timing
#include <stdexcept>
#include <execinfo.h>

using namespace std;
using namespace std::chrono;

// Function to handle errors and print stack trace
void handler() {
    void *trace_elems[20];
    int trace_elem_count(backtrace( trace_elems, 20 ));
    char **stack_syms(backtrace_symbols( trace_elems, trace_elem_count ));
    for ( int i = 0 ; i < trace_elem_count ; ++i )
    {
        std::cout << stack_syms[i] << "\n";
    }
    free( stack_syms );

    exit(1);
}  

int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <path_to_matchfile> <device> <datatype> (zvar or zforce)" << endl;
        return 1;
    }

    std::set_terminate(handler);

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

    int maxSharedMem = 0;
    cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Tell the device to use the maximum amount of shared memory.

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "unexpected CUDA error (main after allocation): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // get the datatype from the command line argument
    string datatype = argv[3];
    if (datatype != "zvar" && datatype != "zforce") {
        cerr << "Invalid datatype. Please provide either 'zvar' or 'zforce'." << endl;
        return 1;
    }
    cout << "Using datatype: " << datatype << endl;

    // we create the field names based on the datatype
    // string id_field_name = datatype + "psid";
    // string ra_field_name = datatype + "ra";
    // string dec_field_name = datatype + "dec";
    // string time_field_name = datatype + "bjd";
    // string flux_field_name = datatype + "flux";
    // string fluxErr_field_name = datatype + "flux_err";
    // if (datatype == "zforce") {
    //     time_field_name = "mjd";
    // }

    std::string id_field_name = "psid";
    std::string ra_field_name = "ra";
    std::string dec_field_name = "dec";
    std::string time_field_name = "bjd";
    std::string flux_field_name = "flux";
    std::string fluxErr_field_name = "flux_err";
    
    // check if the log directory exists, create it (and all the parent directories) if it doesn't
    // string logdir = "/data/zvar/logs";
    // instead get the log directory from the environment variable, if not provided, use the default
    // const char* logdir = std::getenv("ZVAR_LOG_DIR");
    // std::cout << "ZVAR_LOG_DIR: " << logdir << std::endl;
    // if (logdir == nullptr) {
    //     std::cerr << "ZVAR_LOG_DIR environment variable not set. Using default: /data/zvar/logs" << std::endl;
    //     // logdir = "/data/swhitebook/zvar_testing/logs";
    //     logdir = "/data/zvar/logs";
    // }
    // string logdir_str(logdir);

    // string mkdir = "mkdir -p " + logdir_str;
    // system(mkdir.c_str());

    // string logpath = logdir_str + "/device_" + to_string(device) + "_log.txt";

    // //check if the log file exists, create one if it doesn't
    // ifstream infile(logpath);
    // if (!infile.good()) {
    //     ofstream outfile(logpath);
    //     if (!outfile) {
    //         cerr << "Failed to create log file: " << logpath << endl;
    //         return 1;
    //     }
    // }

    std::string logpath = "/data/swhitebook/zvar_testing/logs";
    string output_dir = "/data/swhitebook/zvar_testing/stats_testing";

    // Run the file with the provided arguments
    //create an external parameters struct
    ProcessingExternalParams params(
        device,
        1, // num_omp_threads
        0.5f, // percent_indices
        hdf_filename,
        logpath,
        output_dir,
        id_field_name,
        ra_field_name,
        dec_field_name,
        time_field_name,
        flux_field_name,
        fluxErr_field_name,
        false, // seasonal
        false // deep_drilling
    );

    process(params);

    return 0;
}