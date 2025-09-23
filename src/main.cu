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
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <path_to_matchfile> <path_to_output> <device> <datatype> (zvar or zforce)" << endl;
        return 1;
    }

    // Print help/usage if requested
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " <path_to_matchfile> <path_to_output> <device> <datatype> [options]\n";
            std::cout << "\nRequired arguments:\n";
            std::cout << "  <path_to_matchfile>   Path to input HDF5 matchfile\n";
            std::cout << "  <path_to_output>      Path to output directory\n";
            std::cout << "  <device>              CUDA device number (integer)\n";
            std::cout << "  <datatype>            'zvar' or 'zforce'\n";
            std::cout << "\nOptional arguments:\n";
            std::cout << "  --seasonal <bool>         Seasonal mode (true/false or 1/0)\n";
            std::cout << "  --deep_drilling <bool>    Include deep drilling (true/false or 1/0)\n";
            std::cout << "  --remove_baseline <bool>  Remove baseline (true/false or 1/0)\n";
            std::cout << "  --f_min <float>           Minimum frequency in days^-1 (default: 1.0/10.0)\n";
            std::cout << "  --f_max <float>           Maximum frequency in days^-1 (default: 1440.0/5.0)\n";
            std::cout << "  -s, -d, -b, -min, -max    Short forms for above options\n";
            std::cout << "  -h, --help                Show this help message\n";
            std::cout << std::endl;
            return 0;
        }
    }

    std::set_terminate(handler);

    // Parse command-line arguments
    string hdf_filename = argv[1];
    string output_dir = argv[2];
    int device = atoi(argv[3]);

    // get the list of available cuda devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout << "Number of available CUDA devices: " << deviceCount << endl;

    if (device < 0 || device >= deviceCount) {
        cerr << "Invalid device number. Please provide a number between 0 and " << deviceCount - 1 << "." << endl;
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
    cudaFuncSetAttribute(cudaDeltaChi2Multibin, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMem);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "unexpected CUDA error (main after allocation): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // get the datatype from the command line argument
    string datatype = argv[4];
    if (datatype != "zvar" && datatype != "zforce") {
        cerr << "Invalid datatype. Please provide either 'zvar' or 'zforce'." << endl;
        return 1;
    }
    cout << "Using datatype: " << datatype << endl;

    // Default values for optional booleans
    bool seasonal = false;
    bool deep_drilling = false;
    bool remove_baseline = true;

    // Default values for f_min and f_max
    double f_min = 1.0 / 10.0; // 10 day max period
    double f_max = 1440.0 / 5.0; // 5 min min period
    // double f_max = 1.0 / 10.0, // 10 day min period
    // double f_min = 1.0 / (3.0 * 365), // 3 year max period

    //Helper function for parsing optional arguments
    auto parse_bool_arg = [](const std::string& val, const std::string& arg_name) -> bool {
        if (val == "true" || val == "1") return true;
        if (val == "false" || val == "0") return false;
        std::cerr << "Invalid value for " << arg_name << ": " << val << ". Must be true/false or 1/0." << std::endl;
        exit(1);
    };

    // Parse the optional arguments
    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        std::string val;
        // Check for --option=value format
        size_t eq_pos = arg.find('=');
        if (eq_pos != std::string::npos) {
            val = arg.substr(eq_pos + 1);
            arg = arg.substr(0, eq_pos);
        }
        if (arg == "--seasonal" || arg == "-s") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            seasonal = parse_bool_arg(val, arg);
        } else if (arg == "--deep_drilling" || arg == "-d") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            deep_drilling = parse_bool_arg(val, arg);
        } else if (arg == "--remove_baseline" || arg == "-b") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            remove_baseline = parse_bool_arg(val, arg);
        } else if (arg == "--f_min" || arg == "-min") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            try {
                f_min = std::stod(val);
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid value for f_min: " << val << std::endl;
                exit(1);
            }
        } else if (arg == "--f_max" || arg == "-max") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            try {
                f_max = std::stod(val);
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid value for f_max: " << val << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            exit(1);
        }
    }

    std::string id_field_name = "psid";
    std::string ra_field_name = "ra";
    std::string dec_field_name = "dec";
    std::string time_field_name = "bjd";
    std::string flux_field_name = "flux";
    std::string fluxErr_field_name = "fluxerr";

    std::string logpath = "/data/swhitebook/zvar_testing/logs";

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
        seasonal,
        deep_drilling,
        remove_baseline,
        f_min,
        f_max
    );

    process(params);

    return 0;
}