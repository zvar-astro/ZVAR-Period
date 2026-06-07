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
        cerr << "Usage: " << argv[0] << " <path_to_matchfile> <path_to_output> <path_to_logfile> <device>" << endl;
        return 1;
    }

    // Print help/usage if requested
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " <path_to_matchfile> <path_to_output> <device> [options]\n";
            std::cout << "\nRequired arguments:\n";
            std::cout << "  <path_to_matchfile>   Path to input HDF5 matchfile\n";
            std::cout << "  <path_to_output>      Path to output directory\n";
            std::cout << "  <path_to_logfile>     Path to log file\n";
            std::cout << "  <device>              CUDA device number (integer)\n";
            std::cout << "\nOptional arguments:\n";
            std::cout << "  --seasonal <bool>         Seasonal mode (true/false or 1/0) (default: false)\n";
            std::cout << "  --deep_drilling <bool>    Include deep drilling (true/false or 1/0) (default: false)\n";
            std::cout << "  --remove_baseline <bool>  Remove baseline (true/false or 1/0) (default: true)\n";
            std::cout << "  --clip_pgram <bool>       Run two-pass sigma clip on pgram (true/false or 1/0) (default: false)\n";
            std::cout << "  --filter_freq <bool>      Filter sidereal harmonics out of the frequency grid (true/false or 1/0) (default: true)\n";
            std::cout << "  --filter_freq_file <string> Path to csv containing sidereal harmonics to filter. If none, a default is applied. (default: none)\n";
            std::cout << "  --f_min <float>           Minimum frequency in days^-1 (default: 1.0/10.0)\n";
            std::cout << "  --f_max <float>           Maximum frequency in days^-1 (default: 1440.0/5.0)\n";
            std::cout << "  --threshold <float>       Peak detection threshold (default: 10.0)\n";
            std::cout << "  --oversample <int>        Frequency grid oversampling factor (default: 3)\n";
            std::cout << "  --upper_clip_bound optional<int>  Upper clip bound in standard deviations for light curve clipping (default: 5)\n";
            std::cout << "  --lower_clip_bound optional<int>  Lower clip bound in standard deviations for light curve clipping (default: none)\n";
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
    string log_filename = argv[3];
    int device = atoi(argv[4]);

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

    // Default values for optional booleans
    bool seasonal = false;
    bool deep_drilling = false;
    bool remove_baseline = true;
    bool clip_pgram = false;
    bool filter_freq = true;
    string filter_freq_filename = "";
    
    // Default values for f_min and f_max
    double f_min = 1.0 / 10.0; // 10 day max period
    double f_max = 1440.0 / 5.0; // 5 min min period
    float peak_threshold = 10.0; // default peak detection S/N threshold
    int oversample = 3; // default frequency grid oversampling factor

    // Default values for clipping bounds
    std::optional<int> upper_clip_bound = 5; // Default: 5 STD upper clip bound
    std::optional<int> lower_clip_bound = std::nullopt; // Default: No lower clip

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
        } else if (arg == "--clip_pgram" || arg == "-c") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            clip_pgram = parse_bool_arg(val, arg);
        } else if (arg == "--filter_freq" || arg == "-ff") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            filter_freq = parse_bool_arg(val, arg);
        } else if (arg == "--filter_freq_file" || arg == "-fff") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            // Validate that the file exists
            std::ifstream file_check(val);
            if (!file_check.good()) {
                std::cerr << "Filter frequency file does not exist: " << val << std::endl;
                exit(1);
            }
            filter_freq_filename = val;
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
        } else if (arg == "--oversample" || arg == "-o") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            try {
                oversample = std::stoi(val);
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid value for oversample: " << val << std::endl;
                exit(1);
            }
        } else if (arg == "--threshold" || arg == "-p") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            try {
                peak_threshold = std::stof(val);
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid value for peak_threshold: " << val << std::endl;
                exit(1);
            }
        } else if (arg == "--upper_clip_bound" || arg == "-ucb") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            if (val == "none" || val == "None") {
                upper_clip_bound = std::nullopt;
            } else {
                try {
                    upper_clip_bound = std::stoi(val);
                } catch (const std::invalid_argument&) {
                    std::cerr << "Invalid value for upper_clip_bound: " << val << std::endl;
                    exit(1);
                }
            }
        } else if (arg == "--lower_clip_bound" || arg == "-lcb") {
            if (val.empty()) {
                if (i + 1 < argc) {
                    val = argv[++i];
                } else {
                    std::cerr << "Missing value for " << arg << std::endl;
                    exit(1);
                }
            }
            if (val == "none" || val == "None") {
                lower_clip_bound = std::nullopt;
            } else {
                try {
                    lower_clip_bound = std::stoi(val);
                } catch (const std::invalid_argument&) {
                    std::cerr << "Invalid value for lower_clip_bound: " << val << std::endl;
                    exit(1);
                }
            }
        }
        
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            exit(1);
        }
    }

    std::string id_field_name = "gaia_id";
    std::string ra_field_name = "ra";
    std::string dec_field_name = "decl";
    std::string time_field_name = "bjd";
    std::string flux_field_name = "flux";
    std::string fluxErr_field_name = "flux_err";

    // Run the file with the provided arguments
    //create an external parameters struct
    ProcessingExternalParams params(
        device,
        1, // num_omp_threads
        0.5f, // percent_indices
        hdf_filename,
        log_filename,
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
        clip_pgram,
        filter_freq,
        filter_freq_filename,
        f_min,
        f_max,
        oversample, // freq_oversampling
        upper_clip_bound,
        lower_clip_bound,
        15, // model_id
        {20, 10, 5}, // N_bins
        300, // peak_distance
        peak_threshold
    );

    process(params);

    return 0;
}