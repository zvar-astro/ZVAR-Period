#include "polyfit.cuh"

// float* removeBaseline(float* times, float* y, float* dy, float* medians, float* mads, int N_dat, int N_curves, int modelID) {

//     int N_parameters = 0;
//     switch (modelID) {
//         case 13: // Quadratic model
//             N_parameters = 3;
//             break;
//         case 14: // Cubic model
//             N_parameters = 4;
//             break;
//         case 15: // 7 degree model
//             N_parameters = 8;
//             break;
//         case 16: // 15 degree model
//             N_parameters = 16;
//             break;
//         default:
//             throw std::invalid_argument("Invalid model ID");
//             break;
//     }

//     //divide all times by the maximum time so the reinterpret_cast works
//     float* times_fractional = new float[N_dat];
//     float t_max = times[N_dat - 1];
//     for (int i = 0; i < N_dat; ++i) {
//         times_fractional[i] = times[i] / t_max;
//     }

//     //convert dy to weights
//     float* weights = new float[N_dat * N_curves];
//     for (int i = 0; i < N_curves; ++i) {
//         for (int j = 0; j < N_dat; ++j) {
//             weights[i * N_dat + j] = 1.0 / (dy[i * N_dat + j] * dy[i * N_dat + j]);
//         }
//     }

//     // sigma clip with median and MAD
//     // for (int i = 0; i < N_curves; ++i) {
//     //     if (medians[i] != 0 && mads[i] != 0) {
//     //         for (int j = 0; j < N_dat; ++j) {
//     //             // sigma = 1.483 * MAD. We use 5 sigma clipping and remove points above and below the median here.
//     //             if (std::abs(y[i * N_dat + j] - medians[i]) > 10 * 1.483 * mads[i]) {
//     //                 std::cout << "Removing point " << j << " from curve " << i << " with value " << y[i * N_dat + j] << " and median " << medians[i] << " and MAD " << mads[i] << std::endl;
//     //                 weights[i * N_dat + j] = 0.0f;
//     //             }
//     //         }
//     //     }
//     // }

//     // Define other Gpufit parameters
//     float tolerance = 1e-4;
//     int max_n_iterations = 1000; // we care more about speed than accuracy, so n_iter is low.
//     int estimator_id = LSE;  // Least squares estimator
//     float* initial_parameters = new float[N_curves * N_parameters];
//     std::fill_n(initial_parameters, N_curves * N_parameters, 0.0f); // our initial guess is that the baseline is 0
//     int* parameters_to_fit = new int[N_curves * N_parameters];
//     std::fill_n(parameters_to_fit, N_curves * N_parameters, 1);
//     char* times_input = reinterpret_cast<char*>(times_fractional); // gpufit requires a char* for the times input
//     size_t user_info_size = sizeof(REAL) * N_dat;

//     // Allocate memory for Gpufit output
//     float* output_parameters = new float[N_curves * N_parameters];
//     int* output_states = new int[N_curves];
//     float* output_chi_squares = new float[N_curves];
//     int* output_n_iterations = new int[N_curves];

//     // Call Gpufit
//     int status = gpufit(N_curves, N_dat, y, weights, modelID, initial_parameters, tolerance, max_n_iterations, parameters_to_fit, estimator_id, 
//                         user_info_size, times_input, output_parameters, output_states, output_chi_squares, output_n_iterations);

//     // Check status
//     if (status != ReturnState::OK) {
//         std::cout << "Error: Gpufit failed with status " << status << std::endl;
//         const char* error_message = gpufit_get_last_error();
//         std::cerr << "GPUfit error: " << error_message << std::endl;
//     }

//     // subtract the model y from the data y
//     float* y_no_baseline = new float[N_dat * N_curves];
//     for (int i = 0; i < N_curves; ++i) {
//         if (output_states[i] == 0) { // Only subtract the model if the fit was successful
//             for (int j = 0; j < N_dat; ++j) {
//                 float y_model = 0;
//                 for (int k = 0; k < N_parameters; ++k) {
//                     y_model += output_parameters[i * N_parameters + k] * pow(times[j], N_parameters - 1 - k) / pow(t_max, N_parameters - 1 - k); // Rescale model parameters by t_max since we divided times by t_max
//                 }
//                 y_no_baseline[i * N_dat + j] = y[i * N_dat + j] - y_model;
//             }
//         }
//         else {
//             for (int j = 0; j < N_dat; ++j) {
//                 y_no_baseline[i * N_dat + j] = y[i * N_dat + j];
//             }
//         }
//     }

//     // Free allocated memory
//     delete[] times_fractional;
//     delete[] weights;
//     delete[] initial_parameters;
//     delete[] parameters_to_fit;
//     delete[] output_parameters;
//     delete[] output_states;
//     delete[] output_chi_squares;
//     delete[] output_n_iterations;

//     return y_no_baseline;
// }

float* removeBaseline(float* times, float* y, float* dy, int N_dat, int N_curves, int modelID) {

    int N_parameters = 0;
    switch (modelID) {
        case 13: // Quadratic model
            N_parameters = 3;
            break;
        case 14: // Cubic model
            N_parameters = 4;
            break;
        case 15: // 7 degree model
            N_parameters = 8;
            break;
        case 16: // 15 degree model
            N_parameters = 16;
            break;
        default:
            throw std::invalid_argument("Invalid model ID");
            break;
    }

    //divide all times by the maximum time so the reinterpret_cast works
    float* times_fractional = new float[N_dat];
    float t_max = times[N_dat - 1];
    for (int i = 0; i < N_dat; ++i) {
        times_fractional[i] = times[i] / t_max;
    }

    //convert dy to weights
    float* weights = new float[N_dat * N_curves];
    for (int i = 0; i < N_curves; ++i) {
        for (int j = 0; j < N_dat; ++j) {
            weights[i * N_dat + j] = 1.0 / (dy[i * N_dat + j] * dy[i * N_dat + j]);
        }
    }

    // Define other Gpufit parameters
    float tolerance = 1e-4;
    int max_n_iterations = 1000; // we care more about speed than accuracy, so n_iter is low.
    int estimator_id = LSE;  // Least squares estimator
    float* initial_parameters = new float[N_curves * N_parameters];
    std::fill_n(initial_parameters, N_curves * N_parameters, 0.0f); // our initial guess is that the baseline is 0
    int* parameters_to_fit = new int[N_curves * N_parameters];
    std::fill_n(parameters_to_fit, N_curves * N_parameters, 1);
    char* times_input = reinterpret_cast<char*>(times_fractional); // gpufit requires a char* for the times input
    size_t user_info_size = sizeof(REAL) * N_dat;

    // Allocate memory for Gpufit output
    float* output_parameters = new float[N_curves * N_parameters];
    int* output_states = new int[N_curves];
    float* output_chi_squares = new float[N_curves];
    int* output_n_iterations = new int[N_curves];

    // Call Gpufit
    int status = gpufit(N_curves, N_dat, y, weights, modelID, initial_parameters, tolerance, max_n_iterations, parameters_to_fit, estimator_id, 
                        user_info_size, times_input, output_parameters, output_states, output_chi_squares, output_n_iterations);

    // Check status
    if (status != ReturnState::OK) {
        std::cout << "Error: Gpufit failed with status " << status << std::endl;
        const char* error_message = gpufit_get_last_error();
        std::cerr << "GPUfit error: " << error_message << std::endl;
    }

    // subtract the model y from the data y
    float* y_no_baseline = new float[N_dat * N_curves];
    for (int i = 0; i < N_curves; ++i) {
        if (output_states[i] == 0) { // Only subtract the model if the fit was successful
            for (int j = 0; j < N_dat; ++j) {
                float y_model = 0;
                for (int k = 0; k < N_parameters; ++k) {
                    y_model += output_parameters[i * N_parameters + k] * pow(times[j], N_parameters - 1 - k) / pow(t_max, N_parameters - 1 - k); // Rescale model parameters by t_max since we divided times by t_max
                }
                y_no_baseline[i * N_dat + j] = y[i * N_dat + j] - y_model;
            }
        }
        else {
            for (int j = 0; j < N_dat; ++j) {
                y_no_baseline[i * N_dat + j] = y[i * N_dat + j];
            }
        }
    }

    // Free allocated memory
    delete[] times_fractional;
    delete[] weights;
    delete[] initial_parameters;
    delete[] parameters_to_fit;
    delete[] output_parameters;
    delete[] output_states;
    delete[] output_chi_squares;
    delete[] output_n_iterations;

    return y_no_baseline;
}