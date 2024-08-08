#include "polyfit.cuh"

vector<float> removeBaseline(float* times, float* y, float* dy, int N_dat, int N_curves, int modelID) {

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

    // Define other Gpufit parameters
    float tolerance = 1e-4;
    int max_n_iterations = 1000; // we care more about speed than accuracy, so n_iter is low.
    int estimator_id = LSE;  // Least squares estimator
    float* initial_parameters = new float[N_curves * N_parameters];
    std::fill_n(initial_parameters, N_curves * N_parameters, 0.0f);
    int* parameters_to_fit = new int[N_curves * N_parameters];
    std::fill_n(parameters_to_fit, N_curves * N_parameters, 1);

    // Allocate memory for Gpufit output
    float* output_parameters = new float[N_curves * N_parameters];
    int* output_states = new int[N_curves];
    float* output_chi_squares = new float[N_curves];
    int* output_n_iterations = new int[N_curves];

    char* times_input = reinterpret_cast<char*>(times);

    size_t user_info_size = sizeof(REAL) * N_dat;

    // Call Gpufit
    int status = gpufit(N_curves, N_dat, y, dy, modelID, initial_parameters, tolerance, max_n_iterations, parameters_to_fit, estimator_id, 
                        user_info_size, times_input, output_parameters, output_states, output_chi_squares, output_n_iterations);

    // Check status
    if (status != ReturnState::OK) {
        std::cout << "Error: Gpufit failed with status " << status << std::endl;
    }

    // subtract the model y from the data y
    vector<float> y_no_baseline(N_dat * N_curves);
    for (int i = 0; i < N_curves; ++i) {
        for (int j = 0; j < N_dat; ++j) {
            float y_model = 0;
            for (int k = 0; k < N_parameters; ++k) {
                y_model += output_parameters[i * N_parameters + k] * pow(times[j], N_parameters - 1 - k);
            }
            y_no_baseline[i * N_dat + j] = y[i * N_dat + j] - y_model;
        }
    }       

    // Free allocated memory
    delete[] initial_parameters;
    delete[] parameters_to_fit;
    delete[] output_parameters;
    delete[] output_states;
    delete[] output_chi_squares;
    delete[] output_n_iterations;

    return y_no_baseline;
}