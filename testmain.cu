#include "periodograms/FPW/wrappers.cuh"
#include "statistics/polyfit.cuh"

#include <iostream>
#include <random>
#include <ctime>
#include <gpufit.h>

#include <chrono>
#include <cuda_runtime.h> 

using namespace std;


int main() {

    // Make fake data (purely polynomial signal)
    std::mt19937 gen(static_cast<unsigned int>(std::time(0))); // Seed the random number generator

    std::uniform_real_distribution<> dis_uniform(0.0, 10.0); // Uniform distribution between 0 and 10
    std::uniform_real_distribution<> dis_coeff(0.0, 3.0); // Uniform distribution between 0 and 3
    std::normal_distribution<> dis_normal(0.0, 1.0); // Gaussian distribution with mean 0 and variance 1

    int N_dat = 1000;
    int N_curves = 1000;

    float* times = new float[N_dat];

    for (int i = 0; i < N_dat; ++i) {
        times[i] = dis_uniform(gen); // Generate a random number in [0, 10]
    }

    // Generate a quadratic signal with random coefficients between 0 and 3 and add gaussian noise
    float* y = new float[N_dat * N_curves];
    float* dy = new float[N_dat * N_curves];

    float* a = new float[N_curves];
    float* b = new float[N_curves];
    float* c = new float[N_curves];

    for (int i = 0; i < N_curves; ++i) {
        a[i] = dis_coeff(gen);
        b[i] = dis_coeff(gen);
        c[i] = dis_coeff(gen);
    }

    //generate signal
    for (int i = 0; i < N_curves; ++i) {
        for (int j = 0; j < N_dat; ++j) {
            // y[i * N_dat + j] = a[i] * times[j] * times[j] + b[i] * times[j] + c[i] + dis_normal(gen);
            y[i * N_dat + j] = a[i] * times[j] * times[j] + b[i] * times[j] + c[i];
        }
    }

    //populate dy with 1.0
    for (int i = 0; i < N_dat * N_curves; ++i) {
        dy[i] = 1.0;
    }

    // Define the model ID for the polynomial model
    const int POLYNOMIAL_MODEL = 15;
    const int N_params = 8;

    // Define other Gpufit parameters
    float tolerance = 1e-4;
    int max_n_iterations = 1000;
    int estimator_id = LSE;  // Least squares estimator
    float* initial_parameters = new float[N_curves * N_params];  // Allocate memory
    std::fill_n(initial_parameters, N_curves * N_params, 1.0f);  // Fill with ones
    int* parameters_to_fit = new int[N_curves * N_params];  // Allocate memory
    std::fill_n(parameters_to_fit, N_curves * N_params, 1);  // Fill with ones

    // Allocate memory for Gpufit output
    float* output_parameters = new float[N_curves * N_params];
    int* output_states = new int[N_curves];
    float* output_chi_squares = new float[N_curves];
    int* output_n_iterations = new int[N_curves];

    char* times_input = reinterpret_cast<char*>(times);

    size_t user_info_size = sizeof(REAL) * N_dat;

    // Get the start time before the function call
    auto start = std::chrono::high_resolution_clock::now();

    // Call Gpufit
    int status = gpufit(N_curves, N_dat, y, dy, POLYNOMIAL_MODEL, initial_parameters, tolerance, max_n_iterations, parameters_to_fit, estimator_id, 
                        user_info_size, times_input, output_parameters, output_states, output_chi_squares, output_n_iterations);

    // Synchronize the CPU and GPU
    cudaDeviceSynchronize();

    // Get the end time after the function call
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print the duration in milliseconds
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end-start) / static_cast<double>(N_curves);
    std::cout << "Time to execute gpufit: " << diff.count() << " ms\n";

    // Check status
    if (status != ReturnState::OK) {
        std::cout << "Error: Gpufit failed with status " << status << std::endl;
    }

    // Print the fitted parameters and true parameters
    for (int i = 0; i < 10; ++i) {
        // std::cout << "Curve " << i << ": a_fit = " << output_parameters[i * 8 + 5] << ", b_fit = " << output_parameters[i * 8 + 6] << ", c_fit = " << output_parameters[i * 8 + 7] << std::endl;
        std::cout << "Curve " << i << ": a_fit = " << output_parameters[i * N_params + 5] << ", b_fit = " << output_parameters[i * N_params + 6] << ", c_fit = " << output_parameters[i * N_params + 7] << std::endl;
        std::cout << "Curve " << i << ": a_true = " << a[i] << ", b_true = " << b[i] << ", c_true = " << c[i] << std::endl;
    }

    // Write fake light curves to CSV
    ofstream light_curves_file("light_curves.csv");
    light_curves_file << "Curve,Time,Value\n";
    for (int i = 0; i < N_curves; ++i) {
        for (int j = 0; j < N_dat; ++j) {
            light_curves_file << i << "," << times[j] << "," << y[i * N_dat + j] << "\n";
        }
    }
    light_curves_file.close();

    // Write fitted parameters to CSV
    ofstream fitted_parameters_file("fitted_parameters.csv");
    for (int i = 0; i < N_params; ++i) {
        fitted_parameters_file << i;
        for (int j = 0; j < N_params; ++j) {
            fitted_parameters_file << "," << output_parameters[i * N_params + j];
        }
        fitted_parameters_file << "\n";
    }

    // Free allocated memory
    delete[] initial_parameters;
    delete[] parameters_to_fit;
    delete[] output_parameters;
    delete[] output_states;
    delete[] output_chi_squares;
    delete[] output_n_iterations;

    // report found baseline
    delete[] times;
    delete[] y;
    delete[] dy;
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}