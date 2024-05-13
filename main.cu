#include "periodograms/FPW/wrappers.cuh"
#include "statistics/polyfit.cuh"

#include <iostream>
#include <random>
#include <ctime>

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

    float* y_clean = removeBaseline(times, y, dy, N_dat, N_curves, 13); // Quadratic model

    // print a few values
    for (int i = 0; i < 10; ++i) {
        cout << "y_clean[" << i << "] = " << y_clean[i] << endl;
    }

    // report found baseline
    delete[] times;
    delete[] y;
    delete[] dy;
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}