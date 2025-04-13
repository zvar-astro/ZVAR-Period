// #define CATCH_CONFIG_MAIN  // This generates the main() function
#include "../src/processing.cuh"
#include "../src/periodograms/FPW/wrappers.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // For atoi
#include <chrono>  // For timing
#include <stdexcept>
#include <execinfo.h>
#include <ostream>
#include <catch2/catch_all.hpp>

// Function to read a CSV file into a 2D vector
std::vector<std::vector<double>> readCSV(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to extract a column from a 2D vector
std::vector<double> extractColumn(const std::vector<std::vector<double>>& data, size_t colIndex) {
    std::vector<double> column;
    for (const auto& row : data) {
        if (colIndex < row.size()) {
            column.push_back(row[colIndex]);
        } else {
            throw std::out_of_range("Column index out of range");
        }
    }
    return column;
}

//Function to compute the mean and standard deviation of a vector
std::pair<float, float> computeMeanAndStdDev(const std::vector<float>& data) {
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    float sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f);
    float stdev = std::sqrt(sq_sum / data.size() - mean * mean);
    return {mean, stdev};
}

TEST_CASE("FPW GPU vs CPU 20 Bin Periodogram Comparison", "[FPW]") {
    try {
        // Read the data file
        std::string dataFile = "../tests/data/tydy.csv";
        std::vector<std::vector<double>> data = readCSV(dataFile);

        // Extract columns
        std::vector<double> times = extractColumn(data, 0);
        std::vector<double> flux = extractColumn(data, 1);
        std::vector<double> fluxErr = extractColumn(data, 2);

        // Read the frequency grid file
        std::string fgridFile = "../tests/data/fgrid.csv";
        std::vector<std::vector<double>> fgridData = readCSV(fgridFile);
        std::vector<double> fgrid = extractColumn(fgridData, 0);

        // Convert to floats
        std::vector<float> timesFloat(times.size());
        std::vector<float> fluxFloat(flux.size());
        std::vector<float> fluxErrFloat(fluxErr.size());
        std::vector<float> fgridFloat(fgrid.size());
        std::transform(times.begin(), times.end(), timesFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(flux.begin(), flux.end(), fluxFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(fluxErr.begin(), fluxErr.end(), fluxErrFloat.begin(), [](double val) { return static_cast<float>(val); });
        std::transform(fgrid.begin(), fgrid.end(), fgridFloat.begin(), [](double val) { return static_cast<float>(val); });

        // Convert the float vectors to pointer arrays
        float* timesPtr = timesFloat.data();
        float* fluxPtr = fluxFloat.data();
        float* fluxErrPtr = fluxErrFloat.data();
        float* fgridPtr = fgridFloat.data();

        // Extract constants
        int N_samples = 3;
        int* N_bins = new int[N_samples]{20, 10, 5};
        int N_dat = timesFloat.size();
        unsigned long N_curves = 1;
        unsigned long N_freqs = fgridFloat.size();
        unsigned long N_freqs_batch = 100000;
        unsigned long N_curves_batch = 1;

        // Run FPW on the data
        std::vector<float> periodograms = cudaRunFPW(timesPtr, fluxPtr, fluxErrPtr, fgridPtr,
                                                     N_dat, N_curves, N_freqs, N_bins, N_freqs_batch, N_curves_batch, N_samples);

        std::vector<float> gpuPeriodogram(periodograms.begin(), periodograms.begin() + N_freqs);

        // Load in the expected periodogram
        std::string expectedFile = "../tests/data/CPU_periodogram.csv";
        std::vector<std::vector<double>> expectedData = readCSV(expectedFile);
        std::vector<double> cpuPeriodogram = extractColumn(expectedData, 0);
        // Convert to float
        std::vector<float> cpuPeriodogramFloat(cpuPeriodogram.size());
        std::transform(cpuPeriodogram.begin(), cpuPeriodogram.end(), cpuPeriodogramFloat.begin(),
                       [](double val) { return static_cast<float>(val); });

        // Element-wise subtract the two periodograms
        std::vector<float> diff(gpuPeriodogram.size());
        for (size_t i = 0; i < gpuPeriodogram.size(); ++i) {
            diff[i] = gpuPeriodogram[i] - cpuPeriodogramFloat[i];
        }

        // Compute the mean and standard deviation of the CPU periodogram
        auto [meanCPU, stdevCPU] = computeMeanAndStdDev(cpuPeriodogramFloat);

        // Compute the mean and standard deviation of the difference
        auto [meanDiff, stdevDiff] = computeMeanAndStdDev(diff);

        // Assertions
        REQUIRE(meanDiff < 1.0f); // Mean of the difference should be less than 1
        REQUIRE(stdevDiff < stdevCPU); // Standard deviation of the difference should be less than that of the CPU periodogram

        // Clean up
        delete[] N_bins;

    } catch (const std::exception& e) {
        FAIL("Exception occurred: " << e.what());
    }
}