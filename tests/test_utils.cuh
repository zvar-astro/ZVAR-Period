#include "../src/processing.cuh"
#include "../src/periodograms/FPW/wrappers.cuh"
#include "../src/periodograms/FPW/kernels.cuh"
#include "../src/statistics/peak_utils.h"
#include "../src/statistics/polyfit.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // For atoi
#include <chrono>  // For timing
#include <stdexcept>
#include <execinfo.h>
#include <ostream>
#include <random>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#ifndef TEST_UTILS_CUH
#define TEST_UTILS_CUH

// Function to read a CSV file into a 2D vector
std::vector<std::vector<double>> readCSV(const std::string& filePath);


// Function to extract a column from a 2D vector
std::vector<double> extractColumn(const std::vector<std::vector<double>>& data, size_t colIndex);

// Function to compute the mean and standard deviation of a vector
std::pair<float, float> computeMeanAndStdDev(const std::vector<float>& data);

// Helper function: Partition the array around a pivot
size_t partitionArray(std::vector<float>& data, size_t left, size_t right, size_t pivotIndex);

// Helper function: Quickselect to find the k-th smallest element
float quickselect(std::vector<float>& data, size_t k);

// Function to compute the median and MAD using Quickselect
std::pair<float, float> computeMedianAndMAD(std::vector<float>& data);

// Function to convert a vector to a string for debugging
std::string vectorToString(const std::vector<float>& vec);

void processTEST(const ProcessingExternalParams& params);

#endif // TEST_UTILS_CUH