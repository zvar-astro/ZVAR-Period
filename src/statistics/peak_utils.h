// Utility functions for opening and reading HDF5 files
//
// By: Zach Vanderbosch

#include <tuple>
#include <string>
#include <vector>
#include <hdf5.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#ifndef PEAK_UTILS_H_
#define PEAK_UTILS_H_


// Function to calculate mean of vector
float mean(
  std::vector<float> // v
);

// Function to calculate standard deviation of vector
float stdev(
  std::vector<float> // v
);

float median(
    float*, // v
    size_t // N
);

float mad(
    float*, // v
    size_t, // N
    float // median
);

__global__ void CudaComputeMedianandMAD(
    float* data, // data
    float* medians, // median outputs
    float* mads, // mad outputs
    int N_dat, // N_dat
    int N_curves // N_curves
);

// Function to return both mean and stdev of vector
std::tuple<float, float> pg_stats(
  std::vector<float> // v
);

// Function to identify and flag terrestrial frequencies
std::vector<int> flagTerrestrialFreq(
  float*, // frequencies
  int     // Nfreq
);

// A peak finding function based on scipy.signal.find_peaks
std::vector<int> find_peaks(
  const std::vector<float>&, // x
  const int,                // Ntop
  const int,                // distance
  const float,              // mean
  const float,              // stdev
  const float               // theshold
);

__global__ void cudaFindPeaks(
    float* data, //input data (N_dat * N_curves)
    float* median, //medians (N_curves)
    float* mad, //median absolute deviations (N_curves)
    int* peakIndices, //output for peak indices (N_curves * Ntop)
    float* peakValues, //output for peak significances (N_curves * Ntop)
    int N_dat, //N_dat
    int N_curves, //N_curves
    int distance, //distance between peaks
    float threshold //minimum significance for a peak to be considered
);

#endif
