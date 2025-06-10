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
#include <math_constants.h>

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

__global__ void cudaComputeSignificance(float* data, float* median, float* mad, int N_dat, int N_curves);

__device__ bool find_next_peak(
    float* curve_data, size_t& sample_idx, const size_t last_peak_check,
    int& peak_idx, float& peak_value
);

__device__ void insert_peak(
    int* top_peak_indices, float* top_peak_values, int N_peaks,
    int candidate_peak_idx, float candidate_peak_value
);

__global__ void cudaFindPeaks(
    float* data, //input data (N_dat * N_curves)
    int* peakIndices, //output for peak indices (N_curves * Ntop)
    float* peakValues, //output for peak significances (N_curves * Ntop)
    int N_dat, //N_dat
    int N_curves, //N_curves
    int distance, //distance between peaks
    float threshold //minimum significance for a peak to be considered
);

__global__ void cudaFindPeaksChunk(
    float* data,                    // [N_curves * N_dat] Input: all curve data, row-major. NOTE: The first two values are the
                                    // last two values from the previous chunk.
    float* last_chunk_data,         // [N_curves * 2] Input: last two values from the previous chunk for each curve. If this is the first chunk, they are set to -1.0f.
    int* peakIndices,               // [N_curves * N_peaks] Output: indices of top peaks per curve
    float* peakValues,              // [N_curves * N_peaks] Output: values of top peaks per curve
    int N_dat,                      // Number of data points per curve for this chunk
    int N_curves,                   // Number of curves
    int distance,                   // Minimum distance between peaks
    float threshold,                // Minimum S/N threshold for a peak
    int chunk_start_global_idx,     // Start index (inclusive) of this chunk in the curve (N_dat * chunk_num)
    int* state_candidate_idx,       // [N_curves] Persistent: candidate peak index (global)
    float* state_candidate_val,     // [N_curves] Persistent: candidate peak value (global)
    int* state_distLastPeak,        // [N_curves] Persistent: distance since last accepted/candidate peak
    float* state_currentThreshold  // [N_curves] Persistent: current dynamic threshold
);

#endif
