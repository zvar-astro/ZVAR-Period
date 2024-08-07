// Utility functions for opening and reading HDF5 files
//
// By: Zach Vanderbosch

#include <tuple>
#include <string>
#include <vector>
#include <hdf5.h>

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

#endif
