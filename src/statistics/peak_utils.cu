// Functions for peak finding, periodogram statistics
// and filtering of terrestrial frequencies.
//
// Author: Zach Vanderbosch (Caltech)
//  Edits: Sam Whitebook (caltech)

#include <tuple>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <chrono>
#include "peak_utils.h"


float mean(std::vector<float> v) {

    /*
    Function to calculate mean of values

    v = vector of values
    */

    float sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    float avg =  sum / v.size();

    return avg;
}


float stdev(std::vector<float> v) {

    /*
    Function to calculate standard deviation of values

    v = vector of values
    */

    const float m = mean(v);
    float accum = 0.0f;
    std::for_each (std::begin(v), std::end(v), [&](const float d) {
        accum += (d - m) * (d - m);
    });

    float stdev = sqrt(accum / (v.size()-1));
    return stdev;
}

float median(float* arr, size_t size) {
    if (size == 0) {
        throw std::invalid_argument("The input array is empty.");
    }

    size_t mid = size / 2;

    // Use nth_element to partially sort the array
    std::nth_element(arr, arr + mid, arr + size);

    if (size % 2 == 0) {
        // If even, find the other middle element and compute the average
        float mid1 = arr[mid];
        std::nth_element(arr, arr + mid - 1, arr + size);
        float mid2 = arr[mid - 1];
        return (mid1 + mid2) / 2.0f;
    } else {
        // If odd, return the middle element
        return arr[mid];
    }
}

float mad(float* arr, size_t size, float array_med) {
    if (size == 0) {
        throw std::invalid_argument("The input array is empty.");
    }

    // Create a temporary array to store the absolute differences
    float* abs_diff = new float[size];
    for (size_t i = 0; i < size; i++) {
        abs_diff[i] = fabs(arr[i] - array_med);
    }

    // Compute the median of the absolute differences
    return median(abs_diff, size);
}

// Function to sort an array using bubble sort on the GPU
__device__ void bubbleSort(float* array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                float temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

__device__ int partition(float* array, int left, int right, int pivotIndex) {
    float pivotValue = array[pivotIndex];
    // Move pivot to the end
    float temp = array[pivotIndex];
    array[pivotIndex] = array[right];
    array[right] = temp;

    int storeIndex = left;
    for (int i = left; i < right; i++) {
        if (array[i] < pivotValue) {
            // Swap array[i] and array[storeIndex]
            temp = array[i];
            array[i] = array[storeIndex];
            array[storeIndex] = temp;
            storeIndex++;
        }
    }
    // Move pivot to its final place
    temp = array[storeIndex];
    array[storeIndex] = array[right];
    array[right] = temp;

    return storeIndex;
}


__global__ void CudaComputeMedianandMAD(float* data, float* medians, float* mads, int N_dat, int N_curves) {
    /*
    Torben's Median Algorithm. This algorithm requires no intermediate arrays. 
    Computes the median and MAD of N_curves of N_data points each.

    data = pointer to the data (N_dat * N_curves)
    medians = pointer to the output array for medians (N_curves)
    mads = pointer to the output array for MADs (N_curves)
    N_dat = number of data points in each curve
    N_curves = number of curves
    */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_curves) {
        // Pointer to the start of the curve data for this thread
        float* curve_data = data + idx * N_dat;

        // Variables for Torben's algorithm
        int less, greater, equal;
        float min, max, guess, maxltguess, mingtguess;

        // Step 1: Initialize min and max for the median
        min = max = curve_data[0];
        for (int i = 1; i < N_dat; i++) {
            float val = curve_data[i];
            if (val < min) min = val;
            if (val > max) max = val;
        }

        // Step 2: Compute the median using Torben's algorithm
        while (true) {
            guess = (min + max) / 2.0f;
            less = 0;
            greater = 0;
            equal = 0;
            maxltguess = min;
            mingtguess = max;

            for (int i = 0; i < N_dat; i++) {
                float val = curve_data[i];
                if (val < guess) {
                    less++;
                    if (val > maxltguess) maxltguess = val;
                } else if (val > guess) {
                    greater++;
                    if (val < mingtguess) mingtguess = val;
                } else {
                    equal++;
                }
            }

            if (less <= (N_dat + 1) / 2 && greater <= (N_dat + 1) / 2) break;
            else if (less > greater) max = maxltguess;
            else min = mingtguess;
        }

        // Determine the median
        float median_value;
        if (less >= (N_dat + 1) / 2) {
            median_value = maxltguess;
        } else if (less + equal >= (N_dat + 1) / 2) {
            median_value = guess;
        } else {
            median_value = mingtguess;
        }
        medians[idx] = median_value;

        // Step 3: Initialize min and max for the MAD
        min = fabsf(curve_data[0] - median_value);
        max = min;
        for (int i = 1; i < N_dat; i++) {
            float abs_diff = fabsf(curve_data[i] - median_value);
            if (abs_diff < min) min = abs_diff;
            if (abs_diff > max) max = abs_diff;
        }

        // Step 4: Compute the MAD using Torben's algorithm
        while (true) {
            guess = (min + max) / 2.0f;
            less = 0;
            greater = 0;
            equal = 0;
            maxltguess = min;
            mingtguess = max;

            for (int i = 0; i < N_dat; i++) {
                float abs_diff = fabsf(curve_data[i] - median_value);
                if (abs_diff < guess) {
                    less++;
                    if (abs_diff > maxltguess) maxltguess = abs_diff;
                } else if (abs_diff > guess) {
                    greater++;
                    if (abs_diff < mingtguess) mingtguess = abs_diff;
                } else {
                    equal++;
                }
            }

            if (less <= (N_dat + 1) / 2 && greater <= (N_dat + 1) / 2) break;
            else if (less > greater) max = maxltguess;
            else min = mingtguess;
        }

        // Determine the MAD
        float mad_value;
        if (less >= (N_dat + 1) / 2) {
            mad_value = maxltguess;
        } else if (less + equal >= (N_dat + 1) / 2) {
            mad_value = guess;
        } else {
            mad_value = mingtguess;
        }
        mads[idx] = mad_value;
    }
}


std::tuple<float, float> pg_stats(std::vector<float> v) {

    /*
    Function to calculate periodogram statistics, namely the
    periodogram mean and standard deviation

    v = vector of values
    */

   std::vector<float> vFiltered;
   std::copy_if(v.begin(), v.end(), std::back_inserter(vFiltered), [](float x){ return !std::isnan(x); });

    const float m = mean(vFiltered);
    float accum = 0.0f;
    std::for_each (std::begin(vFiltered), std::end(vFiltered), [&](const float d) {
        accum += (d - m) * (d - m);
    });

    float stdev = sqrt(accum / (vFiltered.size()-1));
    return {m, stdev};
}


std::vector<int> flagTerrestrialFreq(float* frequencies, int Nfreq) {

    /*
    Function to identify and flag terrestrial frequencies
    from an array of frequency values.

    frequencies = pointer to array of frequencies
    Nfreq       = number of frequencies
    */

    // Define some standard frequencies in terms of Hz
    float f_year = 0.002737803f;   // Sidereal year (365.256363 days)
    float f_lunar = 0.036600996f;  // Lunar sidereal month (27.321661 days)
    float f_day = 1.00273719f;     // Sidereal day (23.9344696 hours)

    // Vector of frequency/width pairs indicating the
    // regions to be excluded from the frequencies vector
    std::vector<std::pair<float,float>> ef = {
        {f_year,   0.00025f},   // 1 year
        {f_year/2, 0.000125f},  // 2 year
        {f_year/3, 0.0000835f}, // 3 year
        {f_year/4, 0.0000625f}, // 4 year
        {f_year/5, 0.0000500f}, // 5 year
        {f_year*2, 0.00025f},   // 1/2 year
        {f_year*3, 0.00025f},   // 1/3 year
        {f_year*4, 0.00025f},   // 1/4 year
        {f_lunar,  0.005f},     // 1 lunar month
        {f_day,    0.05f},      // 1 day
        {f_day/2,  0.02f},      // 2 day
        {f_day/3,  0.01f},      // 3 day
        {f_day/4,  0.01f},      // 4 day
        {f_day/5,  0.01f},      // 5 day
        {f_day*2,  0.05f},      // 1/2 day
        {f_day*3,  0.05f},
        {f_day*2 / 3, 0.02f},      // 3/2 day
        {f_day*3 / 2, 0.02f},   // 2/3 day
        {f_day*4 / 3, 0.02f},   // 3/4 day
        {f_day*5 / 2, 0.02f},   // 2/5 day
        {f_day*7 / 2, 0.02f},   // 2/7 day
        {f_day*4,  0.05f},      // 1/4 day
        {f_day*5,  0.05f},      // 1/5 day
        {f_day*6,  0.05f},      // 1/6 day
        {f_day*7,  0.05f},      // 1/7 day
        {f_day*8,  0.05f},      // 1/8 day
        {f_day*9,  0.05f},      // 1/9 day
        {f_day*10, 0.05f}       // 1/10 day
    };
    size_t Nregions = ef.size();

    // Create vector to store keep/discard indices
    std::vector<int> keepFreq(Nfreq,1);

    // Iterate over all frequencies
    for (int i = 0; i < Nfreq; i++) {
        for (int j = 0; j < Nregions; j++) {
            if ((frequencies[i] > ef[j].first - ef[j].second) &&
                (frequencies[i] < ef[j].first + ef[j].second)) {
                keepFreq[i] = 0;
            }
        }
    }

    // Clear memory
    std::vector<std::pair<float,float>>().swap(ef);

    return keepFreq;
}



std::vector<int> find_peaks(const std::vector<float>& x,
                            const int N_peaks,
                            const int distance,
                            const float median,
                            const float mad,
                            const float threshold) {

    /*
    Function to find peaks in a 1D array of values

    This is a C++ implementation of the scipy.signal.find_peaks
    function. It does not perform any smoothing and treats any
    point as a peak whose two neighboring points are lesser
    values, with options to apply constraints on the peak
    heights and the minimum distance between neighboring peaks.

    x         = Vector of periodogram values
    N_peaks      = number of peaks to return
    distance  = minimum distance between neighboring peaks
    median    = periodogram median
    mad       = periodogram median absolute deviation
    threshold = S/N threshold for a peak [ S/N = (height-median) / mad ]
    */

    std::vector<int> midpoints;
    midpoints.reserve(x.size() / 2); // Reserve memory to avoid frequent reallocations
    int numpeaks = 0;
    int i = 1; // Pointer to current sample (first one can't be a maximum)
    int i_max = x.size() - 1; // Last sample also can't be a maximum
    int i_ahead = i + 1; // Index to look ahead one sample
    float snr; // Param to store S/N of each periodogram value

    // Find the peaks
    while (i < i_max) {

        // First check that value exceeds threshold
        snr = (x[i] - median) / mad;
        if (snr < threshold) {
            i += 1;
            continue;
        }

        // Test if previous sample is smaller
        if (x[i - 1] < x[i]) {
            i_ahead = i + 1;

            // Find next sample unequal to x[i]
            while (i_ahead < i_max && x[i_ahead] == x[i]) {
                i_ahead += 1;
            }

            // Maxima is found if next unequal sample is smaller than x[i]
            if (x[i_ahead] < x[i]) {
                midpoints.push_back((i + i_ahead - 1) / 2);
                i = i_ahead;
                numpeaks += 1;
            }
        }
        i += 1;
    }

    // Create vector to store indices of the "N_peaks" highest peaks
    std::vector<int> itop(N_peaks, -1);

    // First check that some peaks were found.
    if (midpoints.size() < 1) {
        return itop;
    }

    // Create a vector containing peak heights
    std::vector<float> peak_heights(numpeaks);
    for (int i = 0; i < numpeaks; i++) {
        peak_heights[i] = x[midpoints[i]];
    }

    // Create a boolean vector that will eventually identify
    // which peaks pass the distance filter
    std::vector<bool> keep(numpeaks, true);

    // Create vector to store indices of each value in "midpoints"
    std::vector<int> priority_to_position(numpeaks);
    std::iota(priority_to_position.begin(), priority_to_position.end(), 0);

    // Sort priority_to_position vector in ascending order of peak heights
    std::stable_sort(priority_to_position.begin(), priority_to_position.end(),
                     [&peak_heights](int i1, int i2) { return peak_heights[i1] < peak_heights[i2]; });


    // Iterate over peaks in order of highest priority first
    for (int i = numpeaks - 1; i >= 0; --i) {

        // Translate i to j, where j points to current
        // peak whose neighbours are to be evaluated
        int j = priority_to_position[i];

        // Skip evaluation for peaks already marked as "don't keep"
        if (!keep[j]) {
            continue;
        }

        int k = j - 1;
        // Flag earlier peaks for removal until minimal distance is exceeded
        while (k >= 0 && midpoints[j] - midpoints[k] < distance) {
            keep[k] = false;
            --k;
        }

        k = j + 1;
        // Flag later peaks for removal until minimal distance is exceeded
        while (k < numpeaks && midpoints[k] - midpoints[j] < distance) {
            keep[k] = false;
            ++k;
        }
    }


    // Get indices of highest peaks
    int pcount = 0; // Counter to track number of peaks saved
    for (int i = numpeaks - 1; i >= 0; --i) {

        // Get pointer to peak position in the "midpoints" vector
        int j = priority_to_position[i];
        if (keep[j]) {
            itop[pcount] = midpoints[j];
            ++pcount;
        }

        // Break out of the for-loop once "N_peaks" peaks are saved
        if (pcount >= N_peaks) break;
    }

    // Clear some memory
    std::vector<int>().swap(midpoints);
    std::vector<float>().swap(peak_heights);
    std::vector<bool>().swap(keep);
    std::vector<int>().swap(priority_to_position);
    return itop;
}

//Greedy algorithm to find the top N peaks. This algorithm is not perfect, but it is much faster than global search-sort.
#define N_peaks 32
__global__ void cudaFindPeaks(float* data, float* median, float* mad,
                              int* peakIndices, float* peakValues,
                              int N_dat, int N_curves, int distance, float threshold) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_curves) {
        // Pointer to the start of the curve data for this thread
        float* curve_data = data + idx * N_dat;
        // Pointers to the output arrays for this curve
        float* curve_peak_values_out = peakValues + idx * N_peaks;
        int* curve_peak_indices_out = peakIndices + idx * N_peaks;

        // --- Local storage for top N peaks for this thread ---
        int top_indices[N_peaks];
        float top_values[N_peaks];
        for (int k = 0; k < N_peaks; ++k) {
            top_indices[k] = -1;
            top_values[k] = -INFINITY; // Use -INFINITY
        }
        // --- End Local Storage ---

        // --- Step 1: Compute significances in-place ---
        float safe_mad = (mad[idx] > 1e-12f) ? mad[idx] : 1e-12f; // Use safe_mad
        for (int i = 0; i < N_dat; i++) {
            curve_data[i] = (curve_data[i] - median[idx]) / safe_mad;
        }
        // --- End Step 1 ---

        // --- Step 2: Find peaks using threshold pivot and manage top N ---
        size_t i = 1; // Start from the second element
        size_t i_max = N_dat - 1; // Go up to the second-to-last element

        int distLastPeak = distance; // Initialize to allow first peak
        float currentThreshold = threshold; // Start with base threshold
        int candidatePeakIdx = -1;
        float candidatePeakVal = -INFINITY;

        while (i < i_max) {
            // Check 1: Local maximum condition (potential peak start)
            if (curve_data[i] > curve_data[i - 1]) {
                size_t i_ahead = i + 1;
                // Handle plateaus: find next different value
                while (i_ahead < i_max && curve_data[i_ahead] == curve_data[i]) {
                    i_ahead++;
                }

                // Check if it's a peak (next different value is lower)
                if (i_ahead < i_max && curve_data[i_ahead] < curve_data[i]) {
                    // Potential peak found.
                    int current_peak_idx = static_cast<int>((i + i_ahead - 1) / 2); // Midpoint index
                    float current_peak_val = curve_data[i]; // Value at start of plateau

                    // Check 2: Significance threshold and pivot
                    if (current_peak_val >= currentThreshold) {
                        // This peak is higher than the current threshold (or the base threshold)
                        // Make it the new candidate
                        candidatePeakIdx = current_peak_idx;
                        candidatePeakVal = current_peak_val;
                        distLastPeak = 0; // Reset distance since we found a new candidate
                        currentThreshold = current_peak_val; // Pivot threshold: only accept higher peaks now
                    }
                    // Move main loop index past the plateau
                    i = i_ahead;
                    continue; // Continue to next iteration to increment distance counter
                }
            }

            // Increment distance if a candidate is active
            if (candidatePeakIdx != -1) {
                distLastPeak++;
            }

            // Check 3: Distance constraint met, insert candidate
            if (distLastPeak == distance && candidatePeakIdx != -1) {
                // Insert the candidate peak into the sorted top_N list
                int ins_pos = N_peaks;
                for (size_t j = 0; j < N_peaks; ++j) {
                    if (candidatePeakVal > top_values[j]) {
                        ins_pos = j;
                        break;
                    }
                }
                if (ins_pos < N_peaks) {
                    // Shift elements down
                    for (int k = N_peaks - 1; k > ins_pos; --k) {
                        top_indices[k] = top_indices[k - 1];
                        top_values[k] = top_values[k - 1];
                    }
                    // Insert new peak
                    top_indices[ins_pos] = candidatePeakIdx;
                    top_values[ins_pos] = candidatePeakVal;
                }

                // Reset for next search window
                distLastPeak = 0; // Reset distance counter (start counting from the accepted peak)
                currentThreshold = threshold; // Reset threshold to base
                candidatePeakIdx = -1; // Clear candidate
                candidatePeakVal = -INFINITY;
            }

            i++; // Move to the next point if no peak/plateau was handled
        }
        // --- End Step 2 ---

        // --- Step 3: Write results from local arrays to global memory ---
        for (size_t k = 0; k < N_peaks; ++k) {
            curve_peak_indices_out[k] = top_indices[k];
            // Use the actual S/N value if found, otherwise 0.0f
            curve_peak_values_out[k] = (top_indices[k] != -1) ? top_values[k] : 0.0f;
        }
        // --- End Step 3 ---
    }
}


__global__ void cudaFindPeaksChunk(float* data, float* median, float* mad,
                                   int* peakIndices, float* peakValues,
                                   int N_dat, int N_curves, int distance, float threshold,
                                   int* state_candidate_idx, float* state_candidate_val,
                                   int* state_distLastPeak, float* state_currentThreshold,
                                   int chunk_start, int chunk_end) {
        
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_curves) {
        float* curve_data = data + idx * N_dat;
        float* curve_peak_values_out = peakValues + idx * N_peaks;
        int* curve_peak_indices_out = peakIndices + idx * N_peaks;

        // --- Load top N peaks from global memory ---
        int top_indices[N_peaks];
        float top_values[N_peaks];
        for (size_t k = 0; k < N_peaks; ++k) {
            top_indices[k] = curve_peak_indices_out[k];
            top_values[k] = curve_peak_values_out[k];
        }

        // --- Load persistent state ---
        int candidatePeakIdx = state_candidate_idx[idx];
        float candidatePeakVal = state_candidate_val[idx];
        int distLastPeak = state_distLastPeak[idx];
        float currentThreshold = state_currentThreshold[idx];

        float safe_mad = (mad[idx] > 1e-12f) ? mad[idx] : 1e-12f;
        // Only normalize the current chunk
        for (size_t i = chunk_start; i < chunk_end; i++) {
            curve_data[i] = (curve_data[i] - median[idx]) / safe_mad;
        }

        size_t i = max(static_cast<size_t>(chunk_start), size_t(1));
        size_t i_max = min(chunk_end, N_dat - 1);

        while (i < i_max) {
            if (curve_data[i] > curve_data[i - 1]) {
                size_t i_ahead = i + 1;
                while (i_ahead < i_max && curve_data[i_ahead] == curve_data[i]) {
                    i_ahead++;
                }
                if (i_ahead < i_max && curve_data[i_ahead] < curve_data[i]) {
                    int current_peak_idx = static_cast<int>((i + i_ahead - 1) / 2);
                    float current_peak_val = curve_data[i];
                    if (current_peak_val >= currentThreshold) {
                        candidatePeakIdx = current_peak_idx;
                        candidatePeakVal = current_peak_val;
                        distLastPeak = 0;
                        currentThreshold = current_peak_val;
                    }
                    i = i_ahead;
                    continue;
                }
            }
            if (candidatePeakIdx != -1) {
                distLastPeak++;
            }
            if (distLastPeak == distance && candidatePeakIdx != -1) {
                int ins_pos = N_peaks;
                for (size_t j = 0; j < N_peaks; ++j) {
                    if (candidatePeakVal > top_values[j]) {
                        ins_pos = j;
                        break;
                    }
                }
                if (ins_pos < N_peaks) {
                    for (int k = N_peaks - 1; k > ins_pos; --k) {
                        top_indices[k] = top_indices[k - 1];
                        top_values[k] = top_values[k - 1];
                    }
                    top_indices[ins_pos] = candidatePeakIdx;
                    top_values[ins_pos] = candidatePeakVal;
                }
                distLastPeak = 0;
                currentThreshold = threshold;
                candidatePeakIdx = -1;
                candidatePeakVal = -INFINITY;
            }
            i++;
        }

        // --- Save state for next chunk ---
        state_candidate_idx[idx] = candidatePeakIdx;
        state_candidate_val[idx] = candidatePeakVal;
        state_distLastPeak[idx] = distLastPeak;
        state_currentThreshold[idx] = currentThreshold;

        // --- Write results from local arrays to global memory ---
        for (size_t k = 0; k < N_peaks; ++k) {
            curve_peak_indices_out[k] = top_indices[k];
            curve_peak_values_out[k] = (top_indices[k] != -1) ? top_values[k] : 0.0f;
        }
    }
}