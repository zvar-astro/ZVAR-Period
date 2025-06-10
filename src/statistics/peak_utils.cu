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
#include "peak_utils.cuh"


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
        float* curve_data = data + idx * N_dat;

        // Step 1: Find min/max ignoring NaNs
        float min = NAN, max = NAN;
        int valid_count = 0;
        for (int i = 0; i < N_dat; i++) {
            float val = curve_data[i];
            if (!isnan(val)) {
                if (valid_count == 0) min = max = val;
                if (val < min) min = val;
                if (val > max) max = val;
                valid_count++;
            }
        }
        if (valid_count == 0) {
            medians[idx] = NAN;
            mads[idx] = NAN;
            return;
        }

        // Step 2: Torben's median, ignoring NaNs
        float guess, maxltguess, mingtguess;
        int less, greater, equal;
        while (true) {
            guess = (min + max) / 2.0f;
            less = greater = equal = 0;
            maxltguess = min;
            mingtguess = max;
            for (int i = 0; i < N_dat; i++) {
                float val = curve_data[i];
                if (isnan(val)) continue;
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
            if (less <= (valid_count + 1) / 2 && greater <= (valid_count + 1) / 2) break;
            else if (less > greater) max = maxltguess;
            else min = mingtguess;
        }
        float median_value;
        if (less >= (valid_count + 1) / 2) {
            median_value = maxltguess;
        } else if (less + equal >= (valid_count + 1) / 2) {
            median_value = guess;
        } else {
            median_value = mingtguess;
        }
        medians[idx] = median_value;

        // Step 3: MAD min/max ignoring NaNs
        min = max = NAN;
        valid_count = 0;
        for (int i = 0; i < N_dat; i++) {
            float val = curve_data[i];
            if (!isnan(val)) {
                float abs_diff = fabsf(val - median_value);
                if (valid_count == 0) min = max = abs_diff;
                if (abs_diff < min) min = abs_diff;
                if (abs_diff > max) max = abs_diff;
                valid_count++;
            }
        }
        if (valid_count == 0) {
            mads[idx] = NAN;
            return;
        }

        // Step 4: Torben's MAD, ignoring NaNs
        while (true) {
            guess = (min + max) / 2.0f;
            less = greater = equal = 0;
            maxltguess = min;
            mingtguess = max;
            for (int i = 0; i < N_dat; i++) {
                float val = curve_data[i];
                if (isnan(val)) continue;
                float abs_diff = fabsf(val - median_value);
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
            if (less <= (valid_count + 1) / 2 && greater <= (valid_count + 1) / 2) break;
            else if (less > greater) max = maxltguess;
            else min = mingtguess;
        }
        float mad_value;
        if (less >= (valid_count + 1) / 2) {
            mad_value = maxltguess;
        } else if (less + equal >= (valid_count + 1) / 2) {
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

__global__ void cudaComputeSignificance(float* data, float* median, float* mad, int N_dat, int N_curves) {
    /*
    CUDA kernel to compute significance of each data point in the curves.
    The significance is computed as (data - median) / mad for each curve.
    Also cleans out NaN values.
    This is done in-place, modifying the original data array.

    data = pointer to the data (N_dat * N_curves)
    median = pointer to the median values (N_curves)
    mad = pointer to the MAD values (N_curves)
    N_dat = number of data points in each curve
    N_curves = number of curves
    */

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_curves) {
        float* curve_data = data + idx * N_dat;
        float safe_mad = (mad[idx] > 1e-12f) ? mad[idx] : 1e-12f; // Use a safe value for mad

        for (int i = 0; i < N_dat; i++) {
            curve_data[i] = (curve_data[i] - median[idx]) / safe_mad;
            if (isnan(curve_data[i])) {
                curve_data[i] = -INFINITY; // Replace NaN with -INFINITY
            }
        }
    }
}

#define N_peaks 32
//Greedy algorithm to find the top N peaks. This algorithm is not perfect, but it is much faster than global search-sort.
__device__ bool find_next_peak(
    float* curve_data, size_t& sample_idx, const size_t last_peak_check,
    int& peak_idx, float& peak_value
) {
    if (curve_data[sample_idx] > curve_data[sample_idx - 1] &&
        curve_data[sample_idx] > curve_data[sample_idx + 1]) {
        peak_idx = static_cast<int>(sample_idx);
        peak_value = curve_data[sample_idx];
        return true;
    }
    return false;
}

__device__ void insert_peak(
    int* top_peak_indices, float* top_peak_values,
    int candidate_peak_idx, float candidate_peak_value
) {
    int insert_pos = N_peaks;
    for (int rank = 0; rank < N_peaks; ++rank) {
        if (candidate_peak_value > top_peak_values[rank]) {
            insert_pos = rank;
            break;
        }
    }
    if (insert_pos < N_peaks) {
        for (int k = N_peaks - 1; k > insert_pos; --k) {
            top_peak_indices[k] = top_peak_indices[k - 1];
            top_peak_values[k] = top_peak_values[k - 1];
        }
        top_peak_indices[insert_pos] = candidate_peak_idx;
        top_peak_values[insert_pos] = candidate_peak_value;
    }
}

__global__ void cudaFindPeaks(
    float* data,
    int* peakIndices,
    float* peakValues,
    int N_dat,
    int N_curves,
    int distance,
    float threshold
) {
    size_t curve_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (curve_idx >= N_curves) return;

    float* curve_data = data + curve_idx * N_dat;
    float* curve_peak_values_out = peakValues + curve_idx * N_peaks;
    int* curve_peak_indices_out = peakIndices + curve_idx * N_peaks;

    int top_peak_indices[N_peaks];
    float top_peak_values[N_peaks];
    for (int peak_rank = 0; peak_rank < N_peaks; ++peak_rank) {
        top_peak_indices[peak_rank] = -1;
        top_peak_values[peak_rank] = -INFINITY;
    }

    size_t sample_idx = 1;
    const size_t last_peak_check = N_dat - 1;

    int distance_since_last_peak = distance;
    float dynamic_threshold = threshold;
    int candidate_peak_idx = -1;
    float candidate_peak_value = -INFINITY;

    while (sample_idx < last_peak_check) {
        int local_peak_idx = -1;
        float local_peak_value = -INFINITY;
        if (find_next_peak(curve_data, sample_idx, last_peak_check, local_peak_idx, local_peak_value)) {
            if (local_peak_value >= dynamic_threshold) {
                candidate_peak_idx = local_peak_idx;
                candidate_peak_value = local_peak_value;
                distance_since_last_peak = 0;
                dynamic_threshold = local_peak_value;
            }
        }

        if (candidate_peak_idx != -1) {
            distance_since_last_peak++;
        }

        if (distance_since_last_peak == distance && candidate_peak_idx != -1) {
            insert_peak(top_peak_indices, top_peak_values, candidate_peak_idx, candidate_peak_value);
            distance_since_last_peak = 0;
            dynamic_threshold = threshold;
            candidate_peak_idx = -1;
            candidate_peak_value = -INFINITY;
        }

        sample_idx++;
    }

    for (int peak_rank = 0; peak_rank < N_peaks; ++peak_rank) {
        curve_peak_indices_out[peak_rank] = top_peak_indices[peak_rank];
        curve_peak_values_out[peak_rank] = (top_peak_indices[peak_rank] != -1) ? top_peak_values[peak_rank] : 0.0f;
    }
}


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
){
    size_t curve_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (curve_idx >= N_curves) return;

    // Pointers to this curve's data and output
    float* curve_data = data + curve_idx * N_dat;
    float* last_chunk_curve_data = last_chunk_data + curve_idx * 2;
    float* curve_peak_values_out = peakValues + curve_idx * N_peaks;
    int* curve_peak_indices_out = peakIndices + curve_idx * N_peaks;

    // --- Local storage for top N peaks (descending order) ---
    int top_peak_indices[N_peaks];
    float top_peak_values[N_peaks];
    for (int peak_rank = 0; peak_rank < N_peaks; ++peak_rank) {
        top_peak_indices[peak_rank] = curve_peak_indices_out[peak_rank];
        top_peak_values[peak_rank] = curve_peak_values_out[peak_rank];
    }

    // --- Load persistent state for this curve ---
    int candidate_peak_idx = state_candidate_idx[curve_idx];
    float candidate_peak_value = state_candidate_val[curve_idx];
    int distance_since_last_peak = state_distLastPeak[curve_idx];
    float dynamic_threshold = state_currentThreshold[curve_idx];

    // --- Peak finding loop for this chunk ---
    size_t sample_idx = 1; // Index being checked for a peak
    size_t last_peak_check = N_dat - 1; // Last index to check for a peak, this is the second to last sample of this chunk

    int local_peak_idx = -1; // Index of a peak found locally within curve_data
    float local_peak_value = -INFINITY; // Value of the local peak

    // Handle the last sample of the previous chunk and the first sample of this chunk
    if (chunk_start_global_idx > 0) {
        // If this is not the first chunk, we have two samples from the previous chunk
        // We need to check if the last sample from the last chunk is a peak.
        if (last_chunk_curve_data[1] > last_chunk_curve_data[0] &&
            last_chunk_curve_data[1] > curve_data[0]) {
            local_peak_idx = -1;
            local_peak_value = last_chunk_curve_data[1];

            if (local_peak_value >= dynamic_threshold) {
                candidate_peak_idx = local_peak_idx + chunk_start_global_idx; // Convert local peak index to global index
                candidate_peak_value = local_peak_value;
                distance_since_last_peak = 0; // Reset distance for the new candidate
                dynamic_threshold = local_peak_value; // Pivot: new peaks must be higher than this candidate
            }
        }

        // If there's an active candidate peak, increment its distance counter.
        if (candidate_peak_idx != -1) {
            distance_since_last_peak++;
        }

        // If the candidate peak has survived for the required distance
        if (distance_since_last_peak == distance && candidate_peak_idx != -1) {
            // The candidate peak is confirmed. Insert it into the top N peaks.
            insert_peak(top_peak_indices, top_peak_values, candidate_peak_idx, candidate_peak_value);

            // Reset state for finding the next peak
            distance_since_last_peak = 0;
            dynamic_threshold = threshold;
            candidate_peak_idx = -1;
            candidate_peak_value = -INFINITY;
        }
    }

    local_peak_idx = -1; // Reset local peak index for the current chunk
    local_peak_value = -INFINITY; // Reset local peak value for the current chunk
    
    // Handle the first sample of the current chunk
    if (chunk_start_global_idx > 0) {
        // If this is not the first chunk, we have two samples from the previous chunk
        // We need to check if the first sample from this chunk is a peak.
        if (curve_data[0] > last_chunk_curve_data[1] &&
            curve_data[0] > curve_data[1]) {
            local_peak_idx = 0;
            local_peak_value = curve_data[0];

            if (local_peak_value >= dynamic_threshold) {
                candidate_peak_idx = local_peak_idx + chunk_start_global_idx; // Convert local peak index to global index
                candidate_peak_value = local_peak_value;
                distance_since_last_peak = 0; // Reset distance for the new candidate
                dynamic_threshold = local_peak_value; // Pivot: new peaks must be higher than this candidate
            }
        }

        // If there's an active candidate peak, increment its distance counter.
        if (candidate_peak_idx != -1) {
            distance_since_last_peak++;
        }

        // If the candidate peak has survived for the required distance
        if (distance_since_last_peak == distance && candidate_peak_idx != -1) {
            // The candidate peak is confirmed. Insert it into the top N peaks.
            insert_peak(top_peak_indices, top_peak_values, candidate_peak_idx, candidate_peak_value);

            // Reset state for finding the next peak
            distance_since_last_peak = 0;
            dynamic_threshold = threshold;
            candidate_peak_idx = -1;
            candidate_peak_value = -INFINITY;
        }
    }

    // Iterate through the samples in the current chunk
    while (sample_idx < last_peak_check) {
        local_peak_idx = -1; // Index of a peak found locally within curve_data
        local_peak_value = -INFINITY; // Value of the local peak
        if (find_next_peak(curve_data, sample_idx, last_peak_check, local_peak_idx, local_peak_value)) {
            // A local peak was found at local_peak_idx (index relative to curve_data)
            if (local_peak_value >= dynamic_threshold) {
                candidate_peak_idx = local_peak_idx + chunk_start_global_idx; //Convert local peak index to global index
                candidate_peak_value = local_peak_value;
                distance_since_last_peak = 0; // Reset distance for the new candidate
                dynamic_threshold = local_peak_value; // Pivot: new peaks must be higher than this candidate
            }
        }

        // If there's an active candidate peak, increment its distance counter.
        if (candidate_peak_idx != -1) {
            distance_since_last_peak++;
        }

        // If the candidate peak has survived for the required distance
        if (distance_since_last_peak == distance && candidate_peak_idx != -1) {
            // The candidate peak is confirmed. Insert it into the top N peaks.
            insert_peak(top_peak_indices, top_peak_values, candidate_peak_idx, candidate_peak_value);

            // Reset state for finding the next peak
            distance_since_last_peak = 0;
            dynamic_threshold = threshold;
            candidate_peak_idx = -1;
            candidate_peak_value = -INFINITY;
        }

        // Increment the sample index to check the next sample
        // Note: sample_idx is relative to the start of the chunk, so it does not need to be adjusted for global index.
        sample_idx++;
    }
    // --- End of peak finding loop ---

    // --- Save the state for the next chunk ---
    state_candidate_idx[curve_idx] = candidate_peak_idx; // Save the candidate peak index for the next chunk
    state_candidate_val[curve_idx] = candidate_peak_value; // Save the candidate peak value for the next chunk
    state_distLastPeak[curve_idx] = distance_since_last_peak; // Save the distance since last peak for the next chunk
    state_currentThreshold[curve_idx] = dynamic_threshold; // Save the current dynamic threshold for the next chunk
    // --- End of state saving ---

    // --- Write the top N peaks to the output ---
    for (int peak_rank = 0; peak_rank < N_peaks; ++peak_rank) {
        curve_peak_indices_out[peak_rank] = top_peak_indices[peak_rank];
        curve_peak_values_out[peak_rank] = (top_peak_indices[peak_rank] != -1) ? top_peak_values[peak_rank] : 0.0f;
    }
}