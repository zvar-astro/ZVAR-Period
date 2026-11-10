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
                            const int Ntop,
                            const int distance,
                            const float mean,
                            const float stdev,
                            const float threshold) {

    /*
    Function to find peaks in a 1D array of values

    This is a C++ implementation of the scipy.signal.find_peaks
    function. It does not perform any smoothing and treats any
    point as a peak whose two neighboring points are lesser
    values, with options to apply constraints on the peak
    heights and the minimum distance between neighboring peaks.

    x         = Vector of periodogram values
    Ntop      = number of peaks to return
    distance  = minimum distance between neighboring peaks
    mean      = periodogram mean
    stdev     = periodogram standard deviation
    threshold = S/N threshold for a peak [ S/N = (height-mean) / stdev ]
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
        snr = (x[i] - mean) / stdev;
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

    // Create vector to store indices of the "Ntop" highest peaks
    std::vector<int> itop(Ntop, -1);

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

        // Break out of the for-loop once "Ntop" peaks are saved
        if (pcount >= Ntop) break;
    }

    // Clear some memory
    std::vector<int>().swap(midpoints);
    std::vector<float>().swap(peak_heights);
    std::vector<bool>().swap(keep);
    std::vector<int>().swap(priority_to_position);
    return itop;
}
