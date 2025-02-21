/*Function to calculate a generic frequency grid*/
#include "frequencyGrid.h"

/**
 * @brief Creates a grid of frequencies for a periodogram to run over
 *
 * This function uses the time sampling of a field to create a nyquist sampled frequency grid.
 *
 * @param t The timestamps of field data.
 * @param fmin The minimum frequency to consider.
 * @param fmax The maximum frequency to consider.
 * @param oversample The oversampling factor.
 * @return A vector of frequencies.
 */
std::vector<double> freq_grid(const std::vector<double>& t, double fmin, double fmax, int oversample) {
    // Validate inputs
    if (t.empty() || oversample <= 0 || fmin >= fmax) return std::vector<double>();

    double trange = *max_element(t.begin(), t.end()) - *min_element(t.begin(), t.end());

    std::vector<double> sorted_t = t;
    std::sort(sorted_t.begin(), sorted_t.end());
    std::vector<double> diff_t;
    std::adjacent_difference(sorted_t.begin(), sorted_t.end(), std::back_inserter(diff_t));
    diff_t.erase(diff_t.begin()); // Remove the first element
    double texp = *min_element(diff_t.begin(), diff_t.end());

    double fres = 1.0 / (trange * oversample);
    if (fmax == -1) fmax = 0.5 / texp; // Adjust fmax if set to -1
    if (fmin == -1) fmin = fres; // Adjust fmin if set to -1

    std::vector<double> fgrid;
    for (double f = fmin; f < fmax; f += fres) {
        fgrid.push_back(f);
    }

    //Free vectors from memory
    std::vector<double>().swap(sorted_t);
    std::vector<double>().swap(diff_t);

    return fgrid;
}