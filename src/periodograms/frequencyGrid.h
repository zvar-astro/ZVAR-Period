#ifndef FREQUENCYGRID_H
#define FREQUENCYGRID_H

#include <vector>
#include <algorithm>
#include <numeric>

std::vector<double> freq_grid(const std::vector<double>& t, double fmin = -1, double fmax = -1, int oversample = 3);

#endif