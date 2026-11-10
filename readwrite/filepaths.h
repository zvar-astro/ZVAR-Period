#ifndef FILEPATHS_H
#define FILEPATHS_H

#include <numeric>
#include <iomanip>
#include <tuple>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include "hdf5_utils.h"

std::string writeInputFilePath(const int field, const int ccd, const int quad, const std::string &filter);

std::string writeOutputFilePath(const int field, const int ccd, const int quad, const std::string &filter);

std::tuple<unsigned int, unsigned int, std::vector<float>, std::vector<unsigned char>> initializeFrameData(const std::string &filename);

#endif // FILEPATHS_H