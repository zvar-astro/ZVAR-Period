#ifndef READ_DATA_H
#define READ_DATA_H

#include <numeric>
#include <iomanip>
#include <tuple>
#include "hdf5_utils.h"

std::string writeFilePath(const int field, const int ccd, const int quad, const std::string &filter);

std::tuple<unsigned int, unsigned int, std::vector<float>> initializeFrameData(const std::string &filename);

#endif // READ_DATA_H