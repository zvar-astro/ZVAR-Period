#ifndef RUN_FILE_H
#define RUN_FILE_H

#include "periodograms/FPW/wrappers.cuh"
#include "readwrite/hdf5_utils.h"
#include "readwrite/filepaths.h"
#include "periodograms/frequencyGrid.h"
#include "statistics/curve_statistics.cuh"
#include "statistics/polyfit.cuh"

#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

//STUFF FOR LOADING H5 FILES
#include <iomanip>
#include <tuple>

using namespace std;

void runFile(int field, int ccd, int quad, std::string filter);

#endif //RUN_FILE_H