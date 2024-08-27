#ifndef RUN_FILE_H
#define RUN_FILE_H

#include "periodograms/FPW/wrappers.cuh"
#include "readwrite/hdf5_utils.h"
#include "readwrite/filepaths.h"
#include "periodograms/frequencyGrid.h"
#include "statistics/curve_statistics.cuh"
#include "statistics/polyfit.cuh"
#include "statistics/peak_utils.h"
#include "msc/diagnostics.h"
#include "msc/semaphores.h"

#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <iomanip>
#include <tuple>
#include <omp.h>

using namespace std;

void runFile(int field, int ccd, int quad, std::string filter, std::string logpath);

#endif //RUN_FILE_H