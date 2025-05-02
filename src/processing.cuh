#ifndef RUN_FILE_H
#define RUN_FILE_H

#include "periodograms/FPW/wrappers.cuh"
#include "readwrite/hdf5_utils.h"
#include "statistics/polyfit.cuh"
#include "statistics/peak_utils.h"
#include "periodograms/frequencyGrid.h"
#include "statistics/curve_statistics.cuh"
#include "msc/diagnostics.h"
#include "msc/semaphores.h"
#include "msc/memory_managers.cuh"

#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <iomanip>
#include <tuple>
#include <omp.h>

using namespace std;

void process(
    std::string hdf_filename,
    int device,
    std::string logpath,
    const char *id_field_name,
    const char *ra_field_name,
    const char *dec_field_name,
    const char *time_field_name,
    const char *flux_field_name,
    const char *fluxErr_field_name
);

#endif //RUN_FILE_H