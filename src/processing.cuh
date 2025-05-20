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
#define N_peaks 32

struct ProcessingExternalParams {
    // Hardware/Concurrency
    int device = 0; // Default GPU device
    int num_omp_threads = 1; // Default OpenMP threads
    float percent_indices = 0.5f; // Default percentage of indices to use

    // File/Directory Names
    std::string input_hdf_filename;
    std::string log_path; // Default log path
    std::string output_dir; // Default output directory

    // HDF5 Field Names
    std::string id_field_name;
    std::string ra_field_name;
    std::string dec_field_name;
    std::string time_field_name;
    std::string flux_field_name;
    std::string fluxErr_field_name;

    // Frequency Grid Generation
    double f_min = 1.0 / 10.0; // Default: 10 day max period
    double f_max = 1440.0 / 5.0; // Default: 5 min min period
    int freq_oversampling = 3; // Default: 3x oversampling

    // Baseline Removal Parameters
    int model_id = 15; // Default: 15 (7th order polynomial)

    // Periodogram (FPW) Parameters
    std::vector<int> N_bins = {20, 10, 5}; // Default: 20, 10, 5 bins
    int N_samples = N_bins.size(); // Compute N_samples as the length of N_bins

    // Peak Finding Parameters
    int peak_distance = 300; // Minimum distance between peaks
    float peak_threshold = 7.0f; // Minimum significance threshold
    bool success = false; // To indicate if loading was successful

    // Constructor
    ProcessingExternalParams(
        int device = 0,
        int num_omp_threads = 1,
        float percent_indices = 0.5f,
        const std::string& input_hdf_filename = "",
        const std::string& log_path = "",
        const std::string& output_dir = "",
        const std::string& id_field_name = "",
        const std::string& ra_field_name = "",
        const std::string& dec_field_name = "",
        const std::string& time_field_name = "",
        const std::string& flux_field_name = "",
        const std::string& fluxErr_field_name = "",
        double f_min = 1.0 / 10.0,
        double f_max = 1440.0 / 5.0,
        int freq_oversampling = 3,
        int model_id = 15,
        const std::vector<int>& N_bins = {20, 10, 5},
        int peak_distance = 300,
        float peak_threshold = 7.0f
    )
        : device(device),
          num_omp_threads(num_omp_threads),
          input_hdf_filename(input_hdf_filename),
          log_path(log_path),
          output_dir(output_dir),
          id_field_name(id_field_name),
          ra_field_name(ra_field_name),
          dec_field_name(dec_field_name),
          time_field_name(time_field_name),
          flux_field_name(flux_field_name),
          fluxErr_field_name(fluxErr_field_name),
          f_min(f_min),
          f_max(f_max),
          freq_oversampling(freq_oversampling),
          model_id(model_id),
          N_bins(N_bins),
          N_samples(N_bins.size()),
          peak_distance(peak_distance),
          peak_threshold(peak_threshold),
          success(false) {}
};

struct FieldData {
    std::vector<float> times;
    std::vector<unsigned char> dd_indices;
    std::vector<float> fgrid;
    unsigned int N_exp;
    unsigned int N_src;
    unsigned long N_freqs;
    unsigned long N_freqs_batch;
    unsigned long N_freqs_per_batch;
    unsigned long N_src_batch;
    unsigned long N_src_per_batch; 
    unsigned long N_src_batch_gpu_baseline;
    unsigned long N_batch_gpu_baseline;
    
    bool success = false; // To indicate if loading was successful
};

struct SourcesData {
    std::vector<unsigned long> psids;
    std::vector<float> ras;
    std::vector<float> decs;
    std::vector<float> ratioValid; //Ratio of exposures used to total exposures
    std::vector<float> fluxes;
    std::vector<float> fluxErrs;
    std::vector<float> fluxes_no_baseline;
    bool success = false; // To indicate if loading was successful

    SourcesData(FieldData& field_data)
        : psids(field_data.N_src_per_batch, 0),
          ras(field_data.N_src_per_batch, 0.0f),
          decs(field_data.N_src_per_batch, 0.0f),
          ratioValid(field_data.N_src_per_batch, 0.0f),
          fluxes(field_data.N_src_per_batch * field_data.N_exp, 0.0f),
          fluxErrs(field_data.N_src_per_batch * field_data.N_exp, INFINITY),
          fluxes_no_baseline(field_data.N_src_per_batch * field_data.N_exp, 0.0f) {}
};

FieldData loadAndPrepareFieldData(const ProcessingExternalParams& params);

void createAndFilterFrequencyGrid(const ProcessingExternalParams& params, FieldData& field_data);

void initializeChunkParameters(const ProcessingExternalParams& params, FieldData& field_data);

void loadChunkData(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data,
                   unsigned long objstart, unsigned long objnum);

void processBaselineRemoval(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data,
                            unsigned long objstart, unsigned long objnum);

void process(const ProcessingExternalParams& params);

#endif //RUN_FILE_H