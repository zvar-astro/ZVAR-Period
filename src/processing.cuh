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
    std::vector<float> flux_ivars;
    bool success = false; // To indicate if loading was successful

    SourcesData(FieldData& field_data)
        : psids(field_data.N_src_per_batch, 0),
          ras(field_data.N_src_per_batch, 0.0f),
          decs(field_data.N_src_per_batch, 0.0f),
          ratioValid(field_data.N_src_per_batch, 0.0f),
          fluxes(field_data.N_src_per_batch * field_data.N_exp, 0.0f),
          fluxErrs(field_data.N_src_per_batch * field_data.N_exp, INFINITY),
          fluxes_no_baseline(field_data.N_src_per_batch * field_data.N_exp, 0.0f),
          flux_ivars(field_data.N_src_per_batch * field_data.N_exp, 0.0f) {}
};

struct KernelLaunchParams {
    dim3 block_size_indices;
    dim3 grid_size_indices;

    dim3 block_size_deltaChi2;
    dim3 grid_size_deltaChi2;
    size_t shared_memory_deltaChi2;

    dim3 block_size_postprocess;
    dim3 grid_size_postprocess;

    KernelLaunchParams(const ProcessingExternalParams& params, FieldData& field_data)
        : block_size_indices(32, 32),
          grid_size_indices((field_data.N_freqs_per_batch + block_size_indices.x - 1) / block_size_indices.x,
                            (field_data.N_exp + block_size_indices.y - 1) / block_size_indices.y),
          block_size_deltaChi2(32, 8),
          grid_size_deltaChi2((field_data.N_freqs_per_batch + block_size_deltaChi2.x - 1) / block_size_deltaChi2.x,
                                (field_data.N_src_per_batch + block_size_deltaChi2.y - 1) / block_size_deltaChi2.y),
          shared_memory_deltaChi2(2 * params.N_bins[0] * block_size_deltaChi2.x * block_size_deltaChi2.y * sizeof(float)),
          block_size_postprocess(256),
          grid_size_postprocess((field_data.N_src_per_batch * params.N_samples + block_size_postprocess.x - 1) / block_size_postprocess.x) {}
};

struct DeviceArrays {
    float* t_device;
    float* y_device;
    float* ivar_device;
    unsigned char* indices_device;
    float* periodograms_device;
    float* freqs_device;
    int* N_bins_device;
    float* medians_device;
    float* mads_device;
    int* peak_indices_device;
    float* peak_values_device;

    DeviceArrays(const ProcessingExternalParams& params, FieldData& field_data)
        : t_device(nullptr),
          y_device(nullptr),
          ivar_device(nullptr),
          indices_device(nullptr),
          periodograms_device(nullptr),
          freqs_device(nullptr),
          N_bins_device(nullptr),
          medians_device(nullptr),
          mads_device(nullptr),
          peak_indices_device(nullptr),
          peak_values_device(nullptr) {
        cudaMalloc(&t_device, field_data.N_exp * sizeof(float));
        cudaMalloc(&y_device, field_data.N_src_per_batch * field_data.N_exp * sizeof(float));
        cudaMalloc(&ivar_device, field_data.N_src_per_batch * field_data.N_exp * sizeof(float));
        cudaMalloc(&indices_device, field_data.N_exp * field_data.N_freqs_per_batch * sizeof(unsigned char));
        cudaMalloc(&periodograms_device, field_data.N_src_per_batch * field_data.N_freqs_per_batch * params.N_samples *  sizeof(float));
        cudaMalloc(&freqs_device, field_data.N_freqs_per_batch * sizeof(float));
        cudaMalloc(&N_bins_device, params.N_samples * sizeof(int));
        cudaMalloc(&medians_device, field_data.N_src_per_batch * sizeof(float));
        cudaMalloc(&mads_device, field_data.N_src_per_batch * sizeof(float));
        cudaMalloc(&peak_indices_device, field_data.N_src_per_batch * N_peaks * sizeof(int));
        cudaMalloc(&peak_values_device, field_data.N_src_per_batch * N_peaks * sizeof(float));

        cudaMemcpyAsync(t_device, field_data.times.data(), field_data.N_exp * sizeof(float), cudaMemcpyHostToDevice); // copy the times over, this is only done once.
        cudaMemcpyAsync(N_bins_device, params.N_bins.data(), params.N_samples * sizeof(int), cudaMemcpyHostToDevice); // copy the N_bins over, this is only done once.
    }
    ~DeviceArrays() {
        // Free device memory here
        cudaFree(t_device);
        cudaFree(y_device);
        cudaFree(ivar_device);
        cudaFree(indices_device);
        cudaFree(periodograms_device);
        cudaFree(freqs_device);
        cudaFree(N_bins_device);
        cudaFree(medians_device);
        cudaFree(mads_device);
        cudaFree(peak_indices_device);
        cudaFree(peak_values_device);
    }
};

struct PeakPersistentArrays {
    int* state_candidate_idx;
    float* state_candidate_vals;
    int* state_dist_last_peak;
    float* state_current_threshold;

    PeakPersistentArrays(const ProcessingExternalParams& params, FieldData& field_data)
        : state_candidate_idx(nullptr),
          state_candidate_vals(nullptr),
          state_dist_last_peak(nullptr),
          state_current_threshold(nullptr) {
        // Allocate device memory here
        cudaMalloc(&state_candidate_idx, field_data.N_src_per_batch * params.N_samples * sizeof(int));
        cudaMalloc(&state_candidate_vals, field_data.N_src_per_batch * params.N_samples * sizeof(float));
        cudaMalloc(&state_dist_last_peak, field_data.N_src_per_batch * params.N_samples * sizeof(int));
        cudaMalloc(&state_current_threshold, field_data.N_src_per_batch * params.N_samples * sizeof(float));

        cudaMemsetAsync(state_candidate_idx, -1, field_data.N_src_per_batch * params.N_samples * sizeof(int));
        cudaMemsetAsync(state_candidate_vals, -INFINITY, field_data.N_src_per_batch * params.N_samples * sizeof(float));
        cudaMemsetAsync(state_dist_last_peak, 0, field_data.N_src_per_batch * params.N_samples * sizeof(int));
        cudaMemsetAsync(state_current_threshold, params.peak_threshold, field_data.N_src_per_batch * params.N_samples * sizeof(float));
    }
    ~PeakPersistentArrays() {
        // Free device memory here
        cudaFree(state_candidate_idx);
        cudaFree(state_candidate_vals);
        cudaFree(state_dist_last_peak);
        cudaFree(state_current_threshold);
    }
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