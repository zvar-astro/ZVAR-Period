#ifndef RUN_FILE_H
#define RUN_FILE_H

#include "periodograms/FPW/wrappers.cuh"
#include "readwrite/hdf5_utils.h"
#include "statistics/polyfit.cuh"
#include "statistics/peak_utils.cuh"
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
    float percent_indices = 0.5f; // Default percentage of memory to use for indices

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
    float peak_threshold = 10.0f; // Minimum significance threshold
    bool success = false; // To indicate if loading was successful

    //Seasonal/Deep Drilling/Baseline Removal Modes
    bool seasonal = false; // Default: not seasonal
    bool deep_drilling = false; // Default: remove deep drilling
    bool remove_baseline = true; // Default: remove baseline
    bool clip_pgram = false; // Default: not clip periodogram

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
        bool seasonal = false,
        bool deep_drilling = false,
        bool remove_baseline = true,
        bool clip_pgram = false,
        double f_min = 1.0 / 10.0, // 10 day max period
        double f_max = 1440.0 / 5.0, // 5 min min period
        // double f_max = 1.0 / 50.0, // 50 day min period
        // double f_min = 1.0 / (3.0 * 365), // 3 year max period
        int freq_oversampling = 3,
        int model_id = 15,
        const std::vector<int>& N_bins = {20, 10, 5},
        int peak_distance = 300,
        float peak_threshold = 10.0f
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
          seasonal(seasonal),
          deep_drilling(deep_drilling),
          remove_baseline(remove_baseline),
          clip_pgram(clip_pgram),
          success(false) {}
};

struct FieldData {
    std::vector<float> times_float;
    std::vector<double> times_double;
    std::vector<unsigned char> dd_indices;
    std::vector<float> fgrid;
    std::vector<double> fgrid_double;
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
    std::vector<float> fluxes_medians;
    std::vector<float> fluxes_mads;
    std::vector<float> periodogram_means;
    std::vector<float> periodogram_stds;
    std::vector<int> peak_indices;
    std::vector<float> peak_freqs;
    std::vector<float> peak_values;
    bool success = false; // To indicate if loading was successful

    SourcesData(const ProcessingExternalParams params, FieldData& field_data)
        : psids(field_data.N_src_per_batch, 0),
          ras(field_data.N_src_per_batch, 0.0f),
          decs(field_data.N_src_per_batch, 0.0f),
          ratioValid(field_data.N_src_per_batch, 0.0f),
          fluxes(field_data.N_src_per_batch * field_data.N_exp, 0.0f),
          fluxErrs(field_data.N_src_per_batch * field_data.N_exp, INFINITY),
          fluxes_no_baseline(field_data.N_src_per_batch * field_data.N_exp, 0.0f),
          flux_ivars(field_data.N_src_per_batch * field_data.N_exp, 0.0f),
          fluxes_medians(field_data.N_src_per_batch, 0.0f),
          fluxes_mads(field_data.N_src_per_batch, 0.0f),
          periodogram_means(field_data.N_src_per_batch * params.N_samples, 0.0f),
          periodogram_stds(field_data.N_src_per_batch * params.N_samples, 0.0f),
          peak_indices(field_data.N_src_per_batch * params.N_samples * N_peaks, -1),
          peak_freqs(field_data.N_src_per_batch * params.N_samples * N_peaks, -1.0f),
          peak_values(field_data.N_src_per_batch * params.N_samples * N_peaks, -1.0f) {}
};

struct KernelLaunchParams {
    dim3 block_size_preprocess;
    dim3 grid_size_preprocess;

    dim3 block_size_indices;
    dim3 grid_size_indices;

    dim3 block_size_deltaChi2;
    dim3 grid_size_deltaChi2;
    size_t shared_memory_deltaChi2;

    dim3 block_size_postprocess;
    dim3 grid_size_postprocess;

    KernelLaunchParams(const ProcessingExternalParams& params, FieldData& field_data)
        : block_size_preprocess(512),
          grid_size_preprocess((field_data.N_src_per_batch + block_size_preprocess.x - 1) / block_size_preprocess.x),
          
          block_size_indices(32, 32),
          grid_size_indices((field_data.N_freqs_per_batch + block_size_indices.x - 1) / block_size_indices.x,
                            (field_data.N_exp + block_size_indices.y - 1) / block_size_indices.y),

          block_size_deltaChi2(32, 8),
          grid_size_deltaChi2((field_data.N_freqs_per_batch + block_size_deltaChi2.x - 1) / block_size_deltaChi2.x,
                                (field_data.N_src_per_batch + block_size_deltaChi2.y - 1) / block_size_deltaChi2.y),
          shared_memory_deltaChi2(2 * params.N_bins[0] * block_size_deltaChi2.x * block_size_deltaChi2.y * sizeof(float)),

          block_size_postprocess(512),
          grid_size_postprocess((field_data.N_src_per_batch * params.N_samples + block_size_postprocess.x - 1) / block_size_postprocess.x) {}
};

struct DeviceArrays {
    float* t_device;
    float* y_device;
    float* ivar_device;
    float* y_medians_device;
    float* y_mads_device;
    unsigned char* indices_device;
    float* periodograms_device;
    float* freqs_device;
    int* N_bins_device;
    float* periodogram_means_device;
    float* periodogram_stds_device;
    int* peak_indices_device;
    float* peak_values_device;

    DeviceArrays(const ProcessingExternalParams& params, FieldData& field_data)
        : t_device(nullptr),
          y_device(nullptr),
          ivar_device(nullptr),
          y_medians_device(nullptr),
          y_mads_device(nullptr),
          indices_device(nullptr),
          periodograms_device(nullptr),
          freqs_device(nullptr),
          N_bins_device(nullptr),
          periodogram_means_device(nullptr),
          periodogram_stds_device(nullptr),
          peak_indices_device(nullptr),
          peak_values_device(nullptr) {
        cudaMalloc(&t_device, field_data.N_exp * sizeof(float));
        cudaMalloc(&y_device, field_data.N_src_per_batch * field_data.N_exp * sizeof(float));
        cudaMalloc(&y_medians_device, field_data.N_src_per_batch * sizeof(float));
        cudaMalloc(&y_mads_device, field_data.N_src_per_batch * sizeof(float));
        cudaMalloc(&ivar_device, field_data.N_src_per_batch * field_data.N_exp * sizeof(float));
        cudaMalloc(&indices_device, field_data.N_exp * field_data.N_freqs_per_batch * sizeof(unsigned char));
        cudaMalloc(&periodograms_device, field_data.N_src_per_batch * field_data.N_freqs_per_batch * params.N_samples * sizeof(float));
        cudaMalloc(&freqs_device, field_data.N_freqs_per_batch * sizeof(float));
        cudaMalloc(&N_bins_device, params.N_samples * sizeof(int));
        cudaMalloc(&periodogram_means_device, field_data.N_src_per_batch * params.N_samples * sizeof(float));
        cudaMalloc(&periodogram_stds_device, field_data.N_src_per_batch * params.N_samples * sizeof(float));
        cudaMalloc(&peak_indices_device, field_data.N_src_per_batch * params.N_samples * N_peaks * sizeof(int));
        cudaMalloc(&peak_values_device, field_data.N_src_per_batch * params.N_samples * N_peaks * sizeof(float));

        cudaMemcpyAsync(t_device, field_data.times_float.data(), field_data.N_exp * sizeof(float), cudaMemcpyHostToDevice); // copy the times over, this is only done once.
        cudaMemcpyAsync(N_bins_device, params.N_bins.data(), params.N_samples * sizeof(int), cudaMemcpyHostToDevice); // copy the N_bins over, this is only done once.

        size_t N_exp_total = field_data.N_src_per_batch * field_data.N_exp;
        size_t N_peaks_total = field_data.N_src_per_batch * params.N_samples * N_peaks;
        size_t threads = 256;
        size_t blocks_exp = (N_exp_total + threads - 1) / threads;
        size_t blocks_peaks = (N_peaks_total + threads - 1) / threads;

        setFloatArrayKernel<<<blocks_exp, threads>>>(y_device, 0.0f, N_exp_total);
        setFloatArrayKernel<<<blocks_exp, threads>>>(ivar_device, 0.0f, N_exp_total);

        setIntArrayKernel<<<blocks_peaks, threads>>>(peak_indices_device, -1, N_peaks_total);
        setFloatArrayKernel<<<blocks_peaks, threads>>>(peak_values_device, -INFINITY, N_peaks_total);

        cudaDeviceSynchronize(); // Ensure all memory operations are complete before proceeding

        //Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA error during device memory allocation");
        }
    }
    ~DeviceArrays() {
        // Free device memory here
        cudaFree(t_device);
        cudaFree(y_device);
        cudaFree(ivar_device);
        cudaFree(y_medians_device);
        cudaFree(y_mads_device);
        cudaFree(indices_device);
        cudaFree(periodograms_device);
        cudaFree(freqs_device);
        cudaFree(N_bins_device);
        cudaFree(periodogram_means_device);
        cudaFree(periodogram_stds_device);
        cudaFree(peak_indices_device);
        cudaFree(peak_values_device);

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA error during device memory deallocation");
        }
    }
};

struct PeakPersistentArrays {
    int* state_candidate_idx;
    float* state_candidate_vals;
    int* state_dist_last_peak;
    float* state_current_threshold;
    float* last_chunk_data_device;

    PeakPersistentArrays(const ProcessingExternalParams& params, FieldData& field_data)
        : state_candidate_idx(nullptr),
          state_candidate_vals(nullptr),
          state_dist_last_peak(nullptr),
          state_current_threshold(nullptr),
          last_chunk_data_device(nullptr) {
        // Allocate device memory
        size_t N = field_data.N_src_per_batch * params.N_samples;
        size_t threads = 256;
        size_t blocks = (N + threads - 1) / threads;

        cudaMalloc(&state_candidate_idx, N * sizeof(int));
        cudaMalloc(&state_candidate_vals, N * sizeof(float));
        cudaMalloc(&state_dist_last_peak, N * sizeof(int));
        cudaMalloc(&state_current_threshold, N * sizeof(float));
        cudaMalloc(&last_chunk_data_device, N * 2 * sizeof(float));

        setIntArrayKernel<<<blocks, threads>>>(state_candidate_idx, -1, N);
        setFloatArrayKernel<<<blocks, threads>>>(state_candidate_vals, -INFINITY, N);
        setIntArrayKernel<<<blocks, threads>>>(state_dist_last_peak, params.peak_distance, N);
        setFloatArrayKernel<<<blocks, threads>>>(state_current_threshold, params.peak_threshold, N);
        setFloatArrayKernel<<<blocks, threads>>>(last_chunk_data_device, 0.0f, N * 2);

        cudaDeviceSynchronize(); // Ensure all memory operations are complete before proceeding

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA error during peak persistent arrays allocation");
        }
    }
    ~PeakPersistentArrays() {
        cudaFree(state_candidate_idx);
        cudaFree(state_candidate_vals);
        cudaFree(state_dist_last_peak);
        cudaFree(state_current_threshold);
        cudaFree(last_chunk_data_device);

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA error during peak persistent arrays deallocation");
        }
    }
};

FieldData loadAndPrepareFieldData(const ProcessingExternalParams& params);

void createAndFilterFrequencyGrid(const ProcessingExternalParams& params, FieldData& field_data);

void initializeChunkParameters(const ProcessingExternalParams& params, FieldData& field_data);

void loadChunkData(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data,
                   unsigned long objstart, unsigned long objnum);

void removeOutliers(FieldData& field_data, SourcesData& sources_data, DeviceArrays& device_arrays,
                    KernelLaunchParams& kernel_params, unsigned long objnum_periodograms);

void processBaselineRemoval(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data,
                            unsigned long objstart, unsigned long objnum);

void copyPhotToDevice(FieldData& field_data, SourcesData& sources_data, DeviceArrays& device_arrays, unsigned long objnum_periodograms);

void copyPeaksToHost(const ProcessingExternalParams& params, FieldData& field_data, SourcesData& sources_data,
                     DeviceArrays& device_arrays, unsigned long objnum_periodograms);

void resetPersitentArrays(DeviceArrays& device_arrays, PeakPersistentArrays& peak_arrays, const ProcessingExternalParams& params, FieldData& field_data);

void copyLastChunkData(PeakPersistentArrays& peak_arrays, DeviceArrays& device_arrays, int num_periodogram_curves, unsigned long N_freqs_current);

void removeTempStatus(const ProcessingExternalParams& params, string results_filename);

void process(const ProcessingExternalParams& params);

#endif //RUN_FILE_H