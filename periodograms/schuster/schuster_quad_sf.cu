// A CUDA program that implements the Schuster
// Periodogram algorithm on GPUs to analyze the
// Caltech Stellar forced photometry light curves
// one ZTF quadrant at a time.
//
// Compile with: 
//
// nvcc schuster_quad_sf.cu hdf5_utils.cu peak_utils.cu -o schuster -I /usr/include/hdf5/serial/ -L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5 -lstdc++fs -Xcompiler -fopenmp -lgomp -O3
//
//
// Author: Zach Vanderbosch (Caltech)


#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <tuple>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <experimental/filesystem>

#include <omp.h>
#include <thrust/complex.h>
#include "hdf5_utils.h"
#include "peak_utils.h"
#include "hdf5.h"

namespace fs = std::experimental::filesystem;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Some global parameters

// Define PI constant as a float
const float pi = 3.141592653589793238462643383279502884197;

// Define Pan-STARRS1 Object ID to save periodogram for
// const unsigned long objID = 161213227363186155;  // In field 769, ccd 2, quad 3 (Ra-dec = 322.7362802 44.3462273, ZTFJ2130+4420, P=39.3 min)
const unsigned long objID = 171792853559000466;  // In field 798, ccd 5, quad 4 (Ra-dec = 285.3559167 53.1581306, ZTFJ1901+5309, P=40.6 min)
const std::string pgfilename = "../data/pgram_obj.dat";

// Set number of threads along one block dimension,
// (e.g. one block has THREADS X THREADS total threads)
const long int THREADS = 16;  // 256 (16x16) threads per block

// Define some periodogram parameters
const float PMIN = 300.0f;                // Minimum period of frequency grid (5 minutes)
const float PMAX = 3 * 365.25 * 86400.0f; // Maximum period of frequency grid (3 years)
const int OFACTOR = 3;       // Over-sampling factor for frequency grid 
                             // (i.e. df = 1/T/ofactor, where T = light curve time baseline)

// Peak finding parameters
const int NPEAKS = 25;              // Number of highest peaks to save
const int DISTANCE = 100 * OFACTOR; // Minimum number of samples required between peaks

// Set max number of CPU threads to use during parallelization
const int MAX_CPU_THREADS = 8;  // Chosen for Feldspar computer with 8-core (16 thread) CPU

// Set base directory for forced photometry HDF5 matchfiles
std::string MATCHFILE_DIR = "/data/zvar/matchfiles/";

// Set base directory to save final period detections
std::string PERIOD_DIR = "/data/zvar/zvar_results/";



// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Kernel Code


__global__ void sincosCalc(const float *time,
                           const float *grid,
                           thrust::complex<float> *sincos,
                           const float pi,
                           long int N_f) {

  /* Kernel for calculating matrix of sine/cosine terms 
     N_f = Number of frequencies (padded)
  */

  // Calculate the global indices for each threadi
  long int col = blockIdx.x * blockDim.x + threadIdx.x; // Indicates the frequency
  long int row = blockIdx.y * blockDim.y + threadIdx.y; // Indicates the exposure

  // Calculate each term
  sincos[row*N_f+col] = thrust::complex<float>(
    std::cos(2.0f * pi * grid[col] * time[row]),
    std::sin(2.0f * pi * grid[col] * time[row])
  );
}


__global__ void pgramCalc(const float *flux,
                          thrust::complex<float> *sincos, 
                          float *pgram,
                          long int N_d,
                          long int N_f) {

  /* Kernel for calcuating the Schuster periodogram
     N_d = Number of data points per source
     N_f = Number of frequencies (padded)
  */
  
  // Calculate the global indices for each threadi
  long int col = blockIdx.x * blockDim.x + threadIdx.x; // Indicates the frequency
  long int row = blockIdx.y * blockDim.y + threadIdx.y; // Indicates the source
  
  thrust::complex<float> complexSum = thrust::complex<float>(0.0f,0.0f);
  for (int i = 0; i < N_d; i++) {
    complexSum += flux[row*N_d+i] * sincos[i*N_f+col];
  }
  
  // Take absolute value and normalize to get periodogram
  pgram[row*N_f+col] = (
    std::pow(complexSum.real(),2) + 
    std::pow(complexSum.imag(),2)
  ) / static_cast<float>(N_d);
}


__global__ void pgramStats(float *pgram, 
                           float *pg_mean,
                           float *pg_stdev,
                           long int N_f, 
                           long int N_f_padded) {

  /* Kernel for calcuating Mean and StDev of each periodogram
     N_f = Number of frequencies (un-padded)
     N_f_padded = Number of frequencies (padded)
  */
  
  // Calculate the global index for each thread
  long int row = blockIdx.x * blockDim.x + threadIdx.x; // Indicates the source
  
  // Calculate mean
  float mean = 0.0f;
  for (int i = 0; i < N_f; i++) {
    mean += pgram[row*N_f_padded+i];
  }
  mean = mean / N_f;
  pg_mean[row] = mean;

  // Calculate standard deviation
  float stdev = 0.0f;
  for (int i = 0; i < N_f; i++) {
    stdev += (pgram[row*N_f_padded+i] - mean) * 
             (pgram[row*N_f_padded+i] - mean);
  }
  stdev = std::sqrt(stdev / (N_f-1));
  pg_stdev[row] = stdev;
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// The main function

int main(int argc, char const *argv[]) {

  // Get timestamp at start of program execution
  std::chrono::steady_clock::time_point t_prog_start = std::chrono::steady_clock::now();


  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Command-line argument parsing

  // Initial check on argument count
  if (argc < 6) {
    printf("Usage: %s [-v] <field> <ccd> <quad> <filter> <device_id>\n", argv[0]);
    return 0;
  }

  // Count required command line args and set verbosity
  bool quiet = true;
  int argc_required = 0;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) != "-v") argc_required++;
    else quiet = false;
  }

  // Check for correct number of required arguments
  if (argc_required != 5) {
    printf("Usage: %s [-v] <field> <ccd> <quad> <filter> <device_id>\n", argv[0]);
    return 0;
  }

  // Set field, ccd, quad, and filter params
  int arg_idx = 0; 
  int field, ccd, quad;
  int deviceNum;
  std::string filter;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) != "-v") {
      if (arg_idx == 0) field = std::stoi(argv[i]);
      if (arg_idx == 1) ccd = std::stoi(argv[i]);
      if (arg_idx == 2) quad = std::stoi(argv[i]);
      if (arg_idx == 3) filter = argv[i];
      if (arg_idx == 4) deviceNum = std::stoi(argv[i]);
      arg_idx++;
    }
  }
  

  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Define some filenames

  // Define the forced photometry HDF5 filename
  std::stringstream ssh;
  ssh << MATCHFILE_DIR;
  ssh << std::setfill('0') << std::setw(4) << field << "/data_";
  ssh << std::setfill('0') << std::setw(4) << field << "_";
  ssh << std::setfill('0') << std::setw(2) << ccd << "_";
  ssh << quad << "_z" << filter << ".h5";
  std::string hdf_filename = ssh.str();

  // Define directories/filename to store 
  // Schuster Periodogram (SP) period detections
  std::stringstream ssp;
  ssp << PERIOD_DIR;
  std::string period_path = ssp.str();
  ssp << std::setfill('0') << std::setw(4) << field;
  std::string period_field_path = ssp.str();
  ssp << "/periods_SP_";
  ssp << std::setfill('0') << std::setw(4) << field << "_";
  ssp << std::setfill('0') << std::setw(2) << ccd << "_";
  ssp << quad << "_z" << filter << ".dat";
  std::string peakfilename = ssp.str();

  // Create directories if they don't already exist
  fs::create_directory(period_path); // create PERIOD_DIR
  fs::create_directory(period_field_path); // create PERIOD_DIR/field folder

  // Define filename to store ALL Schuster Periodogram (SP) period 
  // detections from each frequency chunk (for testing purposes only)
  // std::stringstream ssap;
  // ssap << PERIOD_DIR;
  // ssap << std::setfill('0') << std::setw(4) << field << "/all_periods_SP_";
  // ssap << std::setfill('0') << std::setw(4) << field << "_";
  // ssap << std::setfill('0') << std::setw(2) << ccd << "_";
  // ssap << quad << "_z" << filter << ".dat";
  // std::string allpeakfilename = ssap.str();


  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Some initial setup and GPU selection

  std::cout << "Processing Field " << field;
  std::cout << ", CCD " << ccd;
  std::cout << ", Quad " << quad;
  std::cout << ", " << filter << "-band" << std::endl;

  // Set number of CPU threads for parallelization
  omp_set_num_threads(MAX_CPU_THREADS);

  // Get GPU global memory (restricting to 90% of max memory)
  cudaSetDevice(deviceNum);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, deviceNum);
  size_t bytesGlobal = device_prop.totalGlobalMem;
  size_t bytesMax = bytesGlobal * 90 / 100; // Limit to 90% to prevent memory overload
  
  if (!quiet) {
    std::cout << "\n  GPU Device:" << std::endl;
    std::cout << "  -------------------------------------" << std::endl;
    printf("  Using GPU Device %d\n", deviceNum);
    printf("  %s\n",device_prop.name);
    printf("  %.2f GB Global Memory\n", static_cast<float>(bytesGlobal) / (1<<30));
    printf("  Limiting Memory Usage to %.2f GB\n", static_cast<float>(bytesMax) / (1<<30));
  }

  // Clear out periodogram file if it exists
  const fs::path pgpath{pgfilename};
  if (fs::exists(pgpath)) {
    std::ofstream pgfile;
    pgfile.open(pgfilename);
    pgfile.close();
  }

  
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Load Data from HDF5 file

  // Check for the HDF5 file
  int success = 0;
  const fs::path hdf_path{hdf_filename};
  if (fs::exists(hdf_path)) success = 1;
  
  if (success == 0) {
    std::cout << "No HDF5 File found at " << hdf_filename << std::endl;
    return 0;
  }
  
  // Get number of exposures and sources
  int NEXP, NSRC;
  NEXP = get_number_of_rows_wrapper(hdf_filename, "/data/exposures");
  NSRC = get_number_of_rows_wrapper(hdf_filename, "/data/sources");

  // Use NEXP and NSRC to initialize BJD, time, and sourceID vectors
  std::vector<double> h_bjd(NEXP,0.0);        // BJD must be read in with double precision
  std::vector<unsigned long> h_psid(NSRC,0);  // Source IDs


  // Read HDF5 tables and load into vectors
  int status;
  status = get_exposures_wrapper(hdf_filename, h_bjd.data(), 0, NEXP);
  check_status(status);
  status = get_sources_wrapper(hdf_filename, h_psid.data(), 0, NSRC);
  check_status(status);

  // Get indices of non-deep drilling time stamps
  std::vector<unsigned char> dd_index = flagDeepDrilling(h_bjd, NEXP);
  int non_dd = std::accumulate(dd_index.begin(), dd_index.end(), 0);
  
  // Erase the deep drilling BJDs from bjd vector
  for (int i = NEXP-1; i >= 0; i--) {
    if (dd_index[i] == 0) {
      h_bjd.erase(h_bjd.begin() + i);
    }
  }

  // Reset the NEXP parameter and use to initialize the time vector
  NEXP = non_dd;
  std::vector<float> h_time(NEXP,0.0f); // Time values after subtracting mean BJD



  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Initialize time and frequency grid arrays

  // Subtract mean BJD from times and initialize h_time
  double bjdref = std::accumulate(h_bjd.begin(), h_bjd.end(), 0.0) / h_bjd.size();
  for (int i = 0; i < NEXP; i++) {
    h_time[i] = static_cast<float>(h_bjd[i] - bjdref);
  }

  // Determine the frequency grid params
  float tmax = *std::max_element(h_time.begin(), h_time.end());
  float tmin = *std::min_element(h_time.begin(), h_time.end());
  float fmin = 1.f / PMAX * 86400.f;
  float fmax = 1.f / PMIN * 86400.f;
  float fstep = 1.0f / (tmax - tmin) / static_cast<float>(OFACTOR);
  long int NFREQ = (fmax - fmin) / fstep;
 
  // Generate array to store full frequency grid
  std::vector<float> h_grid_full(NFREQ);

  // Initialize frequency grid
  for (int i = 0; i < NFREQ; i++){
    h_grid_full[i] = fmin + (i*fstep);
  }

  // Get indices to exclude terrestrial frequencies
  std::vector<int> freq_flags = flagTerrestrialFreq(h_grid_full.data(), NFREQ);

  // Erase terrestrial frequencies from vector
  for (int i = NFREQ-1; i >= 0; i--) {
    if (freq_flags[i] == 0) {
      h_grid_full.erase(h_grid_full.begin() + i);
    }
  }

  // Determine frequency grid padding
  NFREQ = std::accumulate(freq_flags.begin(), freq_flags.end(), 0);
  long int NFREQ_PADDED = NFREQ + THREADS - NFREQ % THREADS;
  std::cout << freq_flags.size() - NFREQ << " frequencies removed" << std::endl;

  // Add zero padding to frequency grid
  for (int i = NEXP; i < NFREQ_PADDED; i++) {
    h_grid_full.push_back(0.0f);
  }

  // Print out some HDF5 file summary info
  if (!quiet) {
    std::cout << "\n  HDF5 File Properties:" << std::endl;
    std::cout << "  -------------------------------------" << std::endl;
    std::cout << "  HDF5 file: " << hdf_filename << std::endl;
    std::cout << "  Sources  : " << NSRC << std::endl;
    std::cout << "  Exposures: " << dd_index.size();
    std::cout << " (" << NEXP << " without deep drilling)" <<  std::endl;
    std::cout << "  Time Span: " << tmax-tmin << " days\n";
    std::cout << "  Size of Frequency Grid: " << NFREQ << std::endl;
  }

  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Determine chunk size and padding per chunk

  // Determine source padding if using a single chunk
  long int NSRC_PADDED = NSRC + THREADS - NSRC % THREADS;

  // Determine exposure padding
  long int NEXP_PADDED = NEXP + THREADS - NEXP % THREADS;
  
  // Add zero padding to time vector
  for (int i = NEXP; i < NEXP_PADDED; i++) {
    h_time.push_back(0.0f);
  }

  // Set maximum memory used by sine/cosine terms (50% of max global mem) 
  size_t bytesMaxSincos = bytesMax * 50 / 100;

  // Define data sizes if using a single chunk
  size_t bytesTime = NEXP_PADDED * sizeof(float);
  size_t bytesFlux = NSRC_PADDED * NEXP_PADDED * sizeof(float);
  size_t bytesFreq = NFREQ_PADDED * sizeof(float);
  size_t bytesSincos = NFREQ_PADDED * NEXP_PADDED * sizeof(thrust::complex<float>);
  size_t bytesPgram = NFREQ_PADDED * NSRC_PADDED * sizeof(float);
  size_t bytesStats = 2 * NSRC_PADDED * sizeof(float);
  size_t bytesTotal = bytesTime + bytesFreq + bytesSincos + bytesFlux + bytesPgram + bytesStats;

  if (!quiet) {
    std::cout << "\n  Padding Requirements (Full Data Set)" << std::endl;
    std::cout << "  -------------------------------------" << std::endl;
    printf("  NEXP  padded from %d to %ld\n", NEXP, NEXP_PADDED);
    printf("  NSRC  padded from %d to %ld\n", NSRC, NSRC_PADDED);
    printf("  NFREQ padded from %ld to %ld\n", NFREQ, NFREQ_PADDED);
    std::cout << "\n  Memory Requirements (Full Data Set):\n";
    std::cout << "  -------------------------------------" << std::endl;
    printf("    Time (H2D): %.2f KB\n", static_cast<float>(bytesTime) / (1<<10));
    printf("    Flux (H2D): %.2f MB\n", static_cast<float>(bytesFlux) / (1<<20));
    printf("    Grid (H2D): %.2f MB\n", static_cast<float>(bytesFreq) / (1<<20));
    printf("  Sincos   (D): %.2f GB\n", static_cast<float>(bytesSincos) / (1<<30));
    printf("   Pgram (D2H): %.2f GB\n", static_cast<float>(bytesPgram) / (1<<30));
    printf("   Stats (D2H): %.2f MB\n", static_cast<float>(bytesStats) / (1<<20));
    printf("         TOTAL: %.2f GB\n", static_cast<float>(bytesTotal) / (1<<30));
  }

  // Determine the number of chunks needed to break up the sine/cosine terms
  int NCHUNK_SINCOS = 1;
  while ( bytesSincos / NCHUNK_SINCOS > bytesMaxSincos ) {
    ++NCHUNK_SINCOS;
  }
  
  // Determine number of frequencies per chunk (using ceiling of INT division)
  long int NFREQ_PER_CHUNK = 1 + ((NFREQ_PADDED - 1) / NCHUNK_SINCOS);

  // Reset the sizes of freq/sincos/pgram arrays per chunk
  if (NCHUNK_SINCOS > 1) {
    NFREQ_PADDED = NFREQ_PER_CHUNK + THREADS - NFREQ_PER_CHUNK % THREADS;
  }
  bytesFreq = NFREQ_PADDED * sizeof(float);
  bytesSincos = NFREQ_PADDED * NEXP_PADDED * sizeof(thrust::complex<float>);
  bytesPgram = NFREQ_PADDED * NSRC_PADDED * sizeof(float);

  // Determine how many chunks the data (flux) will require
  size_t bytesNeeded = bytesFlux + bytesPgram + bytesStats;
  size_t bytesAvailable = bytesMax - bytesSincos - bytesTime - bytesFreq;
  int NCHUNK_PGRAM = 1;
  while ( bytesNeeded / NCHUNK_PGRAM > bytesAvailable ) {
    ++NCHUNK_PGRAM;
  }

  // Determine number of sources per chunk
  long int NSRC_PER_CHUNK = 1 + ((NSRC_PADDED - 1) / NCHUNK_PGRAM);

  // Reset the sizes of flux, freq, and periodogram sizes per chunk
  if (NCHUNK_PGRAM > 1) {
    NSRC_PADDED = NSRC_PER_CHUNK + THREADS - NSRC_PER_CHUNK % THREADS;
  }
  bytesFlux = NSRC_PADDED * NEXP_PADDED * sizeof(float);
  bytesPgram = NFREQ_PADDED * NSRC_PADDED * sizeof(float);
  bytesStats = NSRC_PADDED * sizeof(float);
  bytesTotal = bytesTime + bytesFreq + bytesSincos + bytesFlux + bytesPgram + bytesStats;

  if (!quiet) {
    std::cout << "\n  Chunk Size Determinations" << std::endl;
    std::cout << "  -------------------------------------" << std::endl;
    std::cout << "  Splitting frequencies into " << NCHUNK_SINCOS << " ";
    std::cout << "chunks (" << NFREQ_PADDED << " freq per chunk)" << std::endl;
    std::cout << "  Splitting sources into " << NCHUNK_PGRAM << " chunks (";
    std::cout << NSRC_PADDED << " sources per chunk)" << std::endl;
    std::cout << "  Total number of chunks = " << NCHUNK_SINCOS * NCHUNK_PGRAM << std::endl;
    std::cout << "\n  Memory Requirements (Single Chunk):\n";
    std::cout << "  -------------------------------------" << std::endl;
    printf("    Time (H2D): %.2f KB\n", static_cast<float>(bytesTime) / (1<<10));
    printf("    Flux (H2D): %.2f MB\n", static_cast<float>(bytesFlux) / (1<<20));
    printf("    Grid (H2D): %.2f MB\n", static_cast<float>(bytesFreq) / (1<<20));
    printf("  Sincos (H2D): %.2f GB\n", static_cast<float>(bytesSincos) / (1<<30));
    printf("   Pgram (D2H): %.2f GB\n", static_cast<float>(bytesPgram) / (1<<30));
    printf("   Stats (D2H): %.2f MB\n", static_cast<float>(bytesStats) / (1<<20));
    printf("         TOTAL: %.2f GB\n", static_cast<float>(bytesTotal) / (1<<30));
  }

  // Generate arrays to store flux, grid, and periodogram values
  std::vector<float> h_flux(NSRC_PADDED * NEXP_PADDED, 0.0f);
  std::vector<float> h_grid(NFREQ_PADDED, 0.0f);
  std::vector<float> h_pgram(NFREQ_PADDED * NSRC_PADDED, 0.0f);

  // Generate arrays to store mean and stdev values for periodograms
  std::vector<float> h_pgmean(NSRC_PADDED, 0.0f);
  std::vector<float> h_pgstdev(NSRC_PADDED, 0.0f);


  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // GPU Setup

  // Create pointers for device (GPU) arrays
  float *d_time, *d_flux, *d_grid, *d_pgram, *d_pgmean, *d_pgstdev;
  thrust::complex<float> *d_sincos;

  // Threads per CTA dimension
  long int BLOCKS_X = NFREQ_PADDED / THREADS;
  long int BLOCKS_Y1 = NEXP_PADDED / THREADS;
  long int BLOCKS_Y2 = NSRC_PADDED / THREADS;
  dim3 threads(THREADS, THREADS);
  dim3 blocks1(BLOCKS_X, BLOCKS_Y1);
  dim3 blocks2(BLOCKS_X, BLOCKS_Y2);
  if (!quiet) {
    std::cout << "\n  GPU Block & Thread Dimensions" << std::endl;
    std::cout << "  -------------------------------------" << std::endl;
    std::cout << "  Single Thread Block = " << THREADS << "x" << THREADS << " threads\n";
    std::cout << "  Sine/Cosine Kernel  = " << BLOCKS_X << "x" << BLOCKS_Y1 << " blocks\n";
    std::cout << "  Periodogram Kernel  = " << BLOCKS_X << "x" << BLOCKS_Y2 << " blocks\n";
    std::cout << "  Pgram Stats Kernel  = " << BLOCKS_Y2 << " blocks\n";
  }

  // Allocate device (GPU) memory for sincos terms
  cudaMalloc(&d_sincos, bytesSincos);
  cudaMalloc(&d_time, bytesTime);
  cudaMalloc(&d_grid, bytesFreq);
  cudaMalloc(&d_flux, bytesFlux);
  cudaMalloc(&d_pgram, bytesPgram);
  cudaMalloc(&d_pgmean, bytesStats);
  cudaMalloc(&d_pgstdev, bytesStats);

  // Copy time array to device (GPU)
  cudaMemcpy(d_time, h_time.data(), bytesTime, cudaMemcpyHostToDevice);

  // Vectors to store all peak data from all chunks
  int Npeakcol = NPEAKS * NCHUNK_SINCOS;
  std::vector<float> all_peak_frequencies(NSRC * Npeakcol, 0.0f); 
  std::vector<float> all_peak_heights(NSRC * Npeakcol, 0.0f);
  std::vector<float> all_pgram_means(NSRC * NCHUNK_SINCOS, 0.0f); 
  std::vector<float> all_pgram_stdevs(NSRC * NCHUNK_SINCOS, 0.0f); 


  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // Kernel Execution
  
  std::cout << "\n  Kernel Execution" << std::endl;
  std::cout << "  -------------------------------------" << std::endl;

  // First loop iterates over frequency blocks
  for (unsigned long j = 0; j < NCHUNK_SINCOS; j++) {

    std::cout << "  Processing frequency chunk " << j+1 << "/" << NCHUNK_SINCOS << "\n";

    // Set frequency grid values
    for (int i = 0; i < NFREQ_PADDED; i++) {
      if (i < NFREQ_PER_CHUNK) {
        h_grid[i] = h_grid_full[j*NFREQ_PER_CHUNK+i];
      }
      else h_grid[i] = 0.0f;
    }

    // Copy frequency grid to device (GPU)
    cudaMemcpy(d_grid, h_grid.data(), bytesFreq, cudaMemcpyHostToDevice);

    // Calculate the sine/cosine terms
    sincosCalc<<<blocks1, threads>>>(d_time, d_grid, d_sincos, pi, NFREQ_PADDED);
    cudaDeviceSynchronize();

    // Second loop iterates over chunks of sources
    unsigned long objstart, objnum;

    for (unsigned long k = 0; k < NCHUNK_PGRAM; k++) {
      

      // Get the range of objects to analyze
      std::cout << "    Processing source chunk " << k+1 << "/" << NCHUNK_PGRAM << "\n";
      objstart = k * static_cast<unsigned long>(NSRC_PER_CHUNK);
      if (k < NCHUNK_PGRAM-1) {
        objnum = static_cast<unsigned long>(NSRC_PER_CHUNK);
      }
      else {
        objnum = static_cast<unsigned long>(NSRC) - 
                 (k * static_cast<unsigned long>(NSRC_PER_CHUNK));
      }

      // Check if object of interest is in this chunk
      int oidx;
      int idfound = 0;
      for (oidx = objstart; oidx < objstart+objnum; oidx++) {
        if (h_psid[oidx] == objID) {
          idfound = 1;
          std::cout << "      OBJECT FOUND!  Index = ";
          std::cout << oidx << " (" << oidx-objstart << ")" << std::endl;
          break;
        }
      }
      //if (idfound == 0) continue;
      
      // Load the flux values and normalize the light curves
      status = get_sourcedata_chunk(hdf_filename, h_flux.data(), dd_index, objstart, objnum, NEXP_PADDED);
      check_status(status);
      normalizeLC(h_flux.data(), objnum, NEXP_PADDED);

      // Copy flux data to GPU
      cudaMemcpy(d_flux, h_flux.data(), bytesFlux, cudaMemcpyHostToDevice);
  
      // Launch periodogram kernel
      pgramCalc<<<blocks2, threads>>>(d_flux, d_sincos, d_pgram, NEXP_PADDED, NFREQ_PADDED);

      // Copy periodogram back to the host
      cudaMemcpy(h_pgram.data(), d_pgram, bytesPgram, cudaMemcpyDeviceToHost);

      // Launch periodogram Stats kernel
      pgramStats<<<BLOCKS_Y2, THREADS>>>(d_pgram, d_pgmean, d_pgstdev, NFREQ_PER_CHUNK, NFREQ_PADDED);

      // Copy periodogram stats back to the host
      cudaMemcpy(h_pgmean.data(), d_pgmean, bytesStats, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pgstdev.data(), d_pgstdev, bytesStats, cudaMemcpyDeviceToHost);


      // Perform peak finding (using CPU parallelization)
      std::chrono::steady_clock::time_point t_begin = std::chrono::steady_clock::now();
      # pragma omp parallel for
      for (int i = 0; i < objnum; i++) {

        // Slice the vector
        std::vector<float>::const_iterator first = h_pgram.begin() + i*NFREQ_PADDED;
        std::vector<float>::const_iterator last = h_pgram.begin() + i*NFREQ_PADDED + NFREQ_PER_CHUNK;
        std::vector<float> pgram_slice(first, last);

        // Get indices of peaks for slice
        std::vector<int> peak_indices = find_peaks(pgram_slice, NPEAKS, DISTANCE);

        // Save peak frequencies and heights to full array
        if (peak_indices[0] > 0) { 
          for (int ii = 0; ii < NPEAKS; ii++) {
            all_peak_frequencies[(i+objstart)*Npeakcol + j*NPEAKS + ii] = h_grid[peak_indices[ii]];
            all_peak_heights[(i+objstart)*Npeakcol + j*NPEAKS + ii] = pgram_slice[peak_indices[ii]];
          }
          // Save mean and stdev values
          all_pgram_means[(i+objstart)*NCHUNK_SINCOS + j] = h_pgmean[i];
          all_pgram_stdevs[(i+objstart)*NCHUNK_SINCOS + j] = h_pgstdev[i];
        }
        else {
          for (int ii = 0; ii < NPEAKS; ii++) {
            all_peak_frequencies[(i+objstart)*Npeakcol + j*NPEAKS + ii] = -1.0f; 
            all_peak_heights[(i+objstart)*Npeakcol + j*NPEAKS + ii] = -1.0f;
          }
          // Save mean and stdev values
          all_pgram_means[(i+objstart)*NCHUNK_SINCOS + j] = -1.0f;
          all_pgram_stdevs[(i+objstart)*NCHUNK_SINCOS + j] = -1.0f;
        }

        // Clear memory
        std::vector<float>().swap(pgram_slice);
        std::vector<int>().swap(peak_indices);
      }
      std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();

      // Print out time duration for peak finding
      if (!quiet) {
        std::cout << "      Peak finding time = ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << "[ms]";
        std::cout << std::endl;
      }

      // Save Periodogram for object of interest
      if (idfound == 1) {
        std::cout << "      Saving periodogram to file...";
        std::ofstream pgfile;
        pgfile.open(pgfilename, std::ios_base::app); // Open in append mode
        for (int i = 0; i < NFREQ_PADDED; i++) {
          if (h_grid[i] == 0.0f) continue;
          pgfile << std::fixed << std::setprecision(6) << h_grid[i] << "  ";
          pgfile << std::fixed << std::setprecision(6) << h_pgram[(oidx-objstart)*NFREQ_PADDED + i];
          pgfile << "\n";
        }
        pgfile.close();
        std::cout << "Finished.\n";
      }
    }
  }


  // Perform final peak selection
  int Npeakcol_final = 3 * NPEAKS;
  std::vector<float> final_peaks(NSRC * Npeakcol_final, 0.0f);

  # pragma omp parallel for
  for (int i = 0; i < NSRC; i++) {

    // Slice the vectors
    std::vector<float>::const_iterator freq_first = all_peak_frequencies.begin() + i*Npeakcol;
    std::vector<float>::const_iterator freq_last = all_peak_frequencies.begin() + (i+1)*Npeakcol;
    std::vector<float>::const_iterator height_first = all_peak_heights.begin() + i*Npeakcol;
    std::vector<float>::const_iterator height_last = all_peak_heights.begin() + (i+1)*Npeakcol;
    std::vector<float>::const_iterator mean_first = all_pgram_means.begin() + i*NCHUNK_SINCOS;
    std::vector<float>::const_iterator mean_last = all_pgram_means.begin() + (i+1)*NCHUNK_SINCOS;
    std::vector<float>::const_iterator stdev_first = all_pgram_stdevs.begin() + i*NCHUNK_SINCOS;
    std::vector<float>::const_iterator stdev_last = all_pgram_stdevs.begin() + (i+1)*NCHUNK_SINCOS;
    std::vector<float> peak_freq_slice(freq_first, freq_last);
    std::vector<float> peak_height_slice(height_first, height_last);
    std::vector<float> pgram_mean_slice(mean_first, mean_last);
    std::vector<float> pgram_stdev_slice(stdev_first, stdev_last);

    // Calculate averages of the periodogram stats
    float mean_avg = mean(pgram_mean_slice);
    float stdev_avg = mean(pgram_stdev_slice);

    // Check that peaks were found for this object
    if (mean_avg < 0.0f) {
      for (int j = 0; j < NPEAKS; j++) {
        final_peaks[i*Npeakcol_final + 3*j + 0] = -1.0f;
        final_peaks[i*Npeakcol_final + 3*j + 1] = -1.0f;
        final_peaks[i*Npeakcol_final + 3*j + 2] = -1.0f;
      }
    }
    else {

      // Create vectors to store heights & indices of the top N highest peaks
      std::vector<float> xtop(NPEAKS); // Heights of highest peaks
      std::vector<int> itop(NPEAKS);   // Indices of highest peaks

      // Sort peaks in descending height order
      std::partial_sort_copy(
        peak_height_slice.begin(), 
        peak_height_slice.end(),
        xtop.begin(), 
        xtop.end(),
        std::greater<>()
      );

      // Get indices of highest peaks
      for (int j = 0; j < NPEAKS; j++) {
        ptrdiff_t pos = std::distance(
            peak_height_slice.begin(), 
            std::find(peak_height_slice.begin(), peak_height_slice.end(), xtop[j])
        );
        itop[j] = pos;
      }

      // Go through each peak and store in "final_peaks" vector
      for (int j = 0; j < NPEAKS; j++) {
        final_peaks[i*Npeakcol_final + 3*j + 0] = peak_freq_slice[itop[j]];          // Frequency
        final_peaks[i*Npeakcol_final + 3*j + 1] = xtop[j];                           // Height
        final_peaks[i*Npeakcol_final + 3*j + 2] = (xtop[j] - mean_avg) / stdev_avg;  // SNR
      }
      
      // Clear memory
      std::vector<float>().swap(xtop);
      std::vector<int>().swap(itop);
    }

    // Clear memory
    std::vector<float>().swap(peak_freq_slice);
    std::vector<float>().swap(peak_height_slice);
    std::vector<float>().swap(pgram_mean_slice);
    std::vector<float>().swap(pgram_stdev_slice);
  }


  // Save final peak selection to file
  std::ofstream peakfile;
  peakfile.open(peakfilename);
  for (int i = 0; i < NSRC; i++) {
    for (int j = 0; j < Npeakcol_final; j++) {
      peakfile << std::fixed << std::setprecision(4) << final_peaks[i*Npeakcol_final + j] << "  ";
    }
    peakfile << "\n";
  }
  peakfile.close();

  // (For testing purposes) Save ALL peaks to file
  // std::ofstream allpeaksfile;
  // allpeaksfile.open(allpeakfilename);
  // for (int i = 0; i < NSRC; i++) {
  //   for (int j = 0; j < Npeakcol; j++) {
  //     allpeaksfile << std::fixed << std::setprecision(4) << all_peak_frequencies[i*Npeakcol + j] << "  ";
  //     allpeaksfile << std::fixed << std::setprecision(4) << all_peak_heights[i*Npeakcol + j] << "  ";
  //   }
  //   allpeaksfile << "\n";
  // }
  // allpeaksfile.close();


  // Free device (GPU) memory
  cudaFree(d_pgram);
  cudaFree(d_flux);
  cudaFree(d_grid);
  cudaFree(d_time);
  cudaFree(d_sincos);

  // Free all vectors from host (CPU) memory
  std::vector<double>().swap(h_bjd);
  std::vector<unsigned long>().swap(h_psid);
  std::vector<float>().swap(h_time);
  std::vector<float>().swap(h_flux);
  std::vector<float>().swap(h_grid_full);
  std::vector<float>().swap(h_grid);
  std::vector<float>().swap(h_pgram);
  std::vector<float>().swap(all_peak_frequencies);
  std::vector<float>().swap(all_peak_heights);
  std::vector<float>().swap(all_pgram_means);
  std::vector<float>().swap(all_pgram_stdevs);
  std::vector<float>().swap(final_peaks);
  std::vector<int>().swap(freq_flags);

  // Get timestamp at end of program execution
  std::chrono::steady_clock::time_point t_prog_end = std::chrono::steady_clock::now();
  if (!quiet) {
    std::cout << "\n  Execution Times" << std::endl;
    std::cout << "  -------------------------------------" << std::endl;
    std::cout << "  Total Execution time = ";
    std::cout <<  std::chrono::duration_cast<std::chrono::seconds>(t_prog_end - t_prog_start).count();
    std::cout << " sec\n" << std::endl;
  }


  return 0;
}

