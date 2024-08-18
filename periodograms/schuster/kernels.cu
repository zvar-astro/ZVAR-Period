// Kernel functions for Schuster periodograms
//
// Author: Zach Vanderbosch (Caltech)


# include "kernels.h"


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
