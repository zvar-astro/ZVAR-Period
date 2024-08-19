// Header file for Schuster kernels.cu
//
// Author: Zach Vanderbosch (Caltech)


#include <cmath>
#include <thrust/complex.h>


__global__ void sincosCalc(
  const float*,            // time,
  const float*,            // grid,
  thrust::complex<float>*, // sincos,
  const float,             // pi,
  long int                 // N_f
);


__global__ void pgramCalc(
  const float*,            // flux
  thrust::complex<float>*, // sincos
  float*,                  // pgram
  long int,                // N_d
  long int                 // N_f
);


__global__ void pgramStats(
  float*,   // pgram
  float*,   // pg_mean
  float*,   // pg_stdev
  long int, // N_f
  long int  // N_f_paddded
);
