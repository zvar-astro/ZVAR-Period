#ifndef POLYFIT_CUH
#define POLYFIT_CUH

#include <iostream>
#include <gpufit.h>

/**
 * @brief Fits N_curves polynomial models to N_curves light curves of identical length N_dat and subtracts the model from the data.
 * This function uses the Gpufit library to fit the polynomial model to the data. The polynomial model is defined as:
 * modelID = 13: Quadratic (2 degree)
 * modelID = 14: Cubic (3 degree)
 * modelID = 15: 7 degree
 * modelID = 16: 15 degree
 * 
 * @param times The times of the light curves. The length of this array is N_dat.
 * @param y The light curves to fit. The length of this array is N_dat * N_curves in row-major order (each light curve is stored in a row).
 * @param dy The uncertainties of the light curves. The length of this array is N_dat * N_curves in row-major order (each light curve is stored in a row).
 * @param N_dat The length of each light curve.
 * @param N_curves The number of light curves.
 * @param modelID The ID of the polynomial model to fit.
 * @return float* The light curves with the baseline subtracted. The length of this array is N_dat * N_curves in row-major order (each light curve is stored in a row).
 * 
 * @throws std::invalid_argument If an invalid model ID is provided.
*/
std::pair<float*, float> removeBaseline(float* times, float* y, float* dy, int N_dat, int N_curves, int modelID);

#endif // POLYFIT_CUH