#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <cmath>
#include <filesystem>

void logMessage(const std::string& logfile, const std::string& inputfile, const std::string& message);

void computeAmplitudes(const std::vector<float>& fluxes_no_baseline,
                       std::vector<float>& amplitudes,
                       unsigned long N_exp,
                       unsigned long N_src);

void computeAvgError(const std::vector<float>& flux_errs,
                     std::vector<float>& fluxes_avg_var,
                     unsigned long N_exp,
                     unsigned long N_src);

void saveObjectTimesAndFluxToCSV(const std::string& filename,
                                 const std::vector<float>& times,
                                 const std::vector<float>& fluxes_no_baseline,
                                 const std::vector<float>& flux_errs,
                                 unsigned long N_exp,
                                 unsigned long object_index);

void writeFluxStatisticsToCSV(const std::string& filename,
                              const std::vector<float>& flux_medians,
                              const std::vector<float>& flux_mads,
                              const std::vector<float>& amplitudes,
                              unsigned long objstart_periodograms,
                              bool is_first_chunk);

void writePeriodogramStatisticsToCSV(const std::string& base_filename,
                                     const std::vector<float>& periodogram_medians,
                                     const std::vector<float>& periodogram_mads,
                                     unsigned long objstart_periodograms,
                                     unsigned long N_src_per_batch,
                                     unsigned long N_samples,
                                     bool is_first_chunk);
#endif //DIAGNOSTICS_H