// Utility functions for opening and reading HDF5 files
//
// By: Zach Vanderbosch

#ifndef MATCHFILE_DUAL_H_
#define MATCHFILE_DUAL_H_

#include <string>
#include <vector>
#include <hdf5.h>
#include <numeric>
#include <iomanip>
#include <tuple>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <iostream>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Struct definitions

typedef struct exposures
{
  double bjd;
} exposures;

typedef struct sources
{
  unsigned long psid;
  float ra;
  float dec;
} sources;

typedef struct sourcedata
{
  float flux;
  float fluxErr;
} sourcedata;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Function definitions

// Function to check status of HDF5 reads
void check_status(int);

// Function to count number of rows in table
int get_number_of_rows(hid_t);

// Wrapper function to count number of rows in table
// given the HDF5 filename and the table name
int get_number_of_rows_wrapper(std::string, std::string);

// Function to read data from exposures table
int get_exposures(hid_t, double *, int, int, const char *field_name);

// Wrapper function to load BJD timestamps from HDF5 file
// given just the HDF5 filename rather than a table object
int get_exposures_wrapper(std::string, double *, int, int, const char *field_name);

// Function to read data from sources table
int get_sources(hid_t, unsigned long *, int, int, const char *field_name);

// Wrapper function to load source IDs from HDF5 file
// given just the HDF5 filename rather than a table object
int get_sources_wrapper(std::string, unsigned long *, int, int, const char *field_name);

// Function to read chunk of PANSTARRS IDs, RAs, and DECs from sources table
int get_sources_chunk(std::string, unsigned long *, float *, float *, unsigned long, unsigned long, const char *id_field_name, const char *ra_field_name, const char *dec_field_name);

// Function to read data from sourcedata table
int get_sourcedata(hid_t, float *, int, const char *field_name);

// Function to read chunk of data from sourcedata table
// int get_sourcedata_chunk(std::string, float*, float*, float*, unsigned long, unsigned long, unsigned long, unsigned long);
int get_sourcedata_chunk(
    std::string,                   // filename
    float *,                       // flux
    float *,                       // fluxErr
    float *,                       // ratioValid
    std::vector<unsigned char> &,  // keep
    unsigned long,                 // objstart
    unsigned long,                 // objnum
    unsigned long,                 // numentry
    const char *flux_field_name,   // Name of the field for flux values
    const char *fluxErr_field_name // Name of the field for flux error values
);

// Function to calculate median of flux values
float calcMedian(std::vector<float>, size_t);

// Function to calculate deviations b/t flux values and median
void calcDeviations(float *, float *, float, size_t);

// Function to normalize flux values
void normalizeLC(float *, int, int);

// Function to create a new directory for output files
std::string createOutputFilepath(std::string output_dir, std::string input_filename);

// Function to create a new HDF5 file to store periodogram results
void createResults(std::string, int, int, int);

// Function to append periodogram results to an existing HDF5 file
void appendResults(std::string, unsigned long *, float *, float *, float *, float *, float *, float *, float *, int, int, int, unsigned long);

// Function to find and flag deep drilling time stamps
std::vector<unsigned char> flagDeepDrilling(
    std::vector<double>, // time
    int,                 // N_d
    bool  // invert
);

// Function to load a sanitized time vector from an HDF5 file, and return the number of exposures and sources
std::tuple<unsigned int, unsigned int, std::vector<double>, std::vector<float>, std::vector<unsigned char>> initializeFrameData(const std::string &filename, const char *id_field_name, const char *time_field_name, bool invert);

#endif
