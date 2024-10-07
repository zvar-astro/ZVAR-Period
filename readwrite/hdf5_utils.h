// Utility functions for opening and reading HDF5 files
//
// By: Zach Vanderbosch

#include <string>
#include <vector>
#include <hdf5.h>

#ifndef MATCHFILE_DUAL_H_
#define MATCHFILE_DUAL_H_

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Struct definitions

typedef struct exposures {
  double bjd;
} exposures;

typedef struct sources {
  unsigned long psid;
  float ra;
  float dec;
} sources;

typedef struct sourcedata {
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
int get_exposures(hid_t, double*, int, int);


// Wrapper function to load BJD timestamps from HDF5 file
// given just the HDF5 filename rather than a table object
int get_exposures_wrapper(std::string, double*, int, int);


// Function to read data from sources table
int get_sources(hid_t, unsigned long*, int, int);


// Wrapper function to load source IDs from HDF5 file
// given just the HDF5 filename rather than a table object
int get_sources_wrapper(std::string, unsigned long*, int, int);


// Function to read chunk of PANSTARRS IDs, RAs, and DECs from sources table
int get_sources_chunk(std::string, unsigned long*, float*, float*, unsigned long, unsigned long);


// Function to read data from sourcedata table
int get_sourcedata(hid_t, float*, int);


// Function to read chunk of data from sourcedata table
//int get_sourcedata_chunk(std::string, float*, float*, float*, unsigned long, unsigned long, unsigned long, unsigned long);
int get_sourcedata_chunk(
  std::string,                 // filename
  float*,                      // flux
  float*,                      // fluxErr
  float*,                      // ratioValid
  std::vector<unsigned char>&, // keep
  unsigned long,               // objstart
  unsigned long,               // objnum
  unsigned long                // numentry
);

// Function to calculate median of flux values
float calcMedian(std::vector<float>, size_t);


// Function to calculate deviations b/t flux values and median
void calcDeviations(float*, float*, float, size_t);


// Function to normalize flux values
void normalizeLC(float*, int, int);


// Function to create a new HDF5 file to store periodogram results
void writeResults(std::string, int, int, int);


// Function to append periodogram results to an existing HDF5 file
void appendResults(std::string, unsigned long*, float*, float*, float*, float*, float*, float*, float*, int, int, int, unsigned long);


// Function to find and flag deep drilling time stamps
std::vector<unsigned char> flagDeepDrilling(
  std::vector<double>,  // time
  int                   // N_d
);

#endif
