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
} sources;

typedef struct sourcedata {
  float flux;
} sourcedata;


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Function definitions

// Function to check status of HDF5 reads
void check_status(
  int // status
);


// Function to count number of rows in table
int get_number_of_rows(
  hid_t // dataset
);


// Wrapper function to count number of rows in table
// given the HDF5 filename and the table name
int get_number_of_rows_wrapper(
  std::string, // filename
  std::string  // tablename
);
    

// Function to read data from exposures table
int get_exposures(
  hid_t,   // table
  double*, // bjd
  int,     // begin
  int      // N
);


// Wrapper function to load BJD timestamps from HDF5 file
// given just the HDF5 filename rather than a table object
int get_exposures_wrapper(
  std::string, // filename
  double*,     // bjd
  int,         // begin
  int          // N
);


// Function to read data from sources table
int get_sources(
  hid_t,          // table
  unsigned long*, // psid
  int,            // begin
  int             // N
);


// Wrapper function to load source IDs from HDF5 file
// given just the HDF5 filename rather than a table object
int get_sources_wrapper(
  std::string,    // filename
  unsigned long*, // psid
  int,            // begin
  int             // N
);


// Function to read data from sourcedata table
int get_sourcedata(
  hid_t,  // table
  float*, // flux
  int     // N
);


// Function to read chunk of data from sourcedata table
int get_sourcedata_chunk(
  std::string,                // filename
  float*,                     // flux
  std::vector<unsigned char>, // keep
  unsigned long,              // objstart
  unsigned long,              // objnum
  unsigned long               // numentry
);


// Function to calculate median of flux values
float calcMedian(
  std::vector<float>, // flux
  size_t              // N
);


// Function to calculate deviations b/t flux values and median
void calcDeviations(
  float*, // deviations
  float*, // flux
  float,  // median
  size_t  // N
);


// Function to normalize flux values
void normalizeLC(
  float*, // flux
  int,    // N_s
  int     // N_d
);


// Function to find and flag deep drilling time stamps
std::vector<unsigned char> flagDeepDrilling(
  std::vector<double>,  // time
  int                   // N_d
);


#endif
