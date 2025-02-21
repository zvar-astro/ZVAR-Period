// Utility functions for opening and reading HDF5 files
//
// By: Zach Vanderbosch (Caltech)

#include <cstdlib>
#include <iostream>
#include <cassert>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "hdf5.h"
#include "hdf5_utils.h"


void check_status(int status) {
  /* 
  Function to check status of reading HDF5 files

  status = Status of HDF5 file read
  */
  assert(status == 0);
}


int get_number_of_rows(hid_t dataset) {
    
    /* 
    Calculate number of rows in the dataset 

    dataset = HDF5 dataset object
    */
    
    hid_t dspace;
    int ndims;

    dspace = H5Dget_space(dataset);
    ndims =  H5Sget_simple_extent_ndims(dspace);
    if (ndims != 1) return(0);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace,dims,NULL);
    
    return(dims[0]);
}

int get_number_of_rows_wrapper(std::string filename, 
                               std::string tablename) {

  /* 
  A wrapper function for the get_number_of_rows function.
  Does the same thing as get_number_of_rows, but as input
  this function only requires the HDF5 filename and 
  table name rather than an hid_t table object 

  filename  = Name of HDF5 file
  tablename = Name of HDF5 table
  */

  // Create file/table objects
  hid_t file, group, table;
  
  // Open HDF5 file
  file = H5Fopen(filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
  group = H5Gopen2(file,"/data",H5P_DEFAULT);
  table = H5Dopen2(group,tablename.c_str(),H5P_DEFAULT);

  // Get nmuber of rows with original function
  int nrows;
  nrows = get_number_of_rows(table);

  // Close HDF5 objects
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return nrows;
}


int get_exposures(hid_t table, double *bjd, int begin, int N) {

  /* 
  Function that grabs BJD timestamps
  from exposures table

  table = HDF5 table object
  bjd   = Array pointer to store BJD timestamps
  begin = Index at which to start storing data
  N     = Number of BJD timestamps
  */
  
  int status;
  exposures *data;
  
  // Read data from HDF5 table
  hid_t exposures_tid = H5Tcreate(H5T_COMPOUND,sizeof(exposures));
  H5Tinsert(exposures_tid,"bjd",HOFFSET(exposures,bjd),H5T_NATIVE_DOUBLE);
  data = (exposures *) malloc(sizeof(exposures)*N);
  status = H5Dread(table,exposures_tid,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
  
  // Populate BJD array with values
  for (int i = 0; i < N; i++) {
    bjd[i+begin] = data[i].bjd;
  }

  // Deallocate memory
  free(data);

  return status;
}


int get_exposures_wrapper(std::string filename,
                          double *bjd, int begin, int N) {

  /* 
  A wrapper function for the get_exposures function.
  Does the same thing as get_exposures, but as input
  this function only requires the HDF5 filename
  rather than an hid_t table object

  filename = Name of HDF5 file
  bjd      = Array pointer to store BJD timestamps
  begin    = Index at which to start storing data
  N        = Number of BJD timestamps to read
  */

  // Create file/table objects
  int status;
  hid_t file, group, table;
  
  // Open HDF5 file
  file = H5Fopen(filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
  group = H5Gopen2(file,"/data",H5P_DEFAULT);
  table = H5Dopen2(group,"/data/exposures",H5P_DEFAULT);
  
  // Get timestamp data
  status = get_exposures(table, bjd, begin, N);

  // Close HDF5 objects
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return status;
}


int get_sources(hid_t table, unsigned long *psid, int begin, int N) {
  
  /* 
  Function that grabs the Pan-STARRS IDs
  for all objects in the sources table
  
  table = HDF5 table object
  psid  = Array pointer to store Pan-STARRS IDs
  begin = Index at which to start storing data
  N     = Number of Pan-STARRS IDs
  */

  int status;
  sources *data;

  // Read data from HDF5 table
  hid_t sources_tid = H5Tcreate(H5T_COMPOUND,sizeof(sources));
  H5Tinsert(sources_tid,"gaia_id",HOFFSET(sources,psid),H5T_NATIVE_ULONG);
  data = (sources *) malloc(sizeof(sources)*N);
  status = H5Dread(table,sources_tid,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);

  // Populate psid array with values
  for (int i = 0; i < N; i++) {
    psid[i+begin] = data[i].psid;
  }
  
  // Deallocate memory
  free(data);
  
  return status;
}


int get_sources_wrapper(std::string filename,
                        unsigned long *psid, int begin, int N) {

  /* 
  A wrapper function for the get_sources function.
  Does the same thing as get_sources, but as input
  this function only requires the HDF5 filename
  rather than an hid_t table object

  filename = Name of HDF5 file
  psid     = Array pointer to store Pan-STARRS IDs
  begin    = Index at which to start storing data
  N        = Number of Pan-STARRS IDs to read
  */

  // Create file/table objects
  int status;
  hid_t file, group, table;
  
  // Open HDF5 file
  file = H5Fopen(filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
  group = H5Gopen2(file,"/data",H5P_DEFAULT);
  table = H5Dopen2(group,"/data/sources",H5P_DEFAULT);
  
  // Get timestamp data
  status = get_sources(table, psid, begin, N);

  // Close HDF5 objects
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return status;
}


int get_sourcedata(hid_t table, float *flux, int N) {
  
  /* 
  Function that grabs flux values for all
  sources from the sourcedata table

  table = HDF5 table object
  flux  = Array pointer to store flux values
  N     = Number of flux values per source
  */

  int status;
  sourcedata *data;
  
  // Read data from HDF5 file
  hid_t sourcedata_tid = H5Tcreate(H5T_COMPOUND, sizeof(sourcedata));
  H5Tinsert(sourcedata_tid,"flux",HOFFSET(sourcedata,flux),H5T_NATIVE_FLOAT);
  data = (sourcedata *) malloc(sizeof(sourcedata)*N);
  status = H5Dread(table,sourcedata_tid,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);

  // Populate flux array with values
  for (int i = 0; i < N; i++) {
    if (std::isnan(data[i].flux)) {
      flux[i] = 0.0f;
    }
    else {
      flux[i] = data[i].flux;
    }
  }

  // Deallocate memory
  free(data);

  return status;
}



int get_sourcedata_chunk(std::string filename, 
                         float *flux, 
                         std::vector<unsigned char> keep,
                         unsigned long objstart, 
                         unsigned long objnum, 
                         unsigned long numentry) {

  /* 
  Function that reads a contiguous chunk
  of data from a light curve HDF5's
  "sourcedata" table 
  
  filename = Name of HDF5 file
  flux     = Array pointer to store flux values
  keep     = Array indicating non-deep drilling time stamps (1=keep, 0=ignore)
  objstart = Object number at which to start reading data
  objnum   = Number of objects to read
  numentry = Number of entries reserved per source within flux
             including padding
  */
  
  // Create some HDF5 parameters
  int status;
  sourcedata *data;
  hid_t file, group, table;
  hid_t dataspace, memspace;

  // Get data dimensions from the keep vector
  int N = keep.size();  // Original number of exposures per source

  // Open HDF5 files
  file = H5Fopen(filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
  group = H5Gopen2(file,"/data",H5P_DEFAULT);

  // Open table
  table = H5Dopen2(group,"/data/sourcedata",H5P_DEFAULT);

  // Define hyperslab in the dataset
  dataspace = H5Dget_space(table);
  hsize_t offset[] = {objstart * N};
  hsize_t count[] = {objnum * N};
  status = H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL);

  // Define memory hyperslab
  hsize_t dim[] = {objnum * N};
  memspace = H5Screate_simple(1,dim,NULL);
  hsize_t offset_out[] = {0};
  hsize_t count_out[] = {objnum * N};
  status = H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);

  // Read data from table
  hid_t sourcedata_tid = H5Tcreate(H5T_COMPOUND,sizeof(sourcedata));
  H5Tinsert(sourcedata_tid,"flux",HOFFSET(sourcedata,flux),H5T_NATIVE_FLOAT);
  data = (sourcedata *) malloc(sizeof(sourcedata)*objnum*N);
  status = H5Dread(table,sourcedata_tid,memspace,dataspace,H5P_DEFAULT,data);

  // Populate flux array with values, skipping over deep-drilling epochs
  int k; // Points to index within flux
  for (int i = 0; i < objnum; i++) {
    k = 0; // Reset k-index for each source
    for (int j = 0; j < N; j++) {
      if (keep[j] == 1) { // Ignore deep-drilling epochs
        flux[i*numentry+k] = data[i*N+j].flux;
        k++;
      }
    }
  }

  // Deallocate memory
  free(data);
  H5Sclose(memspace);

  // Close files
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return status;
}



float calcMedian(std::vector<float> flux, size_t N) {

  /*
  Function to calculation median of flux values
  
  flux = Vector containing flux values
  N    = Number of flux values in vector
  */
  
  float median;

  std::sort(flux.begin(), flux.end());
  if (N % 2 == 0) median = (flux[N/2-1] + flux[N/2]) / 2.0f;
  else median = flux[N/2];

  return median;
}


void calcDeviations(float *deviations, float *flux, float median, size_t N) {

  /* 
  Calculate absolute deviations between flux
  values and the median
  
  dev    = Deviations between flux and median
  flux   = Flux values 
  median = Median of flux values
  N      = Number of flux values
  */

  for (int i = 0; i < N; i++) {
    deviations[i] = std::abs(flux[i] - median);
  }
}


void normalizeLC(float *flux, int N_s, int N_d) {

  /* 
  Function to median-zero and normalize the flux
  values for each source.

  flux = Array of flux values for all source
  N_s  = Number of sources
  N_d  = Number of flux values per source
  */
  
  size_t size;
  float median, MAD;
  std::vector<float> nzflux;
  std::vector<float> deviations;
  
  // Iterate over each source
  for (int i = 0; i < N_s; i++) {

    // First populate vector with non-zero (non-NaN) values
    for (int j = 0; j < N_d; j++) {
      if (!std::isnan(flux[i*N_d+j])) {
        nzflux.push_back(flux[i*N_d+j]);
        deviations.push_back(0.0f);
      }
    }

    // Calculate median and median absolute deviation (MAD)
    size = nzflux.size();
    if (size > 0) {
      median = calcMedian(nzflux, size);
      calcDeviations(deviations.data(), nzflux.data(), median, size);
      MAD = calcMedian(deviations, size);
      if (MAD == 0.0f) MAD = 1.0f;
    }
    else {
      median = 0.0f;
      MAD = 1.0f;
    }
    
    // Now subtract median and normalize by MAD
    for (int j = 0; j < N_d; j++) {
      if (std::isnan(flux[i*N_d+j])) {
        flux[i*N_d+j] = 0.0f;
      }
      else {
        flux[i*N_d+j] = (flux[i*N_d+j] - median) / MAD;
      }
    }
    
    // Erase elements from nzflux and deviations
    for(int j = size-1; j >= 0; j--) {
      nzflux.erase(nzflux.begin() + j); 
      deviations.erase(deviations.begin() + j);
    }
  }

  // Clear vectors from memory
  std::vector<float>().swap(nzflux);
  std::vector<float>().swap(deviations);
}



std::vector<unsigned char> flagDeepDrilling(std::vector<double> time, int N_d) {

  /* 
  Function to identify any timestamps within light
  curve associated with ZTF deep drilling observations

  time = Array of times in Julian Date (JD) format
  N_d = Number of exposures per source
  */

  // Get all unique integer JDs when data were acquired
  std::vector<int> jd;
  std::vector<int> unique_jd;
  for (int i = 0; i < N_d; i++) {
    jd.push_back(std::floor(time[i]));
    unique_jd.push_back(std::floor(time[i]));
  }
  std::sort(unique_jd.begin(), unique_jd.end());
  unique_jd.erase(std::unique(unique_jd.begin(), unique_jd.end()), unique_jd.end());
  int Njd = unique_jd.size();

  // Create boolean vector to indicate which time stamps should be kept
  std::vector<unsigned char> keep(N_d, 1);

  // Create some vectors to populate during each iteration
  std::vector<int> indices;
  std::vector<double> time_slice;

  // Iterate over unique days and find number of observations per day
  int count = 0;
  for (int i = 0; i < Njd; i++) {

    // Count how many time stamps fall within a given day
    count = std::count(jd.begin(), jd.end(), unique_jd[i]);

    // Get indices of all observations on this day
    for (int j = 0; j < N_d; j++) {
      if (std::floor(time[j]) == unique_jd[i]) {
        indices.push_back(j);
        time_slice.push_back(time[j]);
      }
    }

    // When enough data points are present, check time sampling
    // Using 20 as the limit in order to skip over the triplet
    // program observations, which will always have no more than
    // 15 observations per field per night
    if (count >= 20) {

      // Get differences between successive values
      std::adjacent_difference(
        time_slice.begin(), 
        time_slice.end(), 
        time_slice.begin()
      );

      // Set first element to equal second element
      time_slice[0] = time_slice[1];

      // Determine whether a time stamp is deep drilling
      for (int j = 0; j < count; j++) {
        if (time_slice[j] < 0.003472222) { // i.e. < 5-minutes
          keep[indices[j]] = 0;
        }
      }
    }

    // Erase vector values
    indices.clear();
    time_slice.clear();
  }

  // Clear memory
  std::vector<int>().swap(jd);
  std::vector<int>().swap(unique_jd);
  std::vector<int>().swap(indices);
  std::vector<double>().swap(time_slice);

  return keep;
}
