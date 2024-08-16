// Utility functions for opening and reading HDF5 files
//
// By: Zach Vanderbosch

#include <cstdlib>
#include <iostream>
#include <cassert>
#include <string>
#include <cmath>
#include <algorithm>
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


int get_sources_chunk(std::string filename, unsigned long *psids, float *ra, float *dec, unsigned long objstart, unsigned long objnum) {

  /* 
  Function to read a contiguous chunk of data from the sources table
  Used to return Pan-STARRS IDs for a subset of objects
  
  filename = Name of HDF5 file
  psids    = Array pointer to store Pan-STARRS IDs
  objstart = Object number at which to start reading data
  objnum   = Number of objects to read
  */

  int status;
  sources *data;
  hid_t file, group, table;
  hid_t dataspace, memspace;

  // Open HDF5 files
  file = H5Fopen(filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
  group = H5Gopen2(file,"/data",H5P_DEFAULT);

  // Open table
  table = H5Dopen2(group,"/data/sources",H5P_DEFAULT);

  // Define hyperslab in the dataset
  dataspace = H5Dget_space(table);
  hsize_t offset[] = {objstart};
  hsize_t count[] = {objnum};
  status = H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL);

  // Define memory hyperslap
  hsize_t dim[] = {objnum};
  memspace = H5Screate_simple(1,dim,NULL);
  hsize_t offset_out[] = {0};
  hsize_t count_out[] = {objnum};
  status = H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);

  // Read data from table
  hid_t sources_tid = H5Tcreate(H5T_COMPOUND,sizeof(sources));
  H5Tinsert(sources_tid,"gaia_id",HOFFSET(sources,psid),H5T_NATIVE_ULONG);
  H5Tinsert(sources_tid,"ra",HOFFSET(sources,ra),H5T_NATIVE_FLOAT);
  H5Tinsert(sources_tid,"decl",HOFFSET(sources,dec),H5T_NATIVE_FLOAT);
  data = (sources *) malloc(sizeof(sources)*objnum);
  status = H5Dread(table,sources_tid,memspace,dataspace,H5P_DEFAULT,data);

  // Populate psid array with values
  for (int i = 0; i < objnum; i++) {
    psids[i] = data[i].psid;
    ra[i] = data[i].ra;
    dec[i] = data[i].dec;
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



int get_sourcedata_chunk(std::string filename, float *flux, float* fluxErr, float* ratioValid, unsigned long objstart, 
  unsigned long objnum, unsigned long N, unsigned long numentry) {

  /*
  Function that reads a contiguous chunk
  of data from the sourcedata table

  filename   = Name of HDF5 file
  flux       = Array pointer to store flux values
  fluxErr    = Array pointer to store flux errors
  ratioValid = Array pointer to store what percentage of a light curve is valid
  objstart   = Object number at which to start reading data
  objnum     = Number of objects to read
  N          = Number of observations per object
  numentry   = Number of entries reserved per source within flux, including padding
  */

  int status;
  sourcedata *data;
  hid_t file, group, table;
  hid_t dataspace, memspace;

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

  // Define memory hyperslap
  hsize_t dim[] = {objnum * N};
  memspace = H5Screate_simple(1,dim,NULL);
  hsize_t offset_out[] = {0};
  hsize_t count_out[] = {objnum * N};
  status = H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);

  // Read data from table
  hid_t sourcedata_tid = H5Tcreate(H5T_COMPOUND,sizeof(sourcedata));
  H5Tinsert(sourcedata_tid,"flux",HOFFSET(sourcedata,flux),H5T_NATIVE_FLOAT);
  H5Tinsert(sourcedata_tid,"flux_err",HOFFSET(sourcedata,fluxErr),H5T_NATIVE_FLOAT);
  data = (sourcedata *) malloc(sizeof(sourcedata)*objnum*N);
  status = H5Dread(table,sourcedata_tid,memspace,dataspace,H5P_DEFAULT,data);

  // Populate flux array with values
  for (int i = 0; i < objnum; i++) {
    int validCount = 0;
    for (int j = 0; j < N; j++) {
      if (std::isnan(data[i*N+j].flux)) {
        flux[i*numentry+j] = 0.0f;
        fluxErr[i*numentry+j] = INFINITY;
      }
      else {
        flux[i*numentry+j] = data[i*N+j].flux;
        fluxErr[i*numentry+j] = data[i*N+j].fluxErr;
        validCount++;
      }
    }
    ratioValid[i] = static_cast<float>(validCount) / static_cast<float>(N);
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


int get_sourcedata_byobj_dualband(std::string filename1, 
                                  std::string filename2, 
                                  std::vector<unsigned long> objIDs, 
                                  float *flux, 
                                  int objstart, int objnum,
                                  unsigned long N1, 
                                  unsigned long N2,
                                  int numentry) {
  /* 

  Function to get sourcedata for a list of object IDs 
  from two HDF5 files for different filters (e.g. g and r)
  but with the same field, ccd, and quad.

  filename1 = Name of first  HDF5 file (e.g. g-band file)
  filename2 = Name of second HDF5 file (e.g. r-band file)
  objIDs    = Vector containing unique PS1 object IDs
  flux      = Array pointer to store flux values
  objstart  = Starting index within objIDs to extract flux values for
  objnum    = Number of objects to read
  N1        = Number of observations per source in first file
  N2        = Number of observations per source in second file
  numentry  = Number of entries reserved per source within flux (including padding)

  */

  int status, nsrc1, nsrc2;
  hid_t file1, file2, group1, group2;
  hid_t table_src_1, table_dat_1;
  hid_t table_src_2, table_dat_2;
  hid_t dataspace, memspace;
  sourcedata *data;

  // Open HDF5 files
  file1 = H5Fopen(filename1.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
  file2 = H5Fopen(filename2.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
  group1 = H5Gopen2(file1,"/data",H5P_DEFAULT);
  group2 = H5Gopen2(file2,"/data",H5P_DEFAULT);

  // Open necessary tables
  table_src_1 = H5Dopen2(group1,"/data/sources",H5P_DEFAULT);
  table_src_2 = H5Dopen2(group2,"/data/sources",H5P_DEFAULT);
  table_dat_1 = H5Dopen2(group1,"/data/sourcedata",H5P_DEFAULT);
  table_dat_2 = H5Dopen2(group2,"/data/sourcedata",H5P_DEFAULT);

  // Get number of sources per table
  nsrc1 = get_number_of_rows(table_src_1);
  nsrc2 = get_number_of_rows(table_src_2);

  // Create vectors to store IDs
  std::vector<unsigned long> psids_1(nsrc1);
  std::vector<unsigned long> psids_2(nsrc2);

  // Read source IDs from source table
  status = get_sources(table_src_1, psids_1.data(), 0, nsrc1);
  check_status(status);
  status = get_sources(table_src_2, psids_2.data(), 0, nsrc2);
  check_status(status);

  // Iterate over object IDs and extract data
  unsigned long obj, objidx1, objidx2;
  int success1, success2;

  for (int i = 0; i < objnum; i++) {
    
    // Find the indices of each PS1 source within HDF5 files (if it exists)
    obj = objIDs[objstart + i];
    success1 = 0;
    success2 = 0;
    for (objidx1 = 0; objidx1 < nsrc1; objidx1++) {
      if (psids_1[objidx1] == obj) {
        success1 = 1;
        break;
      }
    }
    for (objidx2 = 0; objidx2 < nsrc2; objidx2++) {
      if (psids_2[objidx2] == obj) {
        success2 = 1;
        break;
      }
    }

    // Get data from first file
    if (success1 == 1) {

      // Define hyperslab in the dataset
      dataspace = H5Dget_space(table_dat_1);
      hsize_t offset[] = {objidx1*N1};
      hsize_t count[] = {N1};
      status = H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL);

      // Define memory hyperslap
      hsize_t dim[] = {N1};
      memspace = H5Screate_simple(1,dim,NULL);
      hsize_t offset_out[] = {0};
      hsize_t count_out[] = {N1};
      status = H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);

      // Read data from table
      hid_t sourcedata_tid = H5Tcreate(H5T_COMPOUND,sizeof(sourcedata));
      H5Tinsert(sourcedata_tid,"flux",HOFFSET(sourcedata,flux),H5T_NATIVE_FLOAT);
      data = (sourcedata *) malloc(sizeof(sourcedata)*N1);
      status = H5Dread(table_dat_1,sourcedata_tid,memspace,dataspace,H5P_DEFAULT,data);

      // Populate flux array with values
      for (int j = 0; j < N1; j++) {
        if (std::isnan(data[j].flux)) {
          flux[i*numentry+j] = 0.0f;
        }
        else {
          flux[i*numentry+j] = data[j].flux;
        }
      }

      // Deallocate memory
      free(data);
      status = H5Sclose(memspace);
    }

    // Get data from second file
    if (success2 == 1) {

      // Define hyperslab in the dataset
      dataspace = H5Dget_space(table_dat_2);
      hsize_t offset[] = {objidx2 * N2};
      hsize_t count[] = {N2};
      status = H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL);

      // Define memory hyperslap
      hsize_t dim[] = {N2};
      memspace = H5Screate_simple(1,dim,NULL);
      hsize_t offset_out[] = {0};
      hsize_t count_out[] = {N2};
      status = H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL);

      // Read data from table
      hid_t sourcedata_tid = H5Tcreate(H5T_COMPOUND,sizeof(sourcedata));
      H5Tinsert(sourcedata_tid,"flux",HOFFSET(sourcedata,flux),H5T_NATIVE_FLOAT);
      data = (sourcedata *) malloc(sizeof(sourcedata)*N2);
      status = H5Dread(table_dat_2,sourcedata_tid,memspace,dataspace,H5P_DEFAULT,data);

      // Populate flux array with values
      for (int j = 0; j < N2; j++) {
        if (std::isnan(data[j].flux)) {
          flux[i*numentry+N1+j] = 0.0f;
        }
        else {
          flux[i*numentry+N1+j] = data[j].flux;
        }
      }

      // Deallocate memory
      free(data);
      status = H5Sclose(memspace);
    }
  }

  // Close HDF5 table and files
  H5Dclose(table_dat_2);
  H5Dclose(table_dat_1);
  H5Dclose(table_src_2);
  H5Dclose(table_src_1);
  H5Gclose(group2);
  H5Gclose(group1);
  H5Fclose(file2);
  H5Fclose(file1);

  // Clear vectors
  std::vector<unsigned long>().swap(psids_1);
  std::vector<unsigned long>().swap(psids_2);

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
      if (flux[i*N_d+j] != 0.0f) {
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
    }
    else {
      median = 0.0f;
      MAD = 1.0f;
    }

    // Now subtract median and normalize by MAD
    for (int j = 0; j < N_d; j++) {
      flux[i*N_d+j] = (flux[i*N_d+j] - median) / MAD;
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

// void normalizeLC(float *flux, float *fluxErr, int N_s, int N_d) {

//   /* 
//   Function to median-zero and normalize the flux
//   values for each source.

//   flux = Array of flux values for all source
//   fluxErr = Array of flux errors for all sources
//   N_s  = Number of sources
//   N_d  = Number of flux values per source
//   */
  
//   size_t size;
//   float median, MAD;
//   std::vector<float> nzflux;
//   std::vector<float> deviations;
  
//   // Iterate over each source
//   for (int i = 0; i < N_s; i++) {

//     // First populate vector with non-zero (non-NaN) values
//     for (int j = 0; j < N_d; j++) {
//       if (flux[i*N_d+j] != 0.0f) {
//         nzflux.push_back(flux[i*N_d+j]);
//         deviations.push_back(0.0f);
//       }
//     }

//     // Calculate median and median absolute deviation (MAD)
//     size = nzflux.size();
//     if (size > 0) {
//       median = calcMedian(nzflux, size);
//       calcDeviations(deviations.data(), nzflux.data(), median, size);
//       MAD = calcMedian(deviations, size);
//     }
//     else {
//       median = 0.0f;
//       MAD = 1.0f;
//     }
    
//     // Now subtract median and normalize by MAD
//     for (int j = 0; j < N_d; j++) {
//       flux[i*N_d+j] = (flux[i*N_d+j] - median) / MAD;
//     }
    
//     // Erase elements from nzflux and deviations
//     for(int j = size-1; j >= 0; j--) {
//       nzflux.erase(nzflux.begin() + j); 
//       deviations.erase(deviations.begin() + j);
//     }
//   }

//   // Clear vectors from memory
//   std::vector<float>().swap(nzflux);
//   std::vector<float>().swap(deviations);
// }


void normalizeLC_dualband(float *flux, int N_s, int N1, int N2, int numentry) {

  /* 
  Function to median-zero and normalize the 
  flux for each source, when flux consists of
  data for two different filters.

  flux     = Array pointer containing flux values
  N_s      = Number of sources
  N1       = Number of observations per source in first band
  N2       = Number of observations per source in second band
  numentry = Number of entries reserved per source within flux
             including padding
  */

  size_t size1, size2;
  float median1, median2, MAD1, MAD2;
  std::vector<float> nzflux1, nzflux2;
  std::vector<float> deviations1, deviations2;
  
  // Iterate over each source
  for (int i = 0; i < N_s; i++) {

    // Iterate over first-band flux values and save non-zero vals
    for (int j = 0; j < N1; j++) {
      if (flux[i*numentry+j] != 0.0f) {
        nzflux1.push_back(flux[i*numentry+j]);
        deviations1.push_back(0.0f);
      }
    }
    // Iterate over second-band to do same as above
    for (int j = 0; j < N2; j++) {
      if (flux[i*numentry+N1+j] != 0.0f) {
        nzflux2.push_back(flux[i*numentry+N1+j]);
        deviations2.push_back(0.0f);
      }
    }
    
    // Calculate median and median absolute deviation (MAD)
    size1 = nzflux1.size();
    size2 = nzflux2.size();
    if (size1 > 0) {
      median1 = calcMedian(nzflux1, size1);
      calcDeviations(deviations1.data(), nzflux1.data(), median1, size1);
      MAD1 = calcMedian(deviations1, size1);
    }
    else {
      median1 = 0.0f;
      MAD1 = 1.0f;
    }
    if (size2 > 0) {
      median2 = calcMedian(nzflux2, size2);
      calcDeviations(deviations2.data(), nzflux2.data(), median2, size2);
      MAD2 = calcMedian(deviations2, size2);
    }
    else {
      median2 = 0.0f;
      MAD2 = 1.0f;
    }
    
    // Now subtract median and normalize by MAD
    for (int j = 0; j < N1; j++) {
      flux[i*numentry+j] = (flux[i*numentry+j] - median1) / MAD1;
    }
    for (int j = 0; j < N2; j++) {
      flux[i*numentry+N1+j] = (flux[i*numentry+N1+j] - median2) / MAD2;
    }
    
    // Erase elements from nzflux and deviations
    for(int j = size1-1; j >= 0; j--) {
      nzflux1.erase(nzflux1.begin() + j); 
      deviations1.erase(deviations1.begin() + j);
    }
    for(int j = size2-1; j >= 0; j--) {
      nzflux2.erase(nzflux2.begin() + j); 
      deviations2.erase(deviations2.begin() + j);
    }
  }

  // Clear vectors from memory
  std::vector<float>().swap(nzflux1);
  std::vector<float>().swap(nzflux2);
  std::vector<float>().swap(deviations1);
  std::vector<float>().swap(deviations2);

}

void writeResults(std::string filename, int N_curves, int total_curves, int N_samples, int N_peaks) {
    // Create a new HDF5 file
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Define the maximum dimensions for each dataset
    hsize_t max_dims_psids[1] = {static_cast<hsize_t>(total_curves)};
    hsize_t max_dims_ra[1] = {static_cast<hsize_t>(total_curves)};
    hsize_t max_dims_dec[1] = {static_cast<hsize_t>(total_curves)};
    hsize_t max_dims_valid[1] = {static_cast<hsize_t>(total_curves)};
    hsize_t max_dims_bestFreqs[1] = {static_cast<hsize_t>(total_curves) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks)};
    hsize_t max_dims_significance[1] = {static_cast<hsize_t>(total_curves) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks)};
    hsize_t max_dims_means[1] = {static_cast<hsize_t>(total_curves) * static_cast<hsize_t>(N_samples)};
    hsize_t max_dims_stds[1] = {static_cast<hsize_t>(total_curves) * static_cast<hsize_t>(N_samples)};

    // Create dataspaces for each dataset with the maximum dimensions
    hid_t dataspace_psids_max = H5Screate_simple(1, max_dims_psids, NULL);
    hid_t dataspace_ra_max = H5Screate_simple(1, max_dims_ra, NULL);
    hid_t dataspace_dec_max = H5Screate_simple(1, max_dims_dec, NULL);
    hid_t dataspace_valid_max = H5Screate_simple(1, max_dims_valid, NULL);
    hid_t dataspace_bestFreqs_max = H5Screate_simple(1, max_dims_bestFreqs, NULL);
    hid_t dataspace_significance_max = H5Screate_simple(1, max_dims_significance, NULL);
    hid_t dataspace_means_max = H5Screate_simple(1, max_dims_means, NULL);
    hid_t dataspace_stds_max = H5Screate_simple(1, max_dims_stds, NULL);

    // Create datasets
    hid_t psids_dataset_id = H5Dcreate(file_id, "psids", H5T_NATIVE_ULONG, dataspace_psids_max, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t ra_dataset_id = H5Dcreate(file_id, "ra", H5T_NATIVE_FLOAT, dataspace_ra_max, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dec_dataset_id = H5Dcreate(file_id, "dec", H5T_NATIVE_FLOAT, dataspace_dec_max, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t valid_dataset_id = H5Dcreate(file_id, "valid", H5T_NATIVE_FLOAT, dataspace_valid_max, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t bestFreqs_dataset_id = H5Dcreate(file_id, "bestFreqs", H5T_NATIVE_FLOAT, dataspace_bestFreqs_max, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t significance_dataset_id = H5Dcreate(file_id, "significance", H5T_NATIVE_FLOAT, dataspace_significance_max, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t means_dataset_id = H5Dcreate(file_id, "means", H5T_NATIVE_FLOAT, dataspace_means_max, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t stds_dataset_id = H5Dcreate(file_id, "stds", H5T_NATIVE_FLOAT, dataspace_stds_max, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Close resources
    H5Sclose(dataspace_psids_max);
    H5Sclose(dataspace_ra_max);
    H5Sclose(dataspace_dec_max);
    H5Sclose(dataspace_valid_max);
    H5Sclose(dataspace_bestFreqs_max);
    H5Sclose(dataspace_significance_max);
    H5Dclose(psids_dataset_id);
    H5Dclose(bestFreqs_dataset_id);
    H5Dclose(significance_dataset_id);
    H5Dclose(means_dataset_id);
    H5Dclose(stds_dataset_id);
    H5Fclose(file_id);
}

void appendResults(std::string filename, unsigned long* psids, float* ra, float* dec, float* valid, float* bestFreqs, float* significance, float* means, float* stdevs, int N_curves, int N_samples, int N_peaks, unsigned long offset) {
    // Open the existing HDF5 file
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    // Open the datasets
    hid_t psids_dataset_id = H5Dopen(file_id, "psids", H5P_DEFAULT);
    hid_t ra_dataset_id = H5Dopen(file_id, "ra", H5P_DEFAULT);
    hid_t dec_dataset_id = H5Dopen(file_id, "dec", H5P_DEFAULT);
    hid_t valid_dataset_id = H5Dopen(file_id, "valid", H5P_DEFAULT);
    hid_t bestFreqs_dataset_id = H5Dopen(file_id, "bestFreqs", H5P_DEFAULT);
    hid_t significance_dataset_id = H5Dopen(file_id, "significance", H5P_DEFAULT);
    hid_t means_dataset_id = H5Dopen(file_id, "means", H5P_DEFAULT);
    hid_t stds_dataset_id = H5Dopen(file_id, "stds", H5P_DEFAULT);

    // Define the dimensions for the data to be written
    hsize_t dims_psids[1] = {static_cast<hsize_t>(N_curves)};
    hsize_t dims_ra[1] = {static_cast<hsize_t>(N_curves)};
    hsize_t dims_dec[1] = {static_cast<hsize_t>(N_curves)};
    hsize_t dims_valid[1] = {static_cast<hsize_t>(N_curves)};
    hsize_t dims_bestFreqs[1] = {static_cast<hsize_t>(N_curves) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks)};
    hsize_t dims_significance[1] = {static_cast<hsize_t>(N_curves) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks)};
    hsize_t dims_means[1] = {static_cast<hsize_t>(N_curves) * static_cast<hsize_t>(N_samples)};
    hsize_t dims_stds[1] = {static_cast<hsize_t>(N_curves) * static_cast<hsize_t>(N_samples)};

    // Define the offset for the hyperslab
    hsize_t offset_psids[1] = {static_cast<hsize_t>(offset)};
    hsize_t offset_ra[1] = {static_cast<hsize_t>(offset)};
    hsize_t offset_dec[1] = {static_cast<hsize_t>(offset)};
    hsize_t offset_valid[1] = {static_cast<hsize_t>(offset)};
    hsize_t offset_bestFreqs[1] = {static_cast<hsize_t>(offset) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks)};
    hsize_t offset_significance[1] = {static_cast<hsize_t>(offset) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks)};
    hsize_t offset_means[1] = {static_cast<hsize_t>(offset) * static_cast<hsize_t>(N_samples)};
    hsize_t offset_stds[1] = {static_cast<hsize_t>(offset) * static_cast<hsize_t>(N_samples)};

    // Define the file dataspace and select the hyperslab
    hid_t file_dataspace_psids = H5Dget_space(psids_dataset_id);
    H5Sselect_hyperslab(file_dataspace_psids, H5S_SELECT_SET, offset_psids, NULL, dims_psids, NULL);

    hid_t file_dataspace_ra = H5Dget_space(ra_dataset_id);
    H5Sselect_hyperslab(file_dataspace_ra, H5S_SELECT_SET, offset_ra, NULL, dims_ra, NULL);

    hid_t file_dataspace_dec = H5Dget_space(dec_dataset_id);
    H5Sselect_hyperslab(file_dataspace_dec, H5S_SELECT_SET, offset_dec, NULL, dims_dec, NULL);

    hid_t file_dataspace_valid = H5Dget_space(valid_dataset_id);
    H5Sselect_hyperslab(file_dataspace_valid, H5S_SELECT_SET, offset_valid, NULL, dims_valid, NULL);

    hid_t file_dataspace_bestFreqs = H5Dget_space(bestFreqs_dataset_id);
    H5Sselect_hyperslab(file_dataspace_bestFreqs, H5S_SELECT_SET, offset_bestFreqs, NULL, dims_bestFreqs, NULL);

    hid_t file_dataspace_significance = H5Dget_space(significance_dataset_id);
    H5Sselect_hyperslab(file_dataspace_significance, H5S_SELECT_SET, offset_significance, NULL, dims_significance, NULL);

    hid_t file_dataspace_means = H5Dget_space(means_dataset_id);
    H5Sselect_hyperslab(file_dataspace_means, H5S_SELECT_SET, offset_means, NULL, dims_means, NULL);

    hid_t file_dataspace_stds = H5Dget_space(stds_dataset_id);
    H5Sselect_hyperslab(file_dataspace_stds, H5S_SELECT_SET, offset_stds, NULL, dims_stds, NULL);

    // Define the memory dataspace
    hid_t mem_dataspace_psids = H5Screate_simple(1, dims_psids, NULL);
    hid_t mem_dataspace_ra = H5Screate_simple(1, dims_ra, NULL);
    hid_t mem_dataspace_dec = H5Screate_simple(1, dims_dec, NULL);
    hid_t mem_dataspace_valid = H5Screate_simple(1, dims_valid, NULL);
    hid_t mem_dataspace_bestFreqs = H5Screate_simple(1, dims_bestFreqs, NULL);
    hid_t mem_dataspace_significance = H5Screate_simple(1, dims_significance, NULL);
    hid_t mem_dataspace_means = H5Screate_simple(1, dims_means, NULL);
    hid_t mem_dataspace_stds = H5Screate_simple(1, dims_stds, NULL);

    // Write data to the selected hyperslab
    H5Dwrite(psids_dataset_id, H5T_NATIVE_ULONG, mem_dataspace_psids, file_dataspace_psids, H5P_DEFAULT, psids);
    H5Dwrite(ra_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace_ra, file_dataspace_ra, H5P_DEFAULT, ra);
    H5Dwrite(dec_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace_dec, file_dataspace_dec, H5P_DEFAULT, dec);
    H5Dwrite(valid_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace_valid, file_dataspace_valid, H5P_DEFAULT, valid);
    H5Dwrite(bestFreqs_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace_bestFreqs, file_dataspace_bestFreqs, H5P_DEFAULT, bestFreqs);
    H5Dwrite(significance_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace_significance, file_dataspace_significance, H5P_DEFAULT, significance);
    H5Dwrite(means_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace_means, file_dataspace_means, H5P_DEFAULT, means);
    H5Dwrite(stds_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace_stds, file_dataspace_stds, H5P_DEFAULT, stdevs);

    // Close resources
    H5Sclose(mem_dataspace_psids);
    H5Sclose(mem_dataspace_ra);
    H5Sclose(mem_dataspace_dec);
    H5Sclose(mem_dataspace_valid);
    H5Sclose(mem_dataspace_bestFreqs);
    H5Sclose(mem_dataspace_significance);
    H5Sclose(mem_dataspace_means);
    H5Sclose(mem_dataspace_stds);
    H5Sclose(file_dataspace_psids);
    H5Sclose(file_dataspace_ra);
    H5Sclose(file_dataspace_dec);
    H5Sclose(file_dataspace_bestFreqs);
    H5Sclose(file_dataspace_significance);
    H5Sclose(file_dataspace_means);
    H5Sclose(file_dataspace_stds);
    H5Dclose(psids_dataset_id);
    H5Dclose(ra_dataset_id);
    H5Dclose(dec_dataset_id);
    H5Dclose(bestFreqs_dataset_id);
    H5Dclose(significance_dataset_id);
    H5Dclose(means_dataset_id);
    H5Dclose(stds_dataset_id);
    H5Fclose(file_id);
}
