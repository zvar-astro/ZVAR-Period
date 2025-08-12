// Utility functions for opening and reading HDF5 files
//
// By: Zach Vanderbosch

#include <cstdlib>
#include <iostream>
#include <cassert>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "hdf5.h"
#include "hdf5_utils.h"

// Define the threshold for deep drilling (5 minutes in fractional days)
const double DEEP_DRILLING_THRESHOLD = 0.003472222;

void check_status(int status)
{
  /*
  Function to check status of reading HDF5 files

  status = Status of HDF5 file read
  */
  assert(status == 0);
}

int get_number_of_rows(hid_t dataset)
{
  /*
  Calculate the number of rows in the dataset.

  dataset = HDF5 dataset object

  Returns:
  - Number of rows in the dataset (>= 0) on success
  - -1 on error
  */

  hid_t dspace = H5Dget_space(dataset);
  if (dspace < 0)
  {
    std::cerr << "Error: Unable to get dataspace for dataset." << std::endl;
    return -1;
  }

  int ndims = H5Sget_simple_extent_ndims(dspace);
  if (ndims != 1)
  {
    H5Sclose(dspace);
    std::cerr << "Error: Dataset is not 1-dimensional." << std::endl;
    return -1;
  }

  hsize_t dims[1];
  if (H5Sget_simple_extent_dims(dspace, dims, NULL) < 0)
  {
    H5Sclose(dspace);
    std::cerr << "Error: Unable to get dataset dimensions." << std::endl;
    return -1;
  }

  H5Sclose(dspace);
  return static_cast<int>(dims[0]);
}

int get_number_of_rows_wrapper(std::string filename, std::string tablename)
{
  /*
  A wrapper function for the get_number_of_rows function.
  Does the same thing as get_number_of_rows, but as input
  this function only requires the HDF5 filename and
  table name rather than an hid_t table object.

  filename  = Name of HDF5 file
  tablename = Name of HDF5 table

  Returns:
  - Number of rows in the dataset (>= 0) on success
  - -1 on error
  */

  // Open HDF5 file
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
  {
    std::cerr << "Error: Unable to open HDF5 file: " << filename << std::endl;
    return -1;
  }

  // Open group
  hid_t group = H5Gopen2(file, "/data", H5P_DEFAULT);
  if (group < 0)
  {
    std::cerr << "Error: Unable to open group '/data' in file: " << filename << std::endl;
    H5Fclose(file);
    return -1;
  }

  // Open dataset
  hid_t table = H5Dopen2(group, tablename.c_str(), H5P_DEFAULT);
  if (table < 0)
  {
    std::cerr << "Error: Unable to open dataset '" << tablename << "' in file: " << filename << std::endl;
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Get number of rows using the original function
  int nrows = get_number_of_rows(table);

  // Close HDF5 objects
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return nrows;
}

// int get_exposures(hid_t table, double *bjd, int begin, int N)
// let's also take the name of the field as an argument
int get_exposures(hid_t table, double *bjd, int begin, int N, const char *field_name)
{
  /*
  Function that grabs BJD timestamps from the exposures table.

  Parameters:
  - table: HDF5 table object
  - bjd: Array pointer to store BJD timestamps
  - begin: Index at which to start storing data
  - N: Number of BJD timestamps to read
  - field_name: Name of the field to read

  Returns:
  - 0 on success
  - -1 on error
  */
    // std::cout << "field_name: " << field_name << std::endl;
  // Create the HDF5 compound datatype for exposures
  hid_t exposures_tid = H5Tcreate(H5T_COMPOUND, sizeof(exposures));
  if (exposures_tid < 0)
  {
    std::cerr << "Error: Unable to create HDF5 compound datatype for exposures." << std::endl;
    return -1;
  }

  if (H5Tinsert(exposures_tid, field_name, HOFFSET(exposures, bjd), H5T_NATIVE_DOUBLE) < 0)
  {
    std::cerr << "Error: Unable to insert '" << field_name << "' field into HDF5 compound datatype." << std::endl;
    H5Tclose(exposures_tid);
    return -1;
  }
  // Use std::vector for memory management
  // This is valid as long as the size of exposures isn't too large
  // Typically, we have no more than ~2000 exposures for ZTF fields
  // so this should be safe, and it avoids manual memory management
  std::vector<exposures> data(N);

  // Read data from the HDF5 table
  int status = H5Dread(table, exposures_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  if (status < 0)
  {
    std::cerr << "Error: Unable to read data from HDF5 table." << std::endl;
    H5Tclose(exposures_tid);
    return -1;
  }

  // Populate the BJD array with values
  for (int i = 0; i < N; i++)
  {
    bjd[i + begin] = data[i].bjd;
  }

  // Clean up resources
  H5Tclose(exposures_tid);

  return 0; // Success
}

int get_exposures_wrapper(std::string filename, double *bjd, int begin, int N, const char *field_name)
{
  /*
  A wrapper function for the get_exposures function.
  Does the same thing as get_exposures, but as input
  this function only requires the HDF5 filename
  rather than an hid_t table object.

  Parameters:
  - filename: Name of the HDF5 file
  - bjd: Array pointer to store BJD timestamps
  - begin: Index at which to start storing data
  - N: Number of BJD timestamps to read
  - field_name: Name of the field to read

  Returns:
  - 0 on success
  - -1 on error
  */

  // Open HDF5 file
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
  {
    std::cerr << "Error: Unable to open HDF5 file: " << filename << std::endl;
    return -1;
  }

  // Open group
  hid_t group = H5Gopen2(file, "/data", H5P_DEFAULT);
  if (group < 0)
  {
    std::cerr << "Error: Unable to open group '/data' in file: " << filename << std::endl;
    H5Fclose(file);
    return -1;
  }

  // Open dataset
  hid_t table = H5Dopen2(group, "/data/exposures", H5P_DEFAULT);
  if (table < 0)
  {
    std::cerr << "Error: Unable to open dataset '/data/exposures' in file: " << filename << std::endl;
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Get timestamp data
  int status = get_exposures(table, bjd, begin, N, field_name);
  if (status < 0)
  {
    std::cerr << "Error: Failed to retrieve exposures from dataset '/data/exposures'." << std::endl;
  }

  // Close HDF5 objects
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return status;
}

int get_sources(hid_t table, unsigned long *psid, int begin, int N, const char *field_name)
{
  /*
  Function that grabs the Pan-STARRS IDs
  for all objects in the sources table.

  Parameters:
  - table: HDF5 table object
  - psid: Array pointer to store Pan-STARRS IDs
  - begin: Index at which to start storing data
  - N: Number of Pan-STARRS IDs to read
  - field_name: Name of the field to read

  Returns:
  - 0 on success
  - -1 on error
  */

  int status;
  sources *data;

  // Create the HDF5 compound datatype for sources
  hid_t sources_tid = H5Tcreate(H5T_COMPOUND, sizeof(sources));
  if (sources_tid < 0)
  {
    std::cerr << "Error: Unable to create HDF5 compound datatype for sources." << std::endl;
    return -1;
  }

  if (H5Tinsert(sources_tid, field_name, HOFFSET(sources, psid), H5T_NATIVE_ULONG) < 0)
  {
    std::cerr << "Error: Unable to insert '" << field_name << "' field into HDF5 compound datatype." << std::endl;
    H5Tclose(sources_tid);
    return -1;
  }
  
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
                        unsigned long *psid,
                        int begin,
                        int N,
                        const char *field_name)
{
  /*
  A wrapper function for the get_sources function.
  Does the same thing as get_sources, but as input
  this function only requires the HDF5 filename
  rather than an hid_t table object.

  Parameters:
  - filename: Name of the HDF5 file
  - psid: Array pointer to store Pan-STARRS IDs
  - begin: Index at which to start storing data
  - N: Number of Pan-STARRS IDs to read
  - field_name: Name of the field to read

  Returns:
  - 0 on success
  - -1 on error
  */

  // Open HDF5 file
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
  {
    std::cerr << "Error: Unable to open HDF5 file: " << filename << std::endl;
    return -1;
  }

  // Open group
  hid_t group = H5Gopen2(file, "/data", H5P_DEFAULT);
  if (group < 0)
  {
    std::cerr << "Error: Unable to open group '/data' in file: " << filename << std::endl;
    H5Fclose(file);
    return -1;
  }

  // Open dataset
  hid_t table = H5Dopen2(group, "/data/sources", H5P_DEFAULT);
  if (table < 0)
  {
    std::cerr << "Error: Unable to open dataset '/data/sources' in file: " << filename << std::endl;
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Call the get_sources function to retrieve data
  int status = get_sources(table, psid, begin, N, field_name);
  if (status < 0)
  {
    std::cerr << "Error: Failed to retrieve sources from dataset '/data/sources'." << std::endl;
  }

  // Close HDF5 objects
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return status;
}

int get_sources_chunk(std::string filename,
                      unsigned long *psids,
                      float *ra,
                      float *dec,
                      unsigned long objstart,
                      unsigned long objnum,
                      const char *id_field_name,
                      const char *ra_field_name,
                      const char *dec_field_name)
{
  /*
  Function to read a contiguous chunk of data from the sources table.
  Used to return Pan-STARRS IDs, RA, and Dec for a subset of objects.

  Parameters:
  - filename: Name of the HDF5 file
  - psids: Array pointer to store Pan-STARRS IDs
  - ra: Array pointer to store RA coordinates
  - dec: Array pointer to store Dec coordinates
  - objstart: Starting index of the objects to read
  - objnum: Number of objects to read in this chunk
  - id_field_name: Name of the field for the IDs
  - ra_field_name: Name of the field for RA coordinates
  - dec_field_name: Name of the field for Dec coordinates

  Returns:
  - 0 on success
  - -1 on error
  */

  // Open HDF5 file
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
  {
    std::cerr << "Error: Unable to open HDF5 file: " << filename << std::endl;
    return -1;
  }

  // Open group
  hid_t group = H5Gopen2(file, "/data", H5P_DEFAULT);
  if (group < 0)
  {
    std::cerr << "Error: Unable to open group '/data' in file: " << filename << std::endl;
    H5Fclose(file);
    return -1;
  }

  // Open dataset
  hid_t table = H5Dopen2(group, "/data/sources", H5P_DEFAULT);
  if (table < 0)
  {
    std::cerr << "Error: Unable to open dataset '/data/sources' in file: " << filename << std::endl;
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Define hyperslab in the dataset
  hid_t dataspace = H5Dget_space(table);
  hsize_t offset[] = {objstart};
  hsize_t count[] = {objnum};
  if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL) < 0)
  {
    std::cerr << "Error: Unable to select hyperslab in dataset." << std::endl;
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Define memory hyperslab
  hsize_t dim[] = {objnum};
  hid_t memspace = H5Screate_simple(1, dim, NULL);
  if (memspace < 0)
  {
    std::cerr << "Error: Unable to create memory dataspace." << std::endl;
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Create the HDF5 compound datatype for sources
  hid_t sources_tid = H5Tcreate(H5T_COMPOUND, sizeof(sources));
  if (sources_tid < 0)
  {
    std::cerr << "Error: Unable to create HDF5 compound datatype for sources." << std::endl;
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  if (H5Tinsert(sources_tid, id_field_name, HOFFSET(sources, psid), H5T_NATIVE_ULONG) < 0 ||
      H5Tinsert(sources_tid, ra_field_name, HOFFSET(sources, ra), H5T_NATIVE_FLOAT) < 0 ||
      H5Tinsert(sources_tid, dec_field_name, HOFFSET(sources, dec), H5T_NATIVE_FLOAT) < 0)
  {
    std::cerr << "Error: Unable to insert fields into HDF5 compound datatype." << std::endl;
    H5Tclose(sources_tid);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Allocate memory for the chunk on the heap
  sources *data = (sources *)malloc(sizeof(sources) * objnum);
  if (data == NULL)
  {
    std::cerr << "Error: Memory allocation failed for sources data." << std::endl;
    H5Tclose(sources_tid);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Read data from the HDF5 table
  if (H5Dread(table, sources_tid, memspace, dataspace, H5P_DEFAULT, data) < 0)
  {
    std::cerr << "Error: Unable to read data from HDF5 table." << std::endl;
    free(data);
    H5Tclose(sources_tid);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Populate the output arrays with values
  for (unsigned long i = 0; i < objnum; i++)
  {
    psids[i] = data[i].psid;
    ra[i] = data[i].ra;
    dec[i] = data[i].dec;
  }

  // Free heap memory
  free(data);

  // Clean up resources
  H5Tclose(sources_tid);
  H5Sclose(memspace);
  H5Sclose(dataspace);
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return 0; // Success
}

int get_sourcedata(hid_t table, float *flux, int N, const char *field_name)
{
  /*
  Function that grabs flux values for all
  sources from the sourcedata table.

  Parameters:
  - table: HDF5 table object
  - flux: Array pointer to store flux values
  - N: Number of flux values per source
  - field_name: Name of the field to read

  Returns:
  - 0 on success
  - -1 on error
  */

  // Create the HDF5 compound datatype for sourcedata
  hid_t sourcedata_tid = H5Tcreate(H5T_COMPOUND, sizeof(sourcedata));
  if (sourcedata_tid < 0)
  {
    std::cerr << "Error: Unable to create HDF5 compound datatype for sourcedata." << std::endl;
    return -1;
  }

  if (H5Tinsert(sourcedata_tid, field_name, HOFFSET(sourcedata, flux), H5T_NATIVE_FLOAT) < 0)
  {
    std::cerr << "Error: Unable to insert '" << field_name << "' field into HDF5 compound datatype." << std::endl;
    H5Tclose(sourcedata_tid);
    return -1;
  }

  // Allocate memory for the data
  sourcedata *data = (sourcedata *)malloc(sizeof(sourcedata) * N);
  if (data == NULL)
  {
    std::cerr << "Error: Memory allocation failed for sourcedata." << std::endl;
    H5Tclose(sourcedata_tid);
    return -1;
  }

  // Read data from the HDF5 table
  int status = H5Dread(table, sourcedata_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  if (status < 0)
  {
    std::cerr << "Error: Unable to read data from HDF5 table." << std::endl;
    free(data);
    H5Tclose(sourcedata_tid);
    return -1;
  }

  // Populate the flux array with values
  for (int i = 0; i < N; i++)
  {
    if (std::isnan(data[i].flux))
    {
      flux[i] = 0.0f; // Replace NaN with 0.0
    }
    else
    {
      flux[i] = data[i].flux;
    }
  }

  // Clean up resources
  free(data);
  H5Tclose(sourcedata_tid);

  return 0; // Success
}

int get_sourcedata_chunk(std::string filename,
                        float *flux,
                        float *fluxErr,
                        float *ratioValid,
                        std::vector<unsigned char> &keep,
                        unsigned long objstart,
                        unsigned long objnum,
                        unsigned long numentry,
                        const char *flux_field_name,
                        const char *fluxErr_field_name)
{
  /*
  Function that reads a contiguous chunk
  of data from the sourcedata table.

  Parameters:
  - filename   : Name of HDF5 file
  - flux       : Array pointer to store flux values
  - fluxErr    : Array pointer to store flux errors
  - ratioValid : Array pointer to store what percentage of a light curve is valid
  - keep       : Array indicating non-deep drilling time stamps (1=keep, 0=ignore)
  - objstart   : Object number at which to start reading data
  - objnum     : Number of objects to read
  - numentry   : Number of entries reserved per source within flux, including padding
  - flux_field_name: Name of the field for flux values
  - fluxErr_field_name: Name of the field for flux error values

  Returns:
  - 0 on success
  - -1 on error
  */

  // Ensure the size of the `keep` vector matches the expected number of exposures
  int N = keep.size(); // Original number of exposures per source
  if (N <= 0)
  {
    std::cerr << "Error: Invalid size of 'keep' vector." << std::endl;
    return -1;
  }

  // Open HDF5 file
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
  {
    std::cerr << "Error: Unable to open HDF5 file: " << filename << std::endl;
    return -1;
  }

  // Open group
  hid_t group = H5Gopen2(file, "/data", H5P_DEFAULT);
  if (group < 0)
  {
    std::cerr << "Error: Unable to open group '/data' in file: " << filename << std::endl;
    H5Fclose(file);
    return -1;
  }

  // Open dataset
  hid_t table = H5Dopen2(group, "/data/sourcedata", H5P_DEFAULT);
  if (table < 0)
  {
    std::cerr << "Error: Unable to open dataset '/data/sourcedata' in file: " << filename << std::endl;
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Define hyperslab in the dataset
  hid_t dataspace = H5Dget_space(table);
  if (dataspace < 0)
  {
    std::cerr << "Error: Unable to get dataspace for dataset." << std::endl;
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  hsize_t offset[] = {objstart * N};
  hsize_t count[] = {objnum * N};
  if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL) < 0)
  {
    std::cerr << "Error: Unable to select hyperslab in dataset." << std::endl;
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Define memory hyperslab
  hsize_t dim[] = {objnum * N};
  hid_t memspace = H5Screate_simple(1, dim, NULL);
  if (memspace < 0)
  {
    std::cerr << "Error: Unable to create memory dataspace." << std::endl;
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Create the HDF5 compound datatype for sourcedata
  hid_t sourcedata_tid = H5Tcreate(H5T_COMPOUND, sizeof(sourcedata));
  if (sourcedata_tid < 0)
  {
    std::cerr << "Error: Unable to create HDF5 compound datatype for sourcedata." << std::endl;
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  if (H5Tinsert(sourcedata_tid, flux_field_name, HOFFSET(sourcedata, flux), H5T_NATIVE_FLOAT) < 0 ||
      H5Tinsert(sourcedata_tid, fluxErr_field_name, HOFFSET(sourcedata, fluxErr), H5T_NATIVE_FLOAT) < 0)
  {
    std::cerr << "Error: Unable to insert fields into HDF5 compound datatype." << std::endl;
    H5Tclose(sourcedata_tid);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Allocate memory for the chunk on the heap
  sourcedata *data = (sourcedata *)malloc(sizeof(sourcedata) * objnum * N);
  if (data == NULL)
  {
    std::cerr << "Error: Memory allocation failed for sourcedata." << std::endl;
    H5Tclose(sourcedata_tid);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Read data from the HDF5 table
  if (H5Dread(table, sourcedata_tid, memspace, dataspace, H5P_DEFAULT, data) < 0)
  {
    std::cerr << "Error: Unable to read data from HDF5 table." << std::endl;
    free(data);
    H5Tclose(sourcedata_tid);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(table);
    H5Gclose(group);
    H5Fclose(file);
    return -1;
  }

  // Declare `k` outside the loop
  int k; // Points to index within flux and fluxErr

  for (int i = 0; i < objnum; i++)
  {
    k = 0; // Reset `k` for each source
    int validCount = 0;

    for (int j = 0; j < N; j++)
    {
      if (keep[j] == 1)
      {
        if (std::isnan(data[i * N + j].flux))
        {
          flux[i * numentry + k] = 0.0f;
        //   fluxErr[i * numentry + k] = INFINITY;
        fluxErr[i * numentry + k] = 100000000.0f;
        }
        else
        {
          flux[i * numentry + k] = data[i * N + j].flux;
          fluxErr[i * numentry + k] = data[i * N + j].fluxErr;
          validCount++;
        }
        k++; // Increment `k` only when `keep[j] == 1`
      }
    }

    // Calculate the ratio of valid points
    ratioValid[i] = static_cast<float>(validCount) / static_cast<float>(k);
  }

  // Clean up resources
  free(data);
  H5Tclose(sourcedata_tid);
  H5Sclose(memspace);
  H5Sclose(dataspace);
  H5Dclose(table);
  H5Gclose(group);
  H5Fclose(file);

  return 0; // Success
}

float calcMedian(const std::vector<float> flux, size_t N)
{
  /*
  Function to calculate the median of flux values.

  Parameters:
  - flux: Vector containing flux values (input is not modified)
  - N   : Number of flux values in the vector

  Returns:
  - Median of the flux values
  - 0.0f if the vector is empty
  */

  // Handle edge case: empty vector
  if (N == 0)
  {
    std::cerr << "Warning: Attempted to calculate median of an empty vector." << std::endl;
    return 0.0f;
  }

  // Create a copy of the input vector for partial sorting
  std::vector<float> temp = flux;

  // Find the median using nth_element
  if (N % 2 == 0)
  {
    // For even-sized vector, find the two middle elements
    std::nth_element(temp.begin(), temp.begin() + N / 2 - 1, temp.end());
    float left = temp[N / 2 - 1];
    std::nth_element(temp.begin(), temp.begin() + N / 2, temp.end());
    float right = temp[N / 2];
    return (left + right) / 2.0f;
  }
  else
  {
    // For odd-sized vector, find the middle element
    std::nth_element(temp.begin(), temp.begin() + N / 2, temp.end());
    return temp[N / 2];
  }
}

void calcDeviations(float *deviations, const float *flux, float median, int N)
{
  /*
  Calculate absolute deviations between flux
  values and the median.

  Parameters:
  - deviations: Array to store deviations (modified in place)
  - flux      : Array of flux values (input is not modified)
  - median    : Median of flux values
  - N         : Number of flux values

  Returns:
  - None
  */

  if (deviations == nullptr || flux == nullptr)
  {
    std::cerr << "Error: Null pointer passed to calcDeviations." << std::endl;
    return;
  }
  if (N <= 0)
  {
    std::cerr << "Warning: No flux values to process in calcDeviations." << std::endl;
    return;
  }

  for (int i = 0; i < N; i++)
  {
    deviations[i] = std::abs(flux[i] - median);
  }
}

void normalizeLC(float *flux, int N_s, int N_d)
{
  /*
  Function to median-zero and normalize the flux
  values for each source.

  Parameters:
  - flux: Array of flux values for all sources (modified in place)
  - N_s : Number of sources
  - N_d : Number of flux values per source

  Returns:
  - None
  */

  int size;
  float median, MAD;
  std::vector<float> nzflux;

  // Iterate over each source
  for (int i = 0; i < N_s; i++)
  {
    nzflux.clear(); // Clear the vector for the current source

    // Populate vector with non-zero and non-NaN values
    for (int j = 0; j < N_d; j++)
    {
      float value = flux[i * N_d + j];
      if (value != 0.0f && !std::isnan(value))
      {
        nzflux.push_back(value);
      }
    }

    // Calculate median and MAD
    size = nzflux.size();
    if (size > 0)
    {
      median = calcMedian(nzflux, size);

      // Calculate absolute deviations in-place
      for (int j = 0; j < size; j++)
      {
        nzflux[j] = std::abs(nzflux[j] - median);
      }
      MAD = calcMedian(nzflux, size);
    }
    else
    {
      median = 0.0f;
      MAD = 1.0f; // Avoid division by zero
    }

    // Subtract median and normalize by MAD
    for (int j = 0; j < N_d; j++)
    {
      float value = flux[i * N_d + j];
      if (value != 0.0f && !std::isnan(value))
      {
        flux[i * N_d + j] = (value - median) / MAD;
      }
      else
      {
        flux[i * N_d + j] = 0.0f; // Set invalid values to 0
      }
    }
  }
}

std::string createOutputFilepath(std::string output_dir, std::string input_hdf_filename)
{
    std::string mkdir = "mkdir -p " + output_dir;
    system(mkdir.c_str());
    std::string hdf_filename_copy = input_hdf_filename;
    std::string::size_type pos = hdf_filename_copy.find_last_of("/");
    if (pos != std::string::npos) {
        hdf_filename_copy = hdf_filename_copy.substr(pos + 1);
    }
    std::string results_filename = output_dir + "/" + hdf_filename_copy + "_results.h5.temp";
    return results_filename;
}

void createResults(std::string filename, int total_curves, int N_samples, int N_peaks)
{
  /*
  Function to create and write datasets to an HDF5 file.

  Parameters:
  - filename    : Name of the HDF5 file to create
  - total_curves: Total number of curves
  - N_samples   : Number of samples per curve
  - N_peaks     : Number of peaks per sample

  Returns:
  - None
  */

  // Helper lambda to create a dataset
  auto createDataset = [](hid_t file_id, const char *name, hid_t type, hsize_t *dims) -> hid_t
  {
    hid_t dataspace = H5Screate_simple(1, dims, NULL);
    if (dataspace < 0)
    {
      std::cerr << "Error: Unable to create dataspace for dataset: " << name << std::endl;
      return -1;
    }
    hid_t dataset = H5Dcreate(file_id, name, type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(dataspace);
    if (dataset < 0)
    {
      std::cerr << "Error: Unable to create dataset: " << name << std::endl;
    }
    return dataset;
  };

  // Open or create the HDF5 file
  hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error: Unable to create HDF5 file: " << filename << std::endl;
    return;
  }

  // Define the maximum dimensions for each dataset
  hsize_t dims_psids[1] = {static_cast<hsize_t>(total_curves)};
  hsize_t dims_ra[1] = {static_cast<hsize_t>(total_curves)};
  hsize_t dims_dec[1] = {static_cast<hsize_t>(total_curves)};
  hsize_t dims_valid[1] = {static_cast<hsize_t>(total_curves)};
  hsize_t dims_bestFreqs[1] = {static_cast<hsize_t>(total_curves) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks)};
  hsize_t dims_significance[1] = {static_cast<hsize_t>(total_curves) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks)};
  hsize_t dims_means[1] = {static_cast<hsize_t>(total_curves) * static_cast<hsize_t>(N_samples)};
  hsize_t dims_stds[1] = {static_cast<hsize_t>(total_curves) * static_cast<hsize_t>(N_samples)};

  // Create datasets
  hid_t psids_dataset = createDataset(file_id, "psids", H5T_NATIVE_ULONG, dims_psids);
  hid_t ra_dataset = createDataset(file_id, "ra", H5T_NATIVE_FLOAT, dims_ra);
  hid_t dec_dataset = createDataset(file_id, "dec", H5T_NATIVE_FLOAT, dims_dec);
  hid_t valid_dataset = createDataset(file_id, "valid", H5T_NATIVE_FLOAT, dims_valid);
  hid_t bestFreqs_dataset = createDataset(file_id, "bestFreqs", H5T_NATIVE_FLOAT, dims_bestFreqs);
  hid_t significance_dataset = createDataset(file_id, "significance", H5T_NATIVE_FLOAT, dims_significance);
  hid_t means_dataset = createDataset(file_id, "means", H5T_NATIVE_FLOAT, dims_means);
  hid_t stds_dataset = createDataset(file_id, "stds", H5T_NATIVE_FLOAT, dims_stds);

  // Check for errors in dataset creation
  if (psids_dataset < 0 || ra_dataset < 0 || dec_dataset < 0 || valid_dataset < 0 ||
      bestFreqs_dataset < 0 || significance_dataset < 0 || means_dataset < 0 || stds_dataset < 0)
  {
    std::cerr << "Error: One or more datasets could not be created." << std::endl;
    H5Fclose(file_id);
    return;
  }

  // Close datasets
  H5Dclose(psids_dataset);
  H5Dclose(ra_dataset);
  H5Dclose(dec_dataset);
  H5Dclose(valid_dataset);
  H5Dclose(bestFreqs_dataset);
  H5Dclose(significance_dataset);
  H5Dclose(means_dataset);
  H5Dclose(stds_dataset);

  // Close the file
  H5Fclose(file_id);
}

void appendResults(std::string filename, unsigned long *psids, float *ra, float *dec, float *valid, float *bestFreqs, float *significance, float *means, float *stdevs, int N_curves, int N_samples, int N_peaks, unsigned long offset)
{
  /*
  Function to append data to an existing HDF5 file.

  Parameters:
  - filename    : Name of the HDF5 file
  - psids       : Array of Pan-STARRS IDs
  - ra          : Array of RA values
  - dec         : Array of Dec values
  - valid       : Array of valid flags
  - bestFreqs   : Array of best frequencies
  - significance: Array of significance values
  - means       : Array of mean values
  - stdevs      : Array of standard deviation values
  - N_curves    : Number of curves to append
  - N_samples   : Number of samples per curve
  - N_peaks     : Number of peaks per sample
  - offset      : Offset in the dataset to start appending

  Returns:
  - None
  */

  // Helper lambda to open a dataset
  auto openDataset = [](hid_t file_id, const char *name) -> hid_t
  {
    hid_t dataset = H5Dopen(file_id, name, H5P_DEFAULT);
    if (dataset < 0)
    {
      std::cerr << "Error: Unable to open dataset: " << name << std::endl;
    }
    return dataset;
  };

  // Helper lambda to define a dataspace and select a hyperslab
  auto defineDataspace = [](hid_t dataset_id, hsize_t *offset, hsize_t *dims) -> hid_t
  {
    hid_t dataspace = H5Dget_space(dataset_id);
    if (dataspace < 0)
    {
      std::cerr << "Error: Unable to get dataspace for dataset." << std::endl;
      return -1;
    }
    if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, dims, NULL) < 0)
    {
      std::cerr << "Error: Unable to select hyperslab in dataset." << std::endl;
      H5Sclose(dataspace);
      return -1;
    }
    return dataspace;
  };

  // Helper lambda to create a memory dataspace
  auto createMemDataspace = [](hsize_t *dims) -> hid_t
  {
    hid_t memspace = H5Screate_simple(1, dims, NULL);
    if (memspace < 0)
    {
      std::cerr << "Error: Unable to create memory dataspace." << std::endl;
    }
    return memspace;
  };

  // Open the existing HDF5 file
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error: Unable to open HDF5 file: " << filename << std::endl;
    return;
  }

  // Dataset names and corresponding data pointers
  const char *dataset_names[] = {"psids", "ra", "dec", "valid", "bestFreqs", "significance", "means", "stds"};
  void *data_pointers[] = {psids, ra, dec, valid, bestFreqs, significance, means, stdevs};
  hsize_t dims[] = {
      static_cast<hsize_t>(N_curves),
      static_cast<hsize_t>(N_curves),
      static_cast<hsize_t>(N_curves),
      static_cast<hsize_t>(N_curves),
      static_cast<hsize_t>(N_curves) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks),
      static_cast<hsize_t>(N_curves) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks),
      static_cast<hsize_t>(N_curves) * static_cast<hsize_t>(N_samples),
      static_cast<hsize_t>(N_curves) * static_cast<hsize_t>(N_samples)};
  hsize_t offsets[] = {
      static_cast<hsize_t>(offset),
      static_cast<hsize_t>(offset),
      static_cast<hsize_t>(offset),
      static_cast<hsize_t>(offset),
      static_cast<hsize_t>(offset) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks),
      static_cast<hsize_t>(offset) * static_cast<hsize_t>(N_samples) * static_cast<hsize_t>(N_peaks),
      static_cast<hsize_t>(offset) * static_cast<hsize_t>(N_samples),
      static_cast<hsize_t>(offset) * static_cast<hsize_t>(N_samples)};

  // Iterate over datasets and write data
  for (int i = 0; i < 8; i++)
  {
    hid_t dataset_id = openDataset(file_id, dataset_names[i]);
    if (dataset_id < 0)
      continue;

    hid_t file_dataspace = defineDataspace(dataset_id, &offsets[i], &dims[i]);
    if (file_dataspace < 0)
    {
      H5Dclose(dataset_id);
      continue;
    }

    hid_t mem_dataspace = createMemDataspace(&dims[i]);
    if (mem_dataspace < 0)
    {
      H5Sclose(file_dataspace);
      H5Dclose(dataset_id);
      continue;
    }

    if (H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, mem_dataspace, file_dataspace, H5P_DEFAULT, data_pointers[i]) < 0)
    {
      std::cerr << "Error: Unable to write data to dataset: " << dataset_names[i] << std::endl;
    }

    // Close resources
    H5Sclose(mem_dataspace);
    H5Sclose(file_dataspace);
    H5Dclose(dataset_id);
  }

  // Close the file
  H5Fclose(file_id);
}

std::vector<unsigned char> flagDeepDrilling(const std::vector<double> time, int N_d, bool invert = false)
{
  /*
  Function to identify any timestamps within light
  curve associated with ZTF deep drilling observations.

  Parameters:
  - time: Vector of times in Julian Date (JD) format
  - N_d : Number of exposures per source
  - invert: If true, invert the selection to keep *ONLY* deep drilling (1=ignore, 0=keep)

  Returns:
  - A vector of unsigned char (1=keep, 0=ignore) indicating
    which timestamps should be kept.
  */

  // Get all unique integer JDs when data were acquired
  std::vector<int> jd(N_d);
  for (int i = 0; i < N_d; i++)
  {
    jd[i] = static_cast<int>(std::floor(time[i]));
  }

  // Derive unique JDs
  std::vector<int> unique_jd = jd;
  std::sort(unique_jd.begin(), unique_jd.end());
  unique_jd.erase(std::unique(unique_jd.begin(), unique_jd.end()), unique_jd.end());
  int Njd = unique_jd.size();

  // Create boolean vector to indicate which timestamps should be kept
  std::vector<unsigned char> keep(N_d, 1);

  // Temporary vectors for indices and time slices
  std::vector<int> indices;
  std::vector<double> time_slice;

  // Iterate over unique days and find number of observations per day
  for (int i = 0; i < Njd; i++)
  {
    // Count how many timestamps fall within a given day
    int count = std::count(jd.begin(), jd.end(), unique_jd[i]);

    // Get indices of all observations on this day
    indices.clear();
    time_slice.clear();
    for (int j = 0; j < N_d; j++)
    {
      if (jd[j] == unique_jd[i])
      {
        indices.push_back(j);
        time_slice.push_back(time[j]);
      }
    }

    // When enough data points are present, check time sampling
    if (count >= 20)
    {
      // Get differences between successive values
      std::adjacent_difference(time_slice.begin(), time_slice.end(), time_slice.begin());

      // Determine whether a timestamp is deep drilling
      for (int j = 1; j < count; j++)
      { // Start from the second element
        if (time_slice[j] < DEEP_DRILLING_THRESHOLD)
        {
          keep[indices[j]] = 0;
        }
      }
    }
  }

    // If invert is true, flip the values in keep vector
    if (invert) {
        std::transform(keep.begin(), keep.end(), keep.begin(), [](unsigned char value) {
            return value == 1 ? 0 : 1;
        });
    }

  return keep;
}

std::tuple<unsigned int, unsigned int, std::vector<float>, std::vector<unsigned char>> initializeFrameData(
  const std::string &filename,
  const char *id_field_name,
  const char *time_field_name,
  bool invert = false
)
{
  /*
  Function to initialize frame data from an HDF5 file.
  Parameters:
  - filename: Name of the HDF5 file
  - id_field_name: Name of the field for source IDs
  - time_field_name: Name of the field for time values
  Returns:
  - A tuple containing:
    - NEXP: Number of exposures
    - NSRC: Number of sources
    - time: Vector of time values
    - deep_drilling_flags: Vector indicating which timestamps to keep
  */

  // Variables to store number of exposures & sources
  int NEXP = get_number_of_rows_wrapper(filename, "/data/exposures");
  int NSRC = get_number_of_rows_wrapper(filename, "/data/sources");

  if (NEXP <= 0 || NSRC <= 0)
  {
      throw std::runtime_error("Invalid number of exposures or sources.");
  }

  // Use NEXP and NSRC to initialize time and source vectors
  std::vector<double> h_bjd(NEXP, 0.0);         // BJD must be read in with double precision
  std::vector<unsigned long> sourceid(NSRC, 0); // Source IDs

  // Read HDF5 tables and load into vectors
  int status = get_exposures_wrapper(filename, h_bjd.data(), 0, NEXP, time_field_name);
  check_status(status);
  status = get_sources_wrapper(filename, sourceid.data(), 0, NSRC, id_field_name);
  check_status(status);

  // Get indices of non-deep drilling time stamps
  std::vector<unsigned char> dd_index = flagDeepDrilling(h_bjd, NEXP, invert);
  int non_dd = std::accumulate(dd_index.begin(), dd_index.end(), 0);
  if (non_dd == 0)
  {
      throw std::runtime_error("No valid BJDs after filtering.");
  }

  // Erase the deep drilling BJDs from bjd vector,
  // or if invert is true, erase non-deep drilling BJDs
  for (int i = NEXP-1; i >= 0; i--) {
    if (dd_index[i] == 0) {
        h_bjd.erase(h_bjd.begin() + i);
    }
  }
  NEXP = non_dd; // Update number of exposures

  // Subtract minimum BJD and convert to float
  std::vector<float> time(NEXP,0.0f); // Time values after subtracting mean BJD
  double bjdref = std::min_element(h_bjd.begin(), h_bjd.end())[0];
  for (int i = 0; i < NEXP; i++) {
    time[i] = static_cast<float>(h_bjd[i] - bjdref);
  }

  return std::make_tuple(NEXP, NSRC, time, dd_index);
}