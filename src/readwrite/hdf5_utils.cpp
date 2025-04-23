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
#include <memory>
#include <functional>
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

// RAII Wrapper classes for HDF5 resources

class H5Handle {
protected:
    hid_t id_;
public:
    explicit H5Handle(hid_t id) : id_(id) {}
    bool valid() const { return id_ >= 0; }
    hid_t id() const { return id_; }
    
    // Prevent copying
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    
    // Allow moving
    H5Handle(H5Handle&& other) noexcept : id_(other.id_) {
        other.id_ = -1;
    }
    
    H5Handle& operator=(H5Handle&& other) noexcept {
        if (this != &other) {
            id_ = other.id_;
            other.id_ = -1;
        }
        return *this;
    }
};

class H5FileHandle : public H5Handle {
public:
    explicit H5FileHandle(const std::string& filename, unsigned flags = H5F_ACC_RDONLY)
        : H5Handle(H5Fopen(filename.c_str(), flags, H5P_DEFAULT)) {}
    ~H5FileHandle() { if (valid()) H5Fclose(id_); }
};

class H5GroupHandle : public H5Handle {
public:
    H5GroupHandle(const H5Handle& parent, const std::string& name)
        : H5Handle(H5Gopen2(parent.id(), name.c_str(), H5P_DEFAULT)) {}
    ~H5GroupHandle() { if (valid()) H5Gclose(id_); }
};

class H5DatasetHandle : public H5Handle {
public:
    H5DatasetHandle(const H5Handle& parent, const std::string& name)
        : H5Handle(H5Dopen2(parent.id(), name.c_str(), H5P_DEFAULT)) {}
    ~H5DatasetHandle() { if (valid()) H5Dclose(id_); }
};

class H5DataspaceHandle : public H5Handle {
public:
    explicit H5DataspaceHandle(const H5DatasetHandle& dataset)
        : H5Handle(H5Dget_space(dataset.id())) {}
    explicit H5DataspaceHandle(int rank, const hsize_t* dims)
        : H5Handle(H5Screate_simple(rank, dims, NULL)) {}
    ~H5DataspaceHandle() { if (valid()) H5Sclose(id_); }
};

class H5TypeHandle : public H5Handle {
public:
    explicit H5TypeHandle(hid_t type_id)
        : H5Handle(type_id) {}
    ~H5TypeHandle() { if (valid()) H5Tclose(id_); }
};

// Utility function to check error status and log
template<typename T>
bool check_and_log(T result, const std::string& error_message) {
    if (result < 0) {
        std::cerr << "Error: " << error_message << std::endl;
        return false;
    }
    return true;
}

// Refactored function example
int get_number_of_rows(hid_t dataset)
{
  /*
  Calculate the number of rows in the dataset.

  dataset = HDF5 dataset object

  Returns:
  - Number of rows in the dataset (>= 0) on success
  - -1 on error
  */

  H5DataspaceHandle dspace(H5DatasetHandle(dataset));
  if (!dspace.valid()) {
    std::cerr << "Error: Unable to get dataspace for dataset." << std::endl;
    return -1;
  }

  int ndims = H5Sget_simple_extent_ndims(dspace.id());
  if (ndims != 1) {
    std::cerr << "Error: Dataset is not 1-dimensional." << std::endl;
    return -1;
  }

  hsize_t dims[1];
  if (H5Sget_simple_extent_dims(dspace.id(), dims, NULL) < 0) {
    std::cerr << "Error: Unable to get dataset dimensions." << std::endl;
    return -1;
  }

  return static_cast<int>(dims[0]);
}

// Refactored wrapper function
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

  try {
    // Use RAII handles - resources will be automatically closed in reverse order
    H5FileHandle file(filename);
    if (!file.valid()) {
      std::cerr << "Error: Unable to open HDF5 file: " << filename << std::endl;
      return -1;
    }

    H5GroupHandle group(file, "/data");
    if (!group.valid()) {
      std::cerr << "Error: Unable to open group '/data' in file: " << filename << std::endl;
      return -1;
    }

    H5DatasetHandle table(group, tablename);
    if (!table.valid()) {
      std::cerr << "Error: Unable to open dataset '" << tablename << "' in file: " << filename << std::endl;
      return -1;
    }

    // Get number of rows using the original function
    return get_number_of_rows(table.id());
  } 
  catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return -1;
  }
}

// Example refactored function for handling compound data types
template<typename T>
class H5CompoundTypeBuilder {
private:
    H5TypeHandle type_;
    
public:
    H5CompoundTypeBuilder() 
        : type_(H5Tcreate(H5T_COMPOUND, sizeof(T))) {}
        
    template<typename FieldType>
    H5CompoundTypeBuilder& addField(const char* name, size_t offset, hid_t field_type) {
        if (type_.valid()) {
            if (H5Tinsert(type_.id(), name, offset, field_type) < 0) {
                std::cerr << "Error: Unable to insert field '" << name << "' into compound type." << std::endl;
            }
        }
        return *this;
    }
    
    hid_t getTypeId() const { return type_.id(); }
    bool valid() const { return type_.valid(); }
};

// Refactored version of get_exposures
int get_exposures(hid_t table, double *bjd, int begin, int N)
{
  /*
  Function that grabs BJD timestamps from the exposures table.

  Parameters:
  - table: HDF5 table object
  - bjd: Array pointer to store BJD timestamps
  - begin: Index at which to start storing data
  - N: Number of BJD timestamps to read

  Returns:
  - 0 on success
  - -1 on error
  */

  // Create compound datatype with builder pattern
  H5CompoundTypeBuilder<exposures> builder;
  builder.addField<double>("mjd", HOFFSET(exposures, bjd), H5T_NATIVE_DOUBLE);
  
  if (!builder.valid()) {
    return -1;
  }

  // Use std::vector for memory management
  std::vector<exposures> data(N);

  // Read data from the HDF5 table
  int status = H5Dread(table, builder.getTypeId(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  if (status < 0) {
    std::cerr << "Error: Unable to read data from HDF5 table." << std::endl;
    return -1;
  }

  // Populate the BJD array with values
  for (int i = 0; i < N; i++) {
    bjd[i + begin] = data[i].bjd;
  }

  return 0; // Success
}

// Example of a more complex function using the new utilities
int get_sourcedata_chunk(std::string filename,
                         float *flux,
                         float *fluxErr,
                         float *ratioValid,
                         std::vector<unsigned char> &keep,
                         unsigned long objstart,
                         unsigned long objnum,
                         unsigned long numentry)
{
  /*
  Function that reads a contiguous chunk
  of data from the sourcedata table.
  */

  // Ensure the size of the `keep` vector matches the expected number of exposures
  int N = keep.size(); // Original number of exposures per source
  if (N <= 0) {
    std::cerr << "Error: Invalid size of 'keep' vector." << std::endl;
    return -1;
  }

  try {
    // Open all resources with RAII
    H5FileHandle file(filename);
    if (!file.valid()) {
      std::cerr << "Error: Unable to open HDF5 file: " << filename << std::endl;
      return -1;
    }

    H5GroupHandle group(file, "/data");
    if (!group.valid()) {
      std::cerr << "Error: Unable to open group '/data' in file: " << filename << std::endl;
      return -1;
    }

    H5DatasetHandle table(group, "/data/sourcedata");
    if (!table.valid()) {
      std::cerr << "Error: Unable to open dataset '/data/sourcedata' in file: " << filename << std::endl;
      return -1;
    }

    // Get dataspace and select hyperslab
    H5DataspaceHandle dataspace(table);
    if (!dataspace.valid()) {
      std::cerr << "Error: Unable to get dataspace for dataset." << std::endl;
      return -1;
    }

    // Define hyperslab
    hsize_t offset[] = {objstart * N};
    hsize_t count[] = {objnum * N};
    if (H5Sselect_hyperslab(dataspace.id(), H5S_SELECT_SET, offset, NULL, count, NULL) < 0) {
      std::cerr << "Error: Unable to select hyperslab in dataset." << std::endl;
      return -1;
    }

    // Create memory dataspace
    hsize_t dim[] = {objnum * N};
    H5DataspaceHandle memspace(1, dim);
    if (!memspace.valid()) {
      std::cerr << "Error: Unable to create memory dataspace." << std::endl;
      return -1;
    }

    // Create compound datatype
    H5CompoundTypeBuilder<sourcedata> builder;
    builder.addField<float>("flux", HOFFSET(sourcedata, flux), H5T_NATIVE_FLOAT)
           .addField<float>("fluxerr", HOFFSET(sourcedata, fluxErr), H5T_NATIVE_FLOAT);
    
    if (!builder.valid()) {
      return -1;
    }

    // Allocate memory for the data using unique_ptr for automatic cleanup
    auto data = std::unique_ptr<sourcedata[]>(new sourcedata[objnum * N]);
    if (!data) {
      std::cerr << "Error: Memory allocation failed for sourcedata." << std::endl;
      return -1;
    }

    // Read data
    if (H5Dread(table.id(), builder.getTypeId(), memspace.id(), dataspace.id(), H5P_DEFAULT, data.get()) < 0) {
      std::cerr << "Error: Unable to read data from HDF5 table." << std::endl;
      return -1;
    }

    // Process data as before
    for (unsigned long i = 0; i < objnum; i++) {
      int k = 0;
      int validCount = 0;

      for (int j = 0; j < N; j++) {
        if (keep[j] == 1) {
          if (std::isnan(data[i * N + j].flux)) {
            flux[i * numentry + k] = 0.0f;
            fluxErr[i * numentry + k] = INFINITY;
          }
          else {
            flux[i * numentry + k] = data[i * N + j].flux;
            fluxErr[i * numentry + k] = data[i * N + j].fluxErr;
            validCount++;
          }
          k++;
        }
      }

      // Calculate the ratio of valid points
      ratioValid[i] = static_cast<float>(validCount) / static_cast<float>(k);
    }

    return 0; // Success
  }
  catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return -1;
  }
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

std::vector<unsigned char> flagDeepDrilling(const std::vector<double> &time, int N_d)
{
  /*
  Function to identify any timestamps within light
  curve associated with ZTF deep drilling observations.

  Parameters:
  - time: Vector of times in Julian Date (JD) format
  - N_d : Number of exposures per source

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

  return keep;
}
