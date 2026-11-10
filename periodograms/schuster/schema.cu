/*
Utilities for building final ZVAR data product
following the defined ZVAR schema. This is currently
envisioned as one HDF5 file per quadrant/filter,
similar to the forced photometry HDF5 files, where
three tables contain the source+Xmatch, LC stats,
and periodicity info, respectively.

Build with:
-----------
nvcc schema_builder.cu -o build_schema -I /usr/include/hdf5/serial/ -L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5 -lm -lstdc++fs


Author: Zach Vanderbosch (Caltech)
*/

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <experimental/filesystem>

#include "schema.h"
#include "hdf5.h"

namespace fs = std::experimental::filesystem;

// HDF5 compression level (0 = lowest/fastest, 9 = highest/slowest)
#define COMPRESSION_LEVEL 1


void check_status(int status) {
  /* 
  Function to check status of reading HDF5 files

  status = Status of HDF5 file read
  */
  assert(status == 0);
}


unsigned long read_starlist(std::string filename, 
                  unsigned long *source_id, 
                  float *ra,
                  float *dec) {

    // Check that file exists
    const fs::path filepath{filename};
    if (!fs::exists(filepath)) 
    {
        std::cout << filename << " does not exist!" << std::endl;
        return 0;
    }

    unsigned long id;
    float r,d,x,y;
   
    unsigned long i = 0;
    std::ifstream file(filename); 
    while (file >> id >> r >> d >> x >> y) 
    {
        source_id[i] = id;
        ra[i] = r;
        dec[i] = d;
        i++;
    }
      
    return i;
}

unsigned long read_periods(std::string filename,
                           int Nperiod,
                           float *period,
                           float *height,
                           float *snr) {

    // Check that file exists
    const fs::path filepath{filename};
    if (!fs::exists(filepath)) 
    {
        std::cout << filename << " does not exist!" << std::endl;
        return 0;
    }

    // Parameters to store peak frequency, height, and significance
    float f,h,s;

    // Create pointer to object number
    unsigned long j = 0;
   
    // Open file and iterate over each line and period
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) 
    {
        std::istringstream iss(line);
        for (int i = 0; i < Nperiod; i++) 
        {
            iss >> f; // Peak Frequency
            iss >> h; // Peak height
            iss >> s; // Peak significance
            
            // Save values to vectors
            period[j*Nperiod + i] = 1.0f / f;
            height[j*Nperiod + i] = h;
            snr[j*Nperiod + i] = s;
        }
        j++;
    }
    return j;
}


herr_t save_sources_table(hid_t group, std::string file_starlist) {

    // Initialize some HDF5 objects
    hid_t space,dset,dcpl;
    herr_t status;

    // Count lines in starlist file
    std::ifstream inFile(file_starlist); 
    unsigned long n_stars = std::count(
        std::istreambuf_iterator<char>(inFile), 
        std::istreambuf_iterator<char>(), 
        '\n'
    );

    // Generate vectors and read the starlist file
    std::vector<unsigned long> source_id(n_stars);
    std::vector<float> ra(n_stars);
    std::vector<float> dec(n_stars);

    unsigned long line_count = read_starlist(
        file_starlist, 
        source_id.data(), 
        ra.data(), 
        dec.data()
    );

    // Confirm line counts are the same
    assert(n_stars == line_count);

    // Generate vector of structs to store all data
    std::vector<starlist> data(n_stars);
    
    
    // Create compound data type, allocate memory and copy data
    hid_t sources_tid = H5Tcreate(H5T_COMPOUND,sizeof(starlist));
    H5Tinsert(sources_tid,"source_id",HOFFSET(starlist,source_id),H5T_NATIVE_ULONG);
    H5Tinsert(sources_tid,"ra",HOFFSET(starlist,ra),H5T_NATIVE_FLOAT);
    H5Tinsert(sources_tid,"dec",HOFFSET(starlist,dec),H5T_NATIVE_FLOAT);

    for (int i = 0; i < n_stars; i++) {
        data[i].source_id = source_id[i];
        data[i].ra = ra[i];
        data[i].dec = dec[i];
    }
    
    // Set Data dimensions
    hsize_t dims[1] = {n_stars};
    hsize_t maxdims[1] = {H5S_UNLIMITED};
    hsize_t chunk[1] = {1024};
    
    // Create dataspace. Setting maximum size to NULL sets the maximum size to be the current size
    space = H5Screate_simple(1,dims,maxdims);
    // Create the dataset creation property list, add the gzip compression filter and set the chunk size
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_deflate(dcpl,COMPRESSION_LEVEL);
    status = H5Pset_chunk(dcpl,1,chunk);
    
    // Create the dataset
    dset = H5Dcreate(group,"/data/sources",sources_tid,space,H5P_DEFAULT,dcpl,H5P_DEFAULT);
    
    // Write the data to the dataset
    status = H5Dwrite(dset,sources_tid,H5S_ALL,H5S_ALL,H5P_DEFAULT,data.data());
    
    // Close and release resources, free memory
    status = H5Pclose(dcpl);
    status = H5Dclose(dset);
    status = H5Sclose(space);
    
    // Clear memoery
    std::vector<starlist>().swap(data);
    std::vector<unsigned long>().swap(source_id);
    std::vector<float>().swap(ra);
    std::vector<float>().swap(dec);
 
    return status;
}


herr_t save_periods_table(hid_t group,
                          std::string file_periods,
                          int n_periods) {

    // Initialize some HDF5 objects
    hid_t space, dset, dcpl;
    herr_t status;

    // Count lines in periods file
    std::ifstream inFile(file_periods); 
    unsigned long n_stars = std::count(
        std::istreambuf_iterator<char>(inFile), 
        std::istreambuf_iterator<char>(), 
        '\n'
    );

    // Generate vectors to store periods, peak heights, and peak S/N
    std::vector<float> period(n_stars*n_periods);
    std::vector<float> height(n_stars*n_periods);
    std::vector<float> snr(n_stars*n_periods);

    unsigned long line_count = read_periods(
        file_periods,
        n_periods,
        period.data(),
        height.data(),
        snr.data()
    );
    assert(n_stars == line_count);

    // Generate vector of structs to store all data
    std::vector<ptable> data(n_stars*n_periods);

    // Create compound data type, allocate memory
    hid_t periods_tid = H5Tcreate(H5T_COMPOUND,sizeof(ptable));
    H5Tinsert(periods_tid,"period_LS",HOFFSET(ptable,period_LS),H5T_NATIVE_FLOAT);
    H5Tinsert(periods_tid,"height_LS",HOFFSET(ptable,height_LS),H5T_NATIVE_FLOAT);
    H5Tinsert(periods_tid,"significance_LS",HOFFSET(ptable,significance_LS),H5T_NATIVE_FLOAT);

    // Copy data
    for (int i = 0; i < n_stars; i++) {
        for (int j = 0; j < n_periods; j++) {
            data[i*n_periods + j].period_LS = period[i*n_periods + j];
            data[i*n_periods + j].height_LS = height[i*n_periods + j];
            data[i*n_periods + j].significance_LS = snr[i*n_periods + j];
        }
    }

    // Set Data dimensions
    hsize_t dims[1] = {n_stars*n_periods};
    hsize_t maxdims[1] = {H5S_UNLIMITED};
    hsize_t chunk[1] = {1024};

    // Create dataspace. Setting maximum size to NULL sets the maximum size to be the current size
    space = H5Screate_simple(1,dims,maxdims);
    // Create the dataset creation property list, add the gzip compression filter and set the chunk size
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_deflate(dcpl,COMPRESSION_LEVEL);
    status = H5Pset_chunk(dcpl,1,chunk);

    // Create the dataset
    dset = H5Dcreate(group,"/data/periods",periods_tid,space,H5P_DEFAULT,dcpl,H5P_DEFAULT);
    
    // Write the data to the dataset
    status = H5Dwrite(dset,periods_tid,H5S_ALL,H5S_ALL,H5P_DEFAULT,data.data());
    
    // Close and release resources, free memory
    status = H5Pclose(dcpl);
    status = H5Dclose(dset);
    status = H5Sclose(space);
    
    // Clear memory
    std::vector<ptable>().swap(data);
    std::vector<float>().swap(period);
    std::vector<float>().swap(height);
    std::vector<float>().swap(snr);

    return status;
}
    

int main (int argc, char *argv[]) {
    
    hid_t file,group;
    herr_t status;
    
    if (argc != 4) {
        printf("Usage: ./build_schema <starlist_file> <period_file> <output_file>\n");
        return 0;
    }
    
    // Open HDF5 file and create base group
    file = H5Fcreate(argv[3],H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    group = H5Gcreate(file,"/data/",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    
    // Save sources to HDF5
    status = save_sources_table(group,argv[1]);
    check_status(status);

    // Save periods to HDF5
    save_periods_table(group, argv[2], 25);
    check_status(status);
 
    // Close HDF5 file
    status = H5Gclose(group);
    status = H5Fclose(file);
    check_status(status);

    return 0;
}
