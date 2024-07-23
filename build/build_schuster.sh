#!/bin/bash

HDF5_INCLUDE="/usr/include/hdf5/serial/"
HDF5_LIBRARY="/usr/lib/x86_64-linux-gnu/hdf5/serial/"
SCHUSTER_CUFILE="../periodograms/schuster/schuster_quad_sf.cu"
HDF5_UTILS_CUFILE="../periodograms/schuster/hdf5_utils.cu"
PEAK_FINDING_CUFILE="../periodograms/schuster/peak_utils.cu"

# Compile the schuster program
nvcc $SCHUSTER_CUFILE $HDF5_UTILS_CUFILE $PEAK_FINDING_CUFILE -o schuster -I $HDF5_INCLUDE -L $HDF5_LIBRARY -lhdf5 -lstdc++fs -Xcompiler -fopenmp -lgomp -O3

exit 0
