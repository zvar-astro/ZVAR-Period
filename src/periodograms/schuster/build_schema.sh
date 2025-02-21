#!/bin/bash

HDF5_INCLUDE="/usr/include/hdf5/serial/"
HDF5_LIBRARY="/usr/lib/x86_64-linux-gnu/hdf5/serial/"

# Compile the schuster program
nvcc schema.cu -o schema_builder -I $HDF5_INCLUDE -L $HDF5_LIBRARY -std=c++17 -lhdf5 -lstdc++fs -O3

exit 0
