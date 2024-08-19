// Utility functions for schuster code
//
// Author: Zach Vanderbosch

#include "utils.h"

std::string MATCHFILE_DIR = "/data/zvar/matchfiles/";

std::string getHDFFilename(int field, int ccd, int quad, std::string filter) {
  /*
  Function that returns filename for HDF5 file
  */

  std::stringstream ssh;
  ssh << MATCHFILE_DIR;
  ssh << std::setfill('0') << std::setw(4) << field << "/data_";
  ssh << std::setfill('0') << std::setw(4) << field << "_";
  ssh << std::setfill('0') << std::setw(2) << ccd << "_";
  ssh << quad << "_z" << filter << ".h5";
  std::string hdf_filename = ssh.str();

  return hdf_filename;
}
