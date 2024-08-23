// Utility functions for schuster code
//
// Author: Zach Vanderbosch

#include "utils.h"

// Base directories for HDF5 matchfiles and for saving
// the final period detection outputs
std::string MATCHFILE_DIR = "/data/zvar/matchfiles/";
std::string PERIOD_DIR = "/data/zvar/periods/";


std::string getHDFFilename(int field, int ccd, int quad, std::string filter) {
  /*
  Function that returns filename for HDF5 file

  field  = ZTF field ID
  ccd    = ZTF CCD ID
  quad   = ZTF quadrant ID
  filter = Filter name (e.g. g or r)
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


std::string getPeakFilename(int field, int ccd, int quad, std::string filter) {
  /*
  Function that returns filename for period detection data

  field  = ZTF field ID
  ccd    = ZTF CCD ID
  quad   = ZTF quadrant ID
  filter = Filter name (e.g. g or r)
  */

  // Define directories/filename to store
  // Schuster Periodogram (SP) period detections
  std::stringstream ssp;
  ssp << PERIOD_DIR;
  std::string period_path = ssp.str();
  ssp << std::setfill('0') << std::setw(4) << field;
  std::string period_field_path = ssp.str();
  ssp << "/periods_SP_";
  ssp << std::setfill('0') << std::setw(4) << field << "_";
  ssp << std::setfill('0') << std::setw(2) << ccd << "_";
  ssp << quad << "_z" << filter << ".dat";
  std::string peakfilename = ssp.str();

  // Create directories if they don't already exist
  fs::create_directory(period_path); // create PERIOD_DIR
  fs::create_directory(period_field_path); // create PERIOD_DIR/field folder

  return peakfilename;
}
