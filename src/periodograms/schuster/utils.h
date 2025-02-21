
#include <iomanip>
#include <string>
#include <sstream>
#include <experimental/filesystem>

#ifndef SCHUSTER_UTILS_H
#define SCHUSTER_UTILS_H

namespace fs = std::experimental::filesystem;

std::string getHDFFilename(
  int,        // field
  int,        // ccd
  int,        // quad
  std::string // filter
);

std::string getPeakFilename(
  int,        // field
  int,        // ccd
  int,        // quad
  std::string // filter
);

#endif
