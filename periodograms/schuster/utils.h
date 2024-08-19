
#include <iomanip>
#include <string>
#include <sstream>

#ifndef SCHUSTER_UTILS_H
#define SCHUSTER_UTILS_H

std::string getHDFFilename(
  int,        // field
  int,        // ccd
  int,        // quad
  std::string // filter
);

#endif
