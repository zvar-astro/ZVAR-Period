#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>

void logMessage(const std::string& logfile, const std::string& message);
#endif //DIAGNOSTICS_H