#include "diagnostics.h"

/**
 * @brief Log a message to a file
 * 
 * @param logfile The file to log to
 * @param message The message to log
 */
void logMessage(const std::string& logfile, const std::string& message) {
    std::ofstream log_file(logfile, std::ios_base::out | std::ios_base::app);
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    log_file << std::ctime(&now_time) << message << std::endl;
}