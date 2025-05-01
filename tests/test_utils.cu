#include "test_utils.cuh"

// Function to read a CSV file into a 2D vector
std::vector<std::vector<double>> readCSV(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to extract a column from a 2D vector
std::vector<double> extractColumn(const std::vector<std::vector<double>>& data, size_t colIndex) {
    std::vector<double> column;
    for (const auto& row : data) {
        if (colIndex < row.size()) {
            column.push_back(row[colIndex]);
        } else {
            throw std::out_of_range("Column index out of range");
        }
    }
    return column;
}

//Function to compute the mean and standard deviation of a vector
std::pair<float, float> computeMeanAndStdDev(const std::vector<float>& data) {
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    float sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f);
    float stdev = std::sqrt(sq_sum / data.size() - mean * mean);
    return {mean, stdev};
}

// Helper function: Partition the array around a pivot
size_t partitionArray(std::vector<float>& data, size_t left, size_t right, size_t pivotIndex) {
    float pivotValue = data[pivotIndex];
    std::swap(data[pivotIndex], data[right]); // Move pivot to the end
    size_t storeIndex = left;

    for (size_t i = left; i < right; ++i) {
        if (data[i] < pivotValue) {
            std::swap(data[i], data[storeIndex]);
            ++storeIndex;
        }
    }

    std::swap(data[storeIndex], data[right]); // Move pivot to its final place
    return storeIndex;
}

// Helper function: Quickselect to find the k-th smallest element
float quickselect(std::vector<float>& data, size_t k) {
    size_t left = 0, right = data.size() - 1;

    while (true) {
        if (left == right) {
            return data[left];
        }

        size_t pivotIndex = left + (right - left) / 2;
        pivotIndex = partitionArray(data, left, right, pivotIndex);

        if (k == pivotIndex) {
            return data[k];
        } else if (k < pivotIndex) {
            right = pivotIndex - 1;
        } else {
            left = pivotIndex + 1;
        }
    }
}

// Function to compute the median and MAD using Quickselect
std::pair<float, float> computeMedianAndMAD(std::vector<float>& data) {
    size_t n = data.size();
    if (n == 0) {
        throw std::invalid_argument("The input array is empty.");
    }

    // Compute the median using Quickselect
    float median;
    if (n % 2 == 0) {
        float mid1 = quickselect(data, n / 2 - 1);
        float mid2 = quickselect(data, n / 2);
        median = (mid1 + mid2) / 2.0f;
    } else {
        median = quickselect(data, n / 2);
    }

    // Compute the absolute deviations from the median
    std::vector<float> absDev(n);
    std::transform(data.begin(), data.end(), absDev.begin(), [median](float x) { return std::abs(x - median); });

    // Compute the MAD using Quickselect
    float mad;
    if (n % 2 == 0) {
        float mid1 = quickselect(absDev, n / 2 - 1);
        float mid2 = quickselect(absDev, n / 2);
        mad = (mid1 + mid2) / 2.0f;
    } else {
        mad = quickselect(absDev, n / 2);
    }

    return {median, mad};
}

// Function to convert a vector to a string for debugging
std::string vectorToString(const std::vector<float>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}