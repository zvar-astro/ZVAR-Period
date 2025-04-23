#include <vector>

// Function to calculate median of flux values
float calcMedian(std::vector<float> &, size_t);

// Function to calculate deviations b/t flux values and median
void calcDeviations(float *, float *, float, size_t);

// Function to normalize flux values
void normalizeLC(float *, int, int);