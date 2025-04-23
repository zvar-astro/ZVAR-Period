float calcMedian(const std::vector<float> &flux, size_t N)
{
  /*
  Function to calculate the median of flux values.

  Parameters:
  - flux: Vector containing flux values (input is not modified)
  - N   : Number of flux values in the vector

  Returns:
  - Median of the flux values
  - 0.0f if the vector is empty
  */

  // Handle edge case: empty vector
  if (N == 0)
  {
    std::cerr << "Warning: Attempted to calculate median of an empty vector." << std::endl;
    return 0.0f;
  }

  // Create a copy of the input vector for partial sorting
  std::vector<float> temp = flux;

  // Find the median using nth_element
  if (N % 2 == 0)
  {
    // For even-sized vector, find the two middle elements
    std::nth_element(temp.begin(), temp.begin() + N / 2 - 1, temp.end());
    float left = temp[N / 2 - 1];
    std::nth_element(temp.begin(), temp.begin() + N / 2, temp.end());
    float right = temp[N / 2];
    return (left + right) / 2.0f;
  }
  else
  {
    // For odd-sized vector, find the middle element
    std::nth_element(temp.begin(), temp.begin() + N / 2, temp.end());
    return temp[N / 2];
  }
}

void calcDeviations(float *deviations, const float *flux, float median, int N)
{
  /*
  Calculate absolute deviations between flux
  values and the median.

  Parameters:
  - deviations: Array to store deviations (modified in place)
  - flux      : Array of flux values (input is not modified)
  - median    : Median of flux values
  - N         : Number of flux values

  Returns:
  - None
  */

  if (deviations == nullptr || flux == nullptr)
  {
    std::cerr << "Error: Null pointer passed to calcDeviations." << std::endl;
    return;
  }
  if (N <= 0)
  {
    std::cerr << "Warning: No flux values to process in calcDeviations." << std::endl;
    return;
  }

  for (int i = 0; i < N; i++)
  {
    deviations[i] = std::abs(flux[i] - median);
  }
}

void normalizeLC(float *flux, int N_s, int N_d)
{
  /*
  Function to median-zero and normalize the flux
  values for each source.

  Parameters:
  - flux: Array of flux values for all sources (modified in place)
  - N_s : Number of sources
  - N_d : Number of flux values per source

  Returns:
  - None
  */

  int size;
  float median, MAD;
  std::vector<float> nzflux;

  // Iterate over each source
  for (int i = 0; i < N_s; i++)
  {
    nzflux.clear(); // Clear the vector for the current source

    // Populate vector with non-zero and non-NaN values
    for (int j = 0; j < N_d; j++)
    {
      float value = flux[i * N_d + j];
      if (value != 0.0f && !std::isnan(value))
      {
        nzflux.push_back(value);
      }
    }

    // Calculate median and MAD
    size = nzflux.size();
    if (size > 0)
    {
      median = calcMedian(nzflux, size);

      // Calculate absolute deviations in-place
      for (int j = 0; j < size; j++)
      {
        nzflux[j] = std::abs(nzflux[j] - median);
      }
      MAD = calcMedian(nzflux, size);
    }
    else
    {
      median = 0.0f;
      MAD = 1.0f; // Avoid division by zero
    }

    // Subtract median and normalize by MAD
    for (int j = 0; j < N_d; j++)
    {
      float value = flux[i * N_d + j];
      if (value != 0.0f && !std::isnan(value))
      {
        flux[i * N_d + j] = (value - median) / MAD;
      }
      else
      {
        flux[i * N_d + j] = 0.0f; // Set invalid values to 0
      }
    }
  }
}