#ifndef POLYNOMIAL_7DEGREE_1D_CUH
#define POLYNOMIAL_7DEGREE_1D_CUH

//IDK Maybe this can work eventually.

__device__ void calculate_7degree1d(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{

    REAL * user_info_float = (REAL*) user_info;
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    value[point_index] = 0; // Initialize to 0
    for (int i = 0; i < 8; i++) {
        value[point_index] += parameters[i] * pow(x, 7 - i);
    }

    REAL * current_derivative = derivative + point_index;

    for (int i = 0; i < 8; i++) {
        current_derivative[i * n_points] = pow(x, 7 - i);
    
    }
}

#endif // POLYNOMIAL_7DEGREE_1D_CUH