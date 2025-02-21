#ifndef CUBIC_1D_CUH
#define CUBIC_1D_CUH

__device__ void calculate_cubic1d(
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
    REAL a = parameters[0];
    REAL b = parameters[1];
    REAL c = parameters[2];
    REAL d = parameters[3];

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

    value[point_index] = a * x * x * x + b * x * x + c * x + d;

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = x * x * x;
    current_derivative[1 * n_points] = x * x;
    current_derivative[2 * n_points] = x;
    current_derivative[3 * n_points] = 1;
}

#endif // CUBIC_1D_CUH