#ifndef MSE_PSNR_H
#define MSE_PSNR_H

#include <cstddef>

double compute_mse_gpu(const double* d_orig, const double* d_comp, size_t size);

double compute_relative_l2_error_gpu(const double* d_orig, const double* d_comp, size_t size);

double compute_max_error_gpu(const double* d_orig, const double* d_comp, size_t size);


#endif // MSE_PSNR_H

