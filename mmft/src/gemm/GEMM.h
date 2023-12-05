#pragma once
#include "../matrix/DenseMat.h"

namespace mmft {

template<typename Scalar_t>
void gemm(const DenseMat<Scalar_t> &A, 
            const DenseMat<Scalar_t> &B, 
            DenseMat<Scalar_t> &C, 
            const Index_t lda, const Index_t ldb,
            const Scalar_t alpha, const Scalar_t beta);

} // namespace mmft