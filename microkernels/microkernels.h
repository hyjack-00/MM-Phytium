#include <cstdio>
#include <arm_neon.h>

void mks32_0(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC);

void mks32_4x4k4_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC);