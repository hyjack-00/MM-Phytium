// Log
#include <iostream>
#include <fstream>
using std::cout;
using std::endl;

#include <arm_neon.h>


//# Micro-Kernel
typedef void mks32_t(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC);

void mks32_0(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC);

void mks32_4x4k4_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC);
void mks32_4x4k4_ldA_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC);

/* 128-bits Vector Registers:
        C                  =  A                     *  B
    [C0      ][C1      ]   [A00      ][A01      ]   [B00      ][B01      ]
    [C2      ][C3      ]   [A10      ][A11      ]   [B10      ][B11      ]
    [C4      ][C5      ]   [A20      ][A21      ]   [B20      ][B21      ]
    [C6      ][C7      ]   [A30      ][A31      ]   [B30      ][B31      ]
                                                    [B40      ][B41      ]
                                                    [B50      ][B51      ]
                                                    [B60      ][B61      ]
                                                    [B70      ][B71      ]
*/
void mks32_4x8k8_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC);
void mks32_4x8k8_ldA_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC);



//# Micro-Kernel + Repacking
typedef void mks32_pAB_t(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk, size_t LDC);

void mks32_4x4k4_ldB_fchC_pkAB(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk, size_t LDC);

void mks32_4x8k8_ldB_fchC_pkAB(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk, size_t LDC);



//# Matrix Repacking
typedef void packs32_A_t(
    int32_t *A, int32_t *Apack,
    size_t it, size_t kt, size_t LDA);
typedef void packs32_B_t(
    int32_t *B, int32_t *Bpack,
    size_t kt, size_t jt, size_t LDB);

// 444
void packs32_4x4k4_A(
    int32_t *A, int32_t *Apack,
    size_t it, size_t kt, size_t LDA);
void packs32_4x4k4_B(
    int32_t *B, int32_t *Bpack,
    size_t kt, size_t jt, size_t LDB);

// 488
void packs32_4x8k8_A(
    int32_t *A, int32_t *Apack,
    size_t ni, size_t nk, size_t LDA);
void packs32_4x8k8_B(
    int32_t *B, int32_t *Bpack,
    size_t nk, size_t nj, size_t LDB);