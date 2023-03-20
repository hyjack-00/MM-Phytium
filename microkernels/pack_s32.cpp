#include "microkernels.h"

void packs32_4x4k4_A(
    int32_t *A, int32_t *Apack,
    size_t ni, size_t nk, size_t LDA) 
{
    int32_t *pA;
    int32_t *pApack = Apack;
    for (size_t i4 = 0; i4 < nk; i4 += 4) {
        for (size_t k = 0; k < nk; k += 4) {
            pA = A + i4*LDA + k;
            vst1q_s32(pApack+0  , vld1q_s32(pA+LDA*0));
            vst1q_s32(pApack+4  , vld1q_s32(pA+LDA*1));
            vst1q_s32(pApack+8  , vld1q_s32(pA+LDA*2));
            vst1q_s32(pApack+12 , vld1q_s32(pA+LDA*3));
            pApack += 16;
        }
    }
}
void packs32_4x4k4_B(
    int32_t *B, int32_t *Bpack,
    size_t nk, size_t nj, size_t LDB) 
{

    int32_t *pB;
    int32_t *pBpack = Bpack;
    for (size_t j = 0; j < nj; j += 4) {
        pB = B + j;
        for (size_t k = 0; k < nk; k += 1) {
            vst1q_s32(pBpack, vld1q_s32(pB));
            pB += LDB;
            pBpack += 4;
        }
    }
}