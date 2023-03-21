#include "microkernels.h"

// 444
void packs32_4x4k4_A(
    int32_t *A, int32_t *Apack,
    size_t it, size_t kt, size_t LDA) 
{
    int32_t *pA;
    int32_t *pApack = Apack;
    for (size_t i = 0; i < it; i += 4) {
        for (size_t k = 0; k < kt; k += 4) {
            pA = A + i*LDA + k;
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
    size_t kt, size_t jt, size_t LDB) 
{
    int32_t *pB;
    int32_t *pBpack = Bpack;
    for (size_t j = 0; j < jt; j += 4) {
        pB = B + j;
        for (size_t k = 0; k < kt; k += 1) {
            vst1q_s32(pBpack, vld1q_s32(pB));
            pB += LDB;
            pBpack += 4;
        }
    }
}


// 488
void packs32_4x8k8_A(
    int32_t *A, int32_t *Apack,
    size_t it, size_t kt, size_t LDA) 
{
    int32_t *pA;
    int32_t *pApack = Apack;
    for (size_t i = 0; i < it; i += 4) {
        for (size_t k = 0; k < kt; k += 8) {
            pA = A + i*LDA + k;
            vst1q_s32(pApack+0  , vld1q_s32(pA + LDA*0));
            vst1q_s32(pApack+4  , vld1q_s32(pA + LDA*0 + 4));
            vst1q_s32(pApack+8  , vld1q_s32(pA + LDA*1));
            vst1q_s32(pApack+12 , vld1q_s32(pA + LDA*1 + 4));
            vst1q_s32(pApack+16 , vld1q_s32(pA + LDA*2));
            vst1q_s32(pApack+20 , vld1q_s32(pA + LDA*2 + 4));
            vst1q_s32(pApack+24 , vld1q_s32(pA + LDA*3));
            vst1q_s32(pApack+28 , vld1q_s32(pA + LDA*3 + 4));
            pApack += 32;
        }
    }
}
void packs32_4x8k8_B(
    int32_t *B, int32_t *Bpack,
    size_t kt, size_t jt, size_t LDB) 
{
    int32_t *pB;
    int32_t *pBpack = Bpack;
    for (size_t j = 0; j < jt; j += 8) {
        pB = B + j;
        for (size_t k = 0; k < kt; k += 1) {
            vst1q_s32(pBpack,   vld1q_s32(pB));
            vst1q_s32(pBpack+4, vld1q_s32(pB + 4));
            pB += LDB;
            pBpack += 8;
        }
    }
}
void packs32_4x8k8_C(
    int32_t *C, int32_t *Cpack,
    size_t it, size_t jt, size_t LDC)
{
    int32_t *pC;
    int32_t *pCpack = Cpack;
    for (size_t i = 0; i < it; i += 4) {
        for (size_t j = 0; j < jt; j += 8) {
            pC = C + i*LDC + j;
            vst1q_s32(pCpack+0  , vld1q_s32(pC + LDC*0));
            vst1q_s32(pCpack+4  , vld1q_s32(pC + LDC*0 + 4));
            vst1q_s32(pCpack+8  , vld1q_s32(pC + LDC*1));
            vst1q_s32(pCpack+12 , vld1q_s32(pC + LDC*1 + 4));
            vst1q_s32(pCpack+16 , vld1q_s32(pC + LDC*2));
            vst1q_s32(pCpack+20 , vld1q_s32(pC + LDC*2 + 4));
            vst1q_s32(pCpack+24 , vld1q_s32(pC + LDC*3));
            vst1q_s32(pCpack+28 , vld1q_s32(pC + LDC*3 + 4));
            pCpack += 32;
        }
    }
}
void unpacks32_4x8k8_C(
    int32_t *C, int32_t *Cpack,
    size_t it, size_t jt, size_t LDC)
{
    int32_t *pC;
    int32_t *pCpack = Cpack;
    for (size_t i = 0; i < it; i += 4) {
        for (size_t j = 0; j < jt; j += 8) {
            pC = C + i*LDC + j;
            vst1q_s32(pC + LDC*0    , vld1q_s32(pCpack+0));
            vst1q_s32(pC + LDC*0 + 4, vld1q_s32(pCpack+4));
            vst1q_s32(pC + LDC*1    , vld1q_s32(pCpack+8));
            vst1q_s32(pC + LDC*1 + 4, vld1q_s32(pCpack+12));
            vst1q_s32(pC + LDC*2    , vld1q_s32(pCpack+16));
            vst1q_s32(pC + LDC*2 + 4, vld1q_s32(pCpack+20));
            vst1q_s32(pC + LDC*3    , vld1q_s32(pCpack+24));
            vst1q_s32(pC + LDC*3 + 4, vld1q_s32(pCpack+28));
            pCpack += 32;
        }
    }
}

// 848
void packs32_8x4k8_A(
    int32_t *A, int32_t *Apack,
    size_t it, size_t kt, size_t LDA) 
{
    int32_t *pA;
    int32_t *pApack = Apack;
    for (size_t i = 0; i < it; i += 8) {
        for (size_t k = 0; k < kt; k += 8) {
            pA = A + i*LDA + k;
            vst1q_s32(pApack+0  , vld1q_s32(pA + LDA*0));
            vst1q_s32(pApack+4  , vld1q_s32(pA + LDA*0 + 4));
            vst1q_s32(pApack+8  , vld1q_s32(pA + LDA*1));
            vst1q_s32(pApack+12 , vld1q_s32(pA + LDA*1 + 4));
            vst1q_s32(pApack+16 , vld1q_s32(pA + LDA*2));
            vst1q_s32(pApack+20 , vld1q_s32(pA + LDA*2 + 4));
            vst1q_s32(pApack+24 , vld1q_s32(pA + LDA*3));
            vst1q_s32(pApack+28 , vld1q_s32(pA + LDA*3 + 4));
            vst1q_s32(pApack+32 , vld1q_s32(pA + LDA*4));
            vst1q_s32(pApack+36 , vld1q_s32(pA + LDA*4 + 4));
            vst1q_s32(pApack+40 , vld1q_s32(pA + LDA*5));
            vst1q_s32(pApack+44 , vld1q_s32(pA + LDA*5 + 4));
            vst1q_s32(pApack+48 , vld1q_s32(pA + LDA*6));
            vst1q_s32(pApack+52 , vld1q_s32(pA + LDA*6 + 4));
            vst1q_s32(pApack+56 , vld1q_s32(pA + LDA*7));
            vst1q_s32(pApack+60 , vld1q_s32(pA + LDA*7 + 4));
            pApack += 64;
        }
    }
}
void packs32_8x4k8_B(
    int32_t *B, int32_t *Bpack,
    size_t kt, size_t jt, size_t LDB) 
{
    int32_t *pB;
    int32_t *pBpack = Bpack;
    for (size_t j = 0; j < jt; j += 4) {
        pB = B + j;
        for (size_t k = 0; k < kt; k += 1) {
            vst1q_s32(pBpack, vld1q_s32(pB));
            pB += LDB;
            pBpack += 4;
        }
    }
}