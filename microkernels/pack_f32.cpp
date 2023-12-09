#include "microkernels.h"

void packf32_4x8k8_A(
    float32_t *A, float32_t *Apack,
    size_t it, size_t kt, size_t LDA) 
{
    float32_t *pA;
    float32_t *pApack = Apack;
    for (size_t i = 0; i < it; i += 4) {
        for (size_t k = 0; k < kt; k += 8) {
            pA = A + i*LDA + k;
            vst1q_f32(pApack+0  , vld1q_f32(pA + LDA*0));
            vst1q_f32(pApack+4  , vld1q_f32(pA + LDA*0 + 4));
            vst1q_f32(pApack+8  , vld1q_f32(pA + LDA*1));
            vst1q_f32(pApack+12 , vld1q_f32(pA + LDA*1 + 4));
            vst1q_f32(pApack+16 , vld1q_f32(pA + LDA*2));
            vst1q_f32(pApack+20 , vld1q_f32(pA + LDA*2 + 4));
            vst1q_f32(pApack+24 , vld1q_f32(pA + LDA*3));
            vst1q_f32(pApack+28 , vld1q_f32(pA + LDA*3 + 4));
            pApack += 32;
        }
    }
}

void packf32_4x8k8_B(
    float32_t *B, float32_t *Bpack,
    size_t kt, size_t jt, size_t LDB)
{
    float32_t *pB;
    float32_t *pBpack = Bpack;
    for (size_t j = 0; j < jt; j += 8) {
        pB = B + j;
        for (size_t k = 0; k < kt; k += 4) {
            vst1q_f32(pBpack,    vld1q_f32(pB + LDB * 0));
            vst1q_f32(pBpack+4,  vld1q_f32(pB + LDB * 0 + 4));
            vst1q_f32(pBpack+8,  vld1q_f32(pB + LDB * 1));
            vst1q_f32(pBpack+12, vld1q_f32(pB + LDB * 1 + 4));
            vst1q_f32(pBpack+16, vld1q_f32(pB + LDB * 2));
            vst1q_f32(pBpack+20, vld1q_f32(pB + LDB * 2 + 4));
            vst1q_f32(pBpack+24, vld1q_f32(pB + LDB * 3));
            vst1q_f32(pBpack+28, vld1q_f32(pB + LDB * 3 + 4));
            pB += LDB * 4;
            pBpack += 32;
        }
    }
}