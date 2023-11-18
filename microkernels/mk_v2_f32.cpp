#include "microkernels.h"
#include <stdlib.h>

#define MKSMM_m8n12_ADD_C \
    ;

#define MKSMM_m8n12_STORE_C \
    ;

// [ Pack B ----------------------------------------------------
#define MKSMM_m8n12_PACKB_BEGIN_K0 \
    vq_A0 = vld1q_f32(AA); AA += 4; \
    vq_B00 = vld1q_f32(B0 + 0); \
    vq_B01 = vld1q_f32(B0 + 4); \
    vq_C00 = vmulq_laneq_f32(vq_B00, vq_A0, 0); \
    vq_C01 = vmulq_laneq_f32(vq_B01, vq_A0, 0); \
    vq_B02 = vld1q_f32(B0 + 8); \
    vq_C02 = vmulq_laneq_f32(vq_B02, vq_A0, 0); \
    \
    B0 += 2 * LN; \
    vq_A1 = vld1q_f32(AA); AA += 4; \
    vq_C10 = vmulq_laneq_f32(vq_B00, vq_A0, 1); \
    vq_C11 = vmulq_laneq_f32(vq_B01, vq_A0, 1); \
    vq_C12 = vmulq_laneq_f32(vq_B02, vq_A0, 1); \
    \
    vq_B10 = vld1q_f32(B1 + 0);  \
    vq_C20 = vmulq_laneq_f32(vq_B00, vq_A0, 2); \
    vq_C21 = vmulq_laneq_f32(vq_B01, vq_A0, 2); \
    vq_C22 = vmulq_laneq_f32(vq_B02, vq_A0, 2); \
    \
    vq_B11 = vld1q_f32(B1 + 4);  \
    vq_C30 = vmulq_laneq_f32(vq_B00, vq_A0, 3); \
    vq_C31 = vmulq_laneq_f32(vq_B01, vq_A0, 3); \
    vq_C32 = vmulq_laneq_f32(vq_B02, vq_A0, 3); \
    \
    vq_A0 = vld1q_f32(AA); AA += 4; \
    vq_C40 = vmulq_laneq_f32(vq_B00, vq_A1, 0); \
    vq_C41 = vmulq_laneq_f32(vq_B01, vq_A1, 0); \
    vq_C42 = vmulq_laneq_f32(vq_B02, vq_A1, 0); \
    \
    vq_B12 = vld1q_f32(B1 + 8);  \
    vq_C50 = vmulq_laneq_f32(vq_B00, vq_A1, 1); \
    vq_C51 = vmulq_laneq_f32(vq_B01, vq_A1, 1); \
    vq_C52 = vmulq_laneq_f32(vq_B02, vq_A1, 1); \
    \
    vst1q_f32(pkBB, vq_B00); pkBB += 4; \
    vst1q_f32(pkBB, vq_B01); pkBB += 4; \
    vq_C60 = vmulq_laneq_f32(vq_B00, vq_A1, 2); \
    vq_C61 = vmulq_laneq_f32(vq_B01, vq_A1, 2); \
    vq_C62 = vmulq_laneq_f32(vq_B02, vq_A1, 2); \
    \
    B1 += 2 * LN; \
    vst1q_f32(pkBB, vq_B02); pkBB += 4; \
    vq_C70 = vmulq_laneq_f32(vq_B00, vq_A1, 3); \
    vq_C71 = vmulq_laneq_f32(vq_B01, vq_A1, 3); \
    vq_C72 = vmulq_laneq_f32(vq_B02, vq_A1, 3); \
    vq_A1 = vld1q_f32(AA); AA += 4; 

// vq_B1X, vq_A0 loaded;  vq_A1 loading
#define MKSMM_m8n12_PACKB_K1 \
    vq_B00 = vld1q_f32(B0 + 0); \
    vq_B01 = vld1q_f32(B0 + 4); \
    vq_C00 = vfmaq_laneq_f32(vq_C00, vq_B10, vq_A0, 0); \
    vq_C01 = vfmaq_laneq_f32(vq_C01, vq_B11, vq_A0, 0); \
    vq_C02 = vfmaq_laneq_f32(vq_C02, vq_B12, vq_A0, 0); \
    \
    vq_B02 = vld1q_f32(B0 + 8); \
    vq_C10 = vfmaq_laneq_f32(vq_C10, vq_B10, vq_A0, 1); \
    vq_C11 = vfmaq_laneq_f32(vq_C11, vq_B11, vq_A0, 1); \
    vq_C12 = vfmaq_laneq_f32(vq_C12, vq_B12, vq_A0, 1); \
    \
    B0 += 2 * LN; \
    vq_C20 = vfmaq_laneq_f32(vq_C20, vq_B10, vq_A0, 2); \
    vq_C21 = vfmaq_laneq_f32(vq_C21, vq_B11, vq_A0, 2); \
    vq_C22 = vfmaq_laneq_f32(vq_C22, vq_B12, vq_A0, 2); \
    \ 
    vq_C30 = vfmaq_laneq_f32(vq_C30, vq_B10, vq_A0, 3); \
    vq_C31 = vfmaq_laneq_f32(vq_C31, vq_B11, vq_A0, 3); \
    vq_C32 = vfmaq_laneq_f32(vq_C32, vq_B12, vq_A0, 3); \
    \
    vq_A0 = vld1q_f32(AA); AA += 4; \ 
    vq_C40 = vfmaq_laneq_f32(vq_C40, vq_B10, vq_A1, 0); \
    vq_C41 = vfmaq_laneq_f32(vq_C41, vq_B11, vq_A1, 0); \
    vq_C42 = vfmaq_laneq_f32(vq_C42, vq_B12, vq_A1, 0); \
    \
    vst1q_f32(pkBB, vq_B10); pkBB += 4; \
    vst1q_f32(pkBB, vq_B11); pkBB += 4; \
    vq_C50 = vfmaq_laneq_f32(vq_C50, vq_B10, vq_A1, 1); \
    vq_C51 = vfmaq_laneq_f32(vq_C51, vq_B11, vq_A1, 1); \
    vq_C52 = vfmaq_laneq_f32(vq_C52, vq_B12, vq_A1, 1); \
    \
    vst1q_f32(pkBB, vq_B12); pkBB += 4; \
    vq_C60 = vfmaq_laneq_f32(vq_C60, vq_B10, vq_A1, 2); \
    vq_C61 = vfmaq_laneq_f32(vq_C61, vq_B11, vq_A1, 2); \
    vq_C62 = vfmaq_laneq_f32(vq_C62, vq_B12, vq_A1, 2); \
    \
    vq_C70 = vfmaq_laneq_f32(vq_C70, vq_B10, vq_A1, 3); \
    vq_C71 = vfmaq_laneq_f32(vq_C71, vq_B11, vq_A1, 3); \
    vq_C72 = vfmaq_laneq_f32(vq_C72, vq_B12, vq_A1, 3); \
    vq_A1 = vld1q_f32(AA); AA += 4; 

// vq_B0X, vq_A0 loaded;  vq_A1 loading 
#define MKSMM_m8n12_PACKB_K0 \
    vq_B10 = vld1q_f32(B1 + 0); \
    vq_B11 = vld1q_f32(B1 + 4); \
    vq_C00 = vfmaq_laneq_f32(vq_C00, vq_B00, vq_A0, 0); \
    vq_C01 = vfmaq_laneq_f32(vq_C01, vq_B01, vq_A0, 0); \
    vq_C02 = vfmaq_laneq_f32(vq_C02, vq_B02, vq_A0, 0); \
    \
    vq_B12 = vld1q_f32(B1 + 8); \
    vq_C10 = vfmaq_laneq_f32(vq_C10, vq_B00, vq_A0, 1); \
    vq_C11 = vfmaq_laneq_f32(vq_C11, vq_B01, vq_A0, 1); \
    vq_C12 = vfmaq_laneq_f32(vq_C12, vq_B02, vq_A0, 1); \
    \
    B1 += 2 * LN; \
    vq_C20 = vfmaq_laneq_f32(vq_C20, vq_B00, vq_A0, 2); \
    vq_C21 = vfmaq_laneq_f32(vq_C21, vq_B01, vq_A0, 2); \
    vq_C22 = vfmaq_laneq_f32(vq_C22, vq_B02, vq_A0, 2); \
    \
    vq_C30 = vfmaq_laneq_f32(vq_C30, vq_B00, vq_A0, 3); \
    vq_C31 = vfmaq_laneq_f32(vq_C31, vq_B01, vq_A0, 3); \
    vq_C32 = vfmaq_laneq_f32(vq_C32, vq_B02, vq_A0, 3); \
    \
    vq_A0 = vld1q_f32(AA); AA += 4; \
    vq_C40 = vfmaq_laneq_f32(vq_C40, vq_B00, vq_A1, 0); \
    vq_C41 = vfmaq_laneq_f32(vq_C41, vq_B01, vq_A1, 0); \
    vq_C42 = vfmaq_laneq_f32(vq_C42, vq_B02, vq_A1, 0); \
    \
    vst1q_f32(pkBB, vq_B00); pkBB += 4; \
    vst1q_f32(pkBB, vq_B01); pkBB += 4; \
    vq_C50 = vfmaq_laneq_f32(vq_C50, vq_B00, vq_A1, 1); \
    vq_C51 = vfmaq_laneq_f32(vq_C51, vq_B01, vq_A1, 1); \
    vq_C52 = vfmaq_laneq_f32(vq_C52, vq_B02, vq_A1, 1); \
    \
    vst1q_f32(pkBB, vq_B12); pkBB += 4; \
    vq_C60 = vfmaq_laneq_f32(vq_C60, vq_B00, vq_A1, 2); \
    vq_C61 = vfmaq_laneq_f32(vq_C61, vq_B01, vq_A1, 2); \
    vq_C62 = vfmaq_laneq_f32(vq_C62, vq_B02, vq_A1, 2); \
    \
    vq_C70 = vfmaq_laneq_f32(vq_C70, vq_B00, vq_A1, 3); \
    vq_C71 = vfmaq_laneq_f32(vq_C71, vq_B01, vq_A1, 3); \
    vq_C72 = vfmaq_laneq_f32(vq_C72, vq_B02, vq_A1, 3); \
    vq_A1 = vld1q_f32(AA); AA += 4; 

// vq_B1X, vq_A0 loaded, vq_A1 loading 
#define MKSMM_m8n12_PACKB_END_K1 \
    vq_C00 = vfmaq_laneq_f32(vq_C00, vq_B10, vq_A0, 0); \
    vq_C01 = vfmaq_laneq_f32(vq_C01, vq_B11, vq_A0, 0); \
    vq_C02 = vfmaq_laneq_f32(vq_C02, vq_B12, vq_A0, 0); \
    vq_C10 = vfmaq_laneq_f32(vq_C10, vq_B10, vq_A0, 1); \
    vq_C11 = vfmaq_laneq_f32(vq_C11, vq_B11, vq_A0, 1); \
    vq_C12 = vfmaq_laneq_f32(vq_C12, vq_B12, vq_A0, 1); \
    vq_C20 = vfmaq_laneq_f32(vq_C20, vq_B10, vq_A0, 2); \
    vq_C21 = vfmaq_laneq_f32(vq_C21, vq_B11, vq_A0, 2); \
    vq_C22 = vfmaq_laneq_f32(vq_C22, vq_B12, vq_A0, 2); \
    vq_C30 = vfmaq_laneq_f32(vq_C30, vq_B10, vq_A0, 3); \
    vq_C31 = vfmaq_laneq_f32(vq_C31, vq_B11, vq_A0, 3); \
    vq_C32 = vfmaq_laneq_f32(vq_C32, vq_B12, vq_A0, 3); \
    \
    vst1q_f32(pkBB, vq_B10); pkBB += 4; \
    vq_C40 = vfmaq_laneq_f32(vq_C40, vq_B10, vq_A1, 0); \
    vq_C41 = vfmaq_laneq_f32(vq_C41, vq_B11, vq_A1, 0); \
    vq_C42 = vfmaq_laneq_f32(vq_C42, vq_B12, vq_A1, 0); \
    vst1q_f32(pkBB, vq_B11); pkBB += 4; \
    vq_C50 = vfmaq_laneq_f32(vq_C50, vq_B10, vq_A1, 1); \
    vq_C51 = vfmaq_laneq_f32(vq_C51, vq_B11, vq_A1, 1); \
    vq_C52 = vfmaq_laneq_f32(vq_C52, vq_B12, vq_A1, 1); \
    vst1q_f32(pkBB, vq_B12); pkBB += 4; \
    vq_C60 = vfmaq_laneq_f32(vq_C60, vq_B10, vq_A1, 2); \
    vq_C61 = vfmaq_laneq_f32(vq_C61, vq_B11, vq_A1, 2); \
    vq_C62 = vfmaq_laneq_f32(vq_C62, vq_B12, vq_A1, 2); \
    vq_C70 = vfmaq_laneq_f32(vq_C70, vq_B10, vq_A1, 3); \
    vq_C71 = vfmaq_laneq_f32(vq_C71, vq_B11, vq_A1, 3); \
    vq_C72 = vfmaq_laneq_f32(vq_C72, vq_B12, vq_A1, 3); 

// Pack B ] ----------------------------------------------------


// 忽略不对齐情况
void mkSMM_f32_m8n12_pkAB(float32_t *C, float32_t *A, float32_t *B, 
                        uint M, uint N, uint K, 
                        uint LN, uint LK, 
                        float32_t *pkB, 
                        uint outer_k)
{
    float32_t *AA = A;
    float32_t *pkBB = pkB;
    float32x4_t vq_A0, vq_A1, 
        vq_B00, vq_B01, vq_B02, vq_B10, vq_B11, vq_B12,
        vq_C00, vq_C01, vq_C02, 
        vq_C10, vq_C11, vq_C12, 
        vq_C20, vq_C21, vq_C22, 
        vq_C30, vq_C31, vq_C32, 
        vq_C40, vq_C41, vq_C42, 
        vq_C50, vq_C51, vq_C52, 
        vq_C60, vq_C61, vq_C62, 
        vq_C70, vq_C71, vq_C72; // 24 + 6 + 2 = 32

    for (int j = 0; j < N; j += 12) {
        float32_t *C0 = C + j;
        float32_t *C1 = C0 + 1 * LN;
        float32_t *C2 = C0 + 2 * LN;
        float32_t *C3 = C0 + 3 * LN;
        float32_t *C4 = C0 + 4 * LN;
        float32_t *C5 = C0 + 5 * LN;
        float32_t *C6 = C0 + 6 * LN;
        float32_t *C7 = C0 + 7 * LN;
        
        float32_t *B0 = B + j;
        float32_t *B1 = B0 + LN;

        // 打包重排 B[0:K][j:j+11] -> SB[0:K][0:11]
        // 顺便算第一个 m8 循环 A[0:7][0:K] x B[0:K][j:j+11]
        MKSMM_m8n12_PACKB_BEGIN_K0
        for (int k = 0; k < K; k += 8) {
            if (k != 0) 
                MKSMM_m8n12_PACKB_K0
            MKSMM_m8n12_PACKB_K1
            MKSMM_m8n12_PACKB_K0
            MKSMM_m8n12_PACKB_K1
            MKSMM_m8n12_PACKB_K0
            MKSMM_m8n12_PACKB_K1
            MKSMM_m8n12_PACKB_K0
            if (k+8 <= K) 
                MKSMM_m8n12_PACKB_K1
        }
        MKSMM_m8n12_PACKB_END_K1

        if (outer_k != 0) MKSMM_m8n12_ADD_C
        MKSMM_m8n12_STORE_C

    }
}

#define Blk_K 320
#define Blk_M 256
void kernelSMM_f32_pkAB_single(float32_t *C, float32_t *A, float32_t *B, 
                            uint M, uint N, uint K)
{
    uint LN = N, LK = K;

    float32_t *pkA, *pkB;
    posix_memalign((void **)&pkA, 64, Blk_M * Blk_K * sizeof(float32_t));  // Pack A: Mc x Kc 
    posix_memalign((void **)&pkB, 64, Blk_K * 12 * sizeof(float32_t));      // Pack B: Kc x 12

    uint bk, bm;  // blk size

    // TODO: jki ?
    for (uint j = 0; j < LN; j ++) {
        for (uint k = 0; k < LK; k += bk) {
            bk = LK-k < Blk_K ? LK-k : Blk_K;

            float32_t *BB = B + k * LN + j;

            for (uint i = 0; i < M; i += bm) {
                bm = M-i < Blk_M ? M-i : Blk_M; 

                float32_t *AA = A + i * LK + k;
                float32_t *CC = C + i * LN + j;

                packSMM_f32_A_k4(AA, pkA, bm, bk, LK);

                mkSMM_f32_m8n12_pkAB(
                    CC, pkA, BB,    // data
                    bm, N, bk,      // m n k
                    N, LK,          // LN LK  534535
                    pkB, 
                    k);  // is outer-k the first block ?  
            }
        }
    }

    free(pkA);
    free(pkB);
}
