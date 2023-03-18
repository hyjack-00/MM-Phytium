#include "microkernels.h"

// NEON example [original]
void mks32_0(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC)
{
    size_t a, b, c;
    int32x4_t A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3;

    for (size_t i = 0; i < ni; i += 4) {
        for (size_t j = 0; j < nj; j += 4) {
            c = i*LDBC + j;
            C0 = vld1q_s32(C + c + LDBC*0);
            C1 = vld1q_s32(C + c + LDBC*1);
            C2 = vld1q_s32(C + c + LDBC*2);
            C3 = vld1q_s32(C + c + LDBC*3);

            for (size_t k = 0; k < nk; k += 4) {
                a = i*LDA + k;
                b = k*LDBC + j;

                B0 = vld1q_s32(B + b + LDBC*0);
                B1 = vld1q_s32(B + b + LDBC*1);
                B2 = vld1q_s32(B + b + LDBC*2);
                B3 = vld1q_s32(B + b + LDBC*3);

                A0 = vld1q_s32(A + a + LDA*0);
                C0 = vmlaq_laneq_s32(C0, B0, A0, 0);
                C0 = vmlaq_laneq_s32(C0, B1, A0, 1);
                C0 = vmlaq_laneq_s32(C0, B2, A0, 2);
                C0 = vmlaq_laneq_s32(C0, B3, A0, 3);

                A1 = vld1q_s32(A + a + LDA*1);
                C1 = vmlaq_laneq_s32(C1, B0, A1, 0);
                C1 = vmlaq_laneq_s32(C1, B1, A1, 1);
                C1 = vmlaq_laneq_s32(C1, B2, A1, 2);
                C1 = vmlaq_laneq_s32(C1, B3, A1, 3);

                A2 = vld1q_s32(A + a + LDA*2);
                C2 = vmlaq_laneq_s32(C2, B0, A2, 0);
                C2 = vmlaq_laneq_s32(C2, B1, A2, 1);
                C2 = vmlaq_laneq_s32(C2, B2, A2, 2);
                C2 = vmlaq_laneq_s32(C2, B3, A2, 3);

                A3 = vld1q_s32(A + a + LDA*3);
                C3 = vmlaq_laneq_s32(C3, B0, A3, 0);
                C3 = vmlaq_laneq_s32(C3, B1, A3, 1);
                C3 = vmlaq_laneq_s32(C3, B2, A3, 2);
                C3 = vmlaq_laneq_s32(C3, B3, A3, 3);
            }

            vst1q_s32(C + c + LDBC*0, C0);
            vst1q_s32(C + c + LDBC*1, C1);
            vst1q_s32(C + c + LDBC*2, C2);
            vst1q_s32(C + c + LDBC*3, C3);
        }
    } 
}


// NEON example: load B
#define s32_444lBfC_vA(k) \
    vA[k] = vld1q_s32(A + a + LDA*k); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[0], vA[k], 0); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[1], vA[k], 1); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[2], vA[k], 2); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[3], vA[k], 3); 

void mks32_4x4k4_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC) 
{
    size_t a, b, c;
    int32x4_t vA[4], vB[4], vC[4];

    for (size_t i = 0; i < ni; i += 4) {
        for (size_t j = 0; j < nj; j += 4) {
            c = i*LDBC + j;
            vC[0] = vld1q_s32(C + c + LDBC*0);
            vC[1] = vld1q_s32(C + c + LDBC*1);
            vC[2] = vld1q_s32(C + c + LDBC*2);
            vC[3] = vld1q_s32(C + c + LDBC*3);

            for (size_t k = 0; k < nk; k += 4) {
                a = i*LDA + k;
                b = k*LDBC + j;
                vB[0] = vld1q_s32(B + b + LDBC*0);
                vB[1] = vld1q_s32(B + b + LDBC*1);
                vB[2] = vld1q_s32(B + b + LDBC*2);
                vB[3] = vld1q_s32(B + b + LDBC*3);

                s32_444lBfC_vA(0);
                s32_444lBfC_vA(1);
                s32_444lBfC_vA(2);
                s32_444lBfC_vA(3);
            }
            vst1q_s32(C + c + LDBC*0, vC[0]);
            vst1q_s32(C + c + LDBC*1, vC[1]);
            vst1q_s32(C + c + LDBC*2, vC[2]);
            vst1q_s32(C + c + LDBC*3, vC[3]);
        }
    }
}