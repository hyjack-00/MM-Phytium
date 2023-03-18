#include "microkernels.h"

// NEON example [original]
void mks32_0(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC)
{
    size_t a, b, c;
    int32x4_t A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3;

    for (int i = 0; i < ni; i += 4) {
        for (int j = 0; j < nj; j += 4) {
            C0 = vld1q_s32(C + c + LDBC*0);
            C1 = vld1q_s32(C + c + LDBC*1);
            C2 = vld1q_s32(C + c + LDBC*2);
            C3 = vld1q_s32(C + c + LDBC*3);

            for (int k = 0; k < nk; k += 4) {
                a = i*LDA + k;
                b = k*L + j;

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

            c = i*LDBC + j;
            vst1q_s32(C + c + LDBC*0, C0);
            vst1q_s32(C + c + LDBC*1, C1);
            vst1q_s32(C + c + LDBC*2, C2);
            vst1q_s32(C + c + LDBC*3, C3);
        }
    } 
}


// NEON example: load B, fetch C
#define s32_444lBfC_vA(k) \
    A[k] = vld1q_f32(A + a + LDA*k); \
    C[k] = vfmaq_laneq_s32(C[k], B[k], A[0], k); \
    C[k] = vfmaq_laneq_s32(C[k], B[k], A[1], k); \
    C[k] = vfmaq_laneq_s32(C[k], B[k], A[2], k); \
    C[k] = vfmaq_laneq_s32(C[k], B[k], A[3], k);

void mks32_4x4k4_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC) 
{
    size_t a, b, c;
    int32x4_t A[4], B[4], C[4];

    for (int i = 0; i < ni; i += 4) {
        for (int j = 0; j < nj; j += 4) {
            C[0] = vld1q_s32(C + c + LDBC*0);
            C[1] = vld1q_s32(C + c + LDBC*1);
            C[2] = vld1q_s32(C + c + LDBC*2);
            C[3] = vld1q_s32(C + c + LDBC*3);

            for (int k = 0; k < nk; k += 4) {
                a = i*LDA + k;
                b = k*LDBC + j;
                B[0] = vld1q_s32(B + b + LDBC*0);
                B[1] = vld1q_s32(B + b + LDBC*1);
                B[2] = vld1q_s32(B + b + LDBC*2);
                B[3] = vld1q_s32(B + b + LDBC*3);

                s32_444lBfC_vA(0);
                s32_444lBfC_vA(1);
                s32_444lBfC_vA(2);
                s32_444lBfC_vA(3);
            }
            c = i*LDBC + j;
            vst1q_s32(C + c + LDBC*0, C[0]);
            vst1q_s32(C + c + LDBC*1, C[1]);
            vst1q_s32(C + c + LDBC*2, C[2]);
            vst1q_s32(C + c + LDBC*3, C[3]);
        }
    }
}

// NEON example: load A, fetch C
#define s32_444lAfC_vB(k) \
    B[k] = vld1q_f32(B + b + LDBC*k); \
    C[0] = vfmaq_laneq_s32(C[0], A[0], B[k], 0); \
    C[1] = vfmaq_laneq_s32(C[1], A[1], B[k], 1); \
    C[2] = vfmaq_laneq_s32(C[2], A[2], B[k], 2); \
    C[3] = vfmaq_laneq_s32(C[3], A[3], B[k], 3); 

void mks32_4x4k4_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC) 
{
    size_t a, b, c;
    int32x4_t A[4], B[4], C[4];

    for (int i = 0; i < ni; i += 4) {
        for (int j = 0; j < nj; j += 4) {
            C[0] = vld1q_s32(C + c + LDBC*0);
            C[1] = vld1q_s32(C + c + LDBC*1);
            C[2] = vld1q_s32(C + c + LDBC*2);
            C[3] = vld1q_s32(C + c + LDBC*3);

            for (int k = 0; k < nk; k += 4) {
                a = i*LDA + k;
                b = k*LDBC + j;
                A[0] = vld1q_s32(A + a + LDA*0);
                A[1] = vld1q_s32(A + a + LDA*1);
                A[2] = vld1q_s32(A + a + LDA*2);
                A[3] = vld1q_s32(A + a + LDA*3);

                s32_444lAfC_vB(0);
                s32_444lAfC_vB(1);
                s32_444lAfC_vB(2);
                s32_444lAfC_vB(3);
            }
            c = i*LDBC + j;
            vst1q_s32(C + c + LDBC*0, C[0]);
            vst1q_s32(C + c + LDBC*1, C[1]);
            vst1q_s32(C + c + LDBC*2, C[2]);
            vst1q_s32(C + c + LDBC*3, C[3]);
        }
    }
}


void mks32_4x8k8_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC) 
{
    
}