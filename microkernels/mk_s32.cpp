#include "microkernels.h"

std::ofstream mk_ofs("output/mk_log.dat");

//# Micro-Kernel

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

/* Tried:

// NEON example: load B, fetch C
#define s32_444lBfC_vA(k) \
    vA = vld1q_s32(A + a + LDA*k); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[0], vA, 0); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[1], vA, 1); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[2], vA, 2); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[3], vA, 3);

void mks32_4x4k4_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC) 
{
    size_t a, b, c;
    int32x4_t vA, vB[4], vC[4];

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


// NEON example: load A, fetch C
#define s32_444lAfC_vB(k) \
    vB = vld1q_s32(B + b + LDBC*k); \
    vC[0] = vmlaq_laneq_s32(vC[0], vB, vA[0], k); \
    vC[1] = vmlaq_laneq_s32(vC[1], vB, vA[1], k); \
    vC[2] = vmlaq_laneq_s32(vC[2], vB, vA[2], k); \
    vC[3] = vmlaq_laneq_s32(vC[3], vB, vA[3], k); 

void mks32_4x4k4_ldA_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC) 
{
    size_t a, b, c;
    int32x4_t vA[4], vB, vC[4];

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
                vA[0] = vld1q_s32(A + a + LDA*0);
                vA[1] = vld1q_s32(A + a + LDA*1);
                vA[2] = vld1q_s32(A + a + LDA*2);
                vA[3] = vld1q_s32(A + a + LDA*3);

                s32_444lAfC_vB(0);
                s32_444lAfC_vB(1);
                s32_444lAfC_vB(2);
                s32_444lAfC_vB(3);
            }
            vst1q_s32(C + c + LDBC*0, vC[0]);
            vst1q_s32(C + c + LDBC*1, vC[1]);
            vst1q_s32(C + c + LDBC*2, vC[2]);
            vst1q_s32(C + c + LDBC*3, vC[3]);
        }
    }
}


// 488, Load B
#define s32_488lBfC_vA(k) \
    vA[0] = vld1q_s32(A + a + LDA * k); \
    vA[1] = vld1q_s32(A + a + LDA * k + 4); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[0],  vA[0], 0); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[2],  vA[0], 1); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[4],  vA[0], 2); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[6],  vA[0], 3); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[8],  vA[1], 0); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[10], vA[1], 1); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[12], vA[1], 2); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[14], vA[1], 3); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[1],  vA[0], 0); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[3],  vA[0], 1); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[5],  vA[0], 2); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[7],  vA[0], 3); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[9],  vA[1], 0); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[11], vA[1], 1); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[13], vA[1], 2); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[15], vA[1], 3); 
#define s32_488lBfC_load2B(k) \
    vB[k*2  ] = vld1q_s32(B + b + LDBC * k); \
    vB[k*2+1] = vld1q_s32(B + b + LDBC * k + 4);
    
void mks32_4x8k8_ldB_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC) 
{
    size_t a, b, c;
    int32x4_t vA[2], vB[16], vC[8];

    for (size_t i = 0; i < ni; i += 4) {
        for (size_t j = 0; j < nj; j += 8) {
            c = i*LDBC + j;
            vC[0] = vld1q_s32(C + c + LDBC*0);
            vC[1] = vld1q_s32(C + c + LDBC*0 + 4);
            vC[2] = vld1q_s32(C + c + LDBC*1);
            vC[3] = vld1q_s32(C + c + LDBC*1 + 4);
            vC[4] = vld1q_s32(C + c + LDBC*2);
            vC[5] = vld1q_s32(C + c + LDBC*2 + 4);
            vC[6] = vld1q_s32(C + c + LDBC*3);
            vC[7] = vld1q_s32(C + c + LDBC*3 + 4);

            for (size_t k = 0; k < nk; k += 8) {
                a = i*LDA + k;
                b = k*LDBC + j;
                s32_488lBfC_load2B(0);
                s32_488lBfC_load2B(1);
                s32_488lBfC_load2B(2);
                s32_488lBfC_load2B(3);
                s32_488lBfC_load2B(4);
                s32_488lBfC_load2B(5);
                s32_488lBfC_load2B(6);
                s32_488lBfC_load2B(7);

                s32_488lBfC_vA(0);
                s32_488lBfC_vA(1);
                s32_488lBfC_vA(2);
                s32_488lBfC_vA(3);
            }
            vst1q_s32(C + c + LDBC*0,       vC[0]);
            vst1q_s32(C + c + LDBC*0 + 4,   vC[1]);
            vst1q_s32(C + c + LDBC*1,       vC[2]);
            vst1q_s32(C + c + LDBC*1 + 4,   vC[3]);
            vst1q_s32(C + c + LDBC*2,       vC[4]);
            vst1q_s32(C + c + LDBC*2 + 4,   vC[5]);
            vst1q_s32(C + c + LDBC*3,       vC[6]);
            vst1q_s32(C + c + LDBC*3 + 4,   vC[7]);
        }
    }
}


// 488, Load A
#define s32_488lAfC_vB0(k) \
    vB[0] = vld1q_s32(B + b + LDA * k); \
    vB[1] = vld1q_s32(B + b + LDA * k + 4); \
    vC[0] = vmlaq_laneq_s32(vC[0], vB[0], vA[0], k); \
    vC[1] = vmlaq_laneq_s32(vC[1], vB[1], vA[0], k); \
    vC[2] = vmlaq_laneq_s32(vC[2], vB[0], vA[2], k); \
    vC[3] = vmlaq_laneq_s32(vC[3], vB[1], vA[2], k); \
    vC[4] = vmlaq_laneq_s32(vC[4], vB[0], vA[4], k); \
    vC[5] = vmlaq_laneq_s32(vC[5], vB[1], vA[4], k); \
    vC[6] = vmlaq_laneq_s32(vC[6], vB[0], vA[6], k); \
    vC[7] = vmlaq_laneq_s32(vC[7], vB[1], vA[6], k); 
#define s32_488lAfC_vB1(k) \
    vB[0] = vld1q_s32(B + b + LDA * (k+4)); \
    vB[1] = vld1q_s32(B + b + LDA * (k+4) + 4); \
    vC[0] = vmlaq_laneq_s32(vC[0], vB[0], vA[1], k); \
    vC[1] = vmlaq_laneq_s32(vC[1], vB[1], vA[1], k); \
    vC[2] = vmlaq_laneq_s32(vC[2], vB[0], vA[3], k); \
    vC[3] = vmlaq_laneq_s32(vC[3], vB[1], vA[3], k); \
    vC[4] = vmlaq_laneq_s32(vC[4], vB[0], vA[5], k); \
    vC[5] = vmlaq_laneq_s32(vC[5], vB[1], vA[5], k); \
    vC[6] = vmlaq_laneq_s32(vC[6], vB[0], vA[7], k); \
    vC[7] = vmlaq_laneq_s32(vC[7], vB[1], vA[7], k); 

void mks32_4x8k8_ldA_fchC(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    size_t LDA, size_t LDBC) 
{
    size_t a, b, c;
    int32x4_t vA[8], vB[2], vC[8];

    for (size_t i = 0; i < ni; i += 4) {
        for (size_t j = 0; j < nj; j += 8) {
            c = i*LDBC + j;
            vC[0] = vld1q_s32(C + c + LDBC*0);
            vC[1] = vld1q_s32(C + c + LDBC*0 + 4);
            vC[2] = vld1q_s32(C + c + LDBC*1);
            vC[3] = vld1q_s32(C + c + LDBC*1 + 4);
            vC[4] = vld1q_s32(C + c + LDBC*2);
            vC[5] = vld1q_s32(C + c + LDBC*2 + 4);
            vC[6] = vld1q_s32(C + c + LDBC*3);
            vC[7] = vld1q_s32(C + c + LDBC*3 + 4);

            for (size_t k = 0; k < nk; k += 8) {
                a = i*LDA + k;
                b = k*LDBC + j;
                vA[0] = vld1q_s32(A + a + LDA * 0);
                vA[1] = vld1q_s32(A + a + LDA * 0 + 4);
                vA[2] = vld1q_s32(A + a + LDA * 1);
                vA[3] = vld1q_s32(A + a + LDA * 1 + 4);
                vA[4] = vld1q_s32(A + a + LDA * 2);
                vA[5] = vld1q_s32(A + a + LDA * 2 + 4);
                vA[6] = vld1q_s32(A + a + LDA * 3);
                vA[7] = vld1q_s32(A + a + LDA * 3 + 4);

                s32_488lAfC_vB0(0);
                s32_488lAfC_vB0(1);
                s32_488lAfC_vB0(2);
                s32_488lAfC_vB0(3);
                s32_488lAfC_vB1(0);
                s32_488lAfC_vB1(1);
                s32_488lAfC_vB1(2);
                s32_488lAfC_vB1(3);
            }
            vst1q_s32(C + c + LDBC*0,       vC[0]);
            vst1q_s32(C + c + LDBC*0 + 4,   vC[1]);
            vst1q_s32(C + c + LDBC*1,       vC[2]);
            vst1q_s32(C + c + LDBC*1 + 4,   vC[3]);
            vst1q_s32(C + c + LDBC*2,       vC[4]);
            vst1q_s32(C + c + LDBC*2 + 4,   vC[5]);
            vst1q_s32(C + c + LDBC*3,       vC[6]);
            vst1q_s32(C + c + LDBC*3 + 4,   vC[7]);
        }
    }
}

Tried */ 


//# Micro-Kernel + Repacking

// 444, Load B, Repacking AB
#define s32_444lBfCpAB_vA(k) \
    vA = vld1q_s32(pA + k*4); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[0], vA, 0); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[1], vA, 1); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[2], vA, 2); \
    vC[k] = vmlaq_laneq_s32(vC[k], vB[3], vA, 3);

void mks32_4x4k4_ldB_fchC_pkAB(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk, size_t LDC)
{
    const int32_t *pA, *pB;
    int32_t *pC;
    int32x4_t vA, vB[4], vC[4];

    for (size_t i = 0; i < ni; i += 4) {
        for (size_t j = 0; j < nj; j += 4) {
            pC = C + i*LDC + j;
            vC[0] = vld1q_s32(pC + LDC*0);
            vC[1] = vld1q_s32(pC + LDC*1);
            vC[2] = vld1q_s32(pC + LDC*2);
            vC[3] = vld1q_s32(pC + LDC*3);

            pA = A + i*nk;
            pB = B + j*nk;
            for (size_t k = 0; k < nk; k += 4) {
                vB[0] = vld1q_s32(pB + 0);
                vB[1] = vld1q_s32(pB + 4);
                vB[2] = vld1q_s32(pB + 8);
                vB[3] = vld1q_s32(pB + 12);

                s32_444lBfCpAB_vA(0);
                s32_444lBfCpAB_vA(1);
                s32_444lBfCpAB_vA(2);
                s32_444lBfCpAB_vA(3);
                pB += 16;
                pA += 16;
            }

            vst1q_s32(pC + LDC*0, vC[0]);
            vst1q_s32(pC + LDC*1, vC[1]);
            vst1q_s32(pC + LDC*2, vC[2]);
            vst1q_s32(pC + LDC*3, vC[3]);
        }
    }
}


// 488, Load B, Repacking AB
#define s32_488lBfCpAB_vA(k) \
    vA[0] = vld1q_s32(pA + 8 * k); \
    vA[1] = vld1q_s32(pA + 8 * k + 4); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[0],  vA[0], 0); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[2],  vA[0], 1); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[4],  vA[0], 2); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[6],  vA[0], 3); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[8],  vA[1], 0); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[10], vA[1], 1); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[12], vA[1], 2); \
    vC[2*k  ] = vmlaq_laneq_s32(vC[2*k  ], vB[14], vA[1], 3); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[1],  vA[0], 0); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[3],  vA[0], 1); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[5],  vA[0], 2); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[7],  vA[0], 3); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[9],  vA[1], 0); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[11], vA[1], 1); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[13], vA[1], 2); \
    vC[2*k+1] = vmlaq_laneq_s32(vC[2*k+1], vB[15], vA[1], 3); 
#define s32_488lBfCpAB_load2B(k) \
    vB[k*2  ] = vld1q_s32(pB + 8 * k); \
    vB[k*2+1] = vld1q_s32(pB + 8 * k + 4);
    
void mks32_4x8k8_ldB_fchC_pkAB(
    const int32_t *A, const int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk, size_t LDC) 
{
    const int32_t *pA, *pB;
    int32_t *pC;
    int32x4_t vA[2], vB[16], vC[8];

    for (size_t i = 0; i < ni; i += 4) {
        for (size_t j = 0; j < nj; j += 8) {
            pC = C + i*LDC + j;
            vC[0] = vld1q_s32(pC + LDC*0);
            vC[1] = vld1q_s32(pC + LDC*0 + 4);
            vC[2] = vld1q_s32(pC + LDC*1);
            vC[3] = vld1q_s32(pC + LDC*1 + 4);
            vC[4] = vld1q_s32(pC + LDC*2);
            vC[5] = vld1q_s32(pC + LDC*2 + 4);
            vC[6] = vld1q_s32(pC + LDC*3);
            vC[7] = vld1q_s32(pC + LDC*3 + 4);

            pA = A + i*nk;
            pB = B + j*nk;
            for (size_t k = 0; k < nk; k += 8) {
                s32_488lBfCpAB_load2B(0);
                s32_488lBfCpAB_load2B(1);
                s32_488lBfCpAB_load2B(2);
                s32_488lBfCpAB_load2B(3);
                s32_488lBfCpAB_load2B(4);
                s32_488lBfCpAB_load2B(5);
                s32_488lBfCpAB_load2B(6);
                s32_488lBfCpAB_load2B(7);

                s32_488lBfCpAB_vA(0);
                s32_488lBfCpAB_vA(1);
                s32_488lBfCpAB_vA(2);
                s32_488lBfCpAB_vA(3);
                pB += 64;
                pA += 32;
            }
            vst1q_s32(pC + LDC*0,       vC[0]);
            vst1q_s32(pC + LDC*0 + 4,   vC[1]);
            vst1q_s32(pC + LDC*1,       vC[2]);
            vst1q_s32(pC + LDC*1 + 4,   vC[3]);
            vst1q_s32(pC + LDC*2,       vC[4]);
            vst1q_s32(pC + LDC*2 + 4,   vC[5]);
            vst1q_s32(pC + LDC*3,       vC[6]);
            vst1q_s32(pC + LDC*3 + 4,   vC[7]);
        }
    }
}
