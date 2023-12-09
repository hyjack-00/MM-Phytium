#include "microkernels.h"

// 488, Load B, Repacking AB
#define s32_488lBfCpAB_vA(k) \
    vA[0] = vld1q_f32(pA + 8 * k); \
    vA[1] = vld1q_f32(pA + 8 * k + 4); \
    vC[2*k  ] = vfmaq_laneq_f32(vC[2*k  ], vB[0],  vA[0], 0); \
    vC[2*k  ] = vfmaq_laneq_f32(vC[2*k  ], vB[2],  vA[0], 1); \
    vC[2*k  ] = vfmaq_laneq_f32(vC[2*k  ], vB[4],  vA[0], 2); \
    vC[2*k  ] = vfmaq_laneq_f32(vC[2*k  ], vB[6],  vA[0], 3); \
    vC[2*k  ] = vfmaq_laneq_f32(vC[2*k  ], vB[8],  vA[1], 0); \
    vC[2*k  ] = vfmaq_laneq_f32(vC[2*k  ], vB[10], vA[1], 1); \
    vC[2*k  ] = vfmaq_laneq_f32(vC[2*k  ], vB[12], vA[1], 2); \
    vC[2*k  ] = vfmaq_laneq_f32(vC[2*k  ], vB[14], vA[1], 3); \
    vC[2*k+1] = vfmaq_laneq_f32(vC[2*k+1], vB[1],  vA[0], 0); \
    vC[2*k+1] = vfmaq_laneq_f32(vC[2*k+1], vB[3],  vA[0], 1); \
    vC[2*k+1] = vfmaq_laneq_f32(vC[2*k+1], vB[5],  vA[0], 2); \
    vC[2*k+1] = vfmaq_laneq_f32(vC[2*k+1], vB[7],  vA[0], 3); \
    vC[2*k+1] = vfmaq_laneq_f32(vC[2*k+1], vB[9],  vA[1], 0); \
    vC[2*k+1] = vfmaq_laneq_f32(vC[2*k+1], vB[11], vA[1], 1); \
    vC[2*k+1] = vfmaq_laneq_f32(vC[2*k+1], vB[13], vA[1], 2); \
    vC[2*k+1] = vfmaq_laneq_f32(vC[2*k+1], vB[15], vA[1], 3); 
#define s32_488lBfCpAB_load2B(k) \
    vB[k*2  ] = vld1q_f32(pB + 8 * k); \
    vB[k*2+1] = vld1q_f32(pB + 8 * k + 4);
    
void mkf32_4x8k8_ldB_fchC_pkAB(
    const float32_t *A, const float32_t *B, float32_t *C,
    size_t ni, size_t nj, size_t nk, size_t LDC) 
{
    const float32_t *pA, *pB;
    float32_t *pC;
    float32x4_t vA[2], vB[16], vC[8];

    for (size_t i = 0; i < ni; i += 4) {
        for (size_t j = 0; j < nj; j += 8) {
            pC = C + i*LDC + j;
            vC[0] = vld1q_f32(pC + LDC*0);
            vC[1] = vld1q_f32(pC + LDC*0 + 4);
            vC[2] = vld1q_f32(pC + LDC*1);
            vC[3] = vld1q_f32(pC + LDC*1 + 4);
            vC[4] = vld1q_f32(pC + LDC*2);
            vC[5] = vld1q_f32(pC + LDC*2 + 4);
            vC[6] = vld1q_f32(pC + LDC*3);
            vC[7] = vld1q_f32(pC + LDC*3 + 4);

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
            vst1q_f32(pC + LDC*0,       vC[0]);
            vst1q_f32(pC + LDC*0 + 4,   vC[1]);
            vst1q_f32(pC + LDC*1,       vC[2]);
            vst1q_f32(pC + LDC*1 + 4,   vC[3]);
            vst1q_f32(pC + LDC*2,       vC[4]);
            vst1q_f32(pC + LDC*2 + 4,   vC[5]);
            vst1q_f32(pC + LDC*3,       vC[6]);
            vst1q_f32(pC + LDC*3 + 4,   vC[7]);
        }
    }
}