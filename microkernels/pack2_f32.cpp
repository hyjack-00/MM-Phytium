#include "microkernels.h"

// 转置并按 m=8 个一行来重排
void packSMM_f32_A_k4(float32_t* A, float32_t* pkA, 
                    size_t M, size_t K, size_t LK) 
{
    size_t ii = 0, kk = 0;
    float32x4x4_t v40, v41;

    float32_t* pkA0 = pkA;
    float32_t* pkA1 = pkA0 + 8;
    float32_t* pkA2 = pkA1 + 8;
    float32_t* pkA3 = pkA2 + 8;

    for (ii = 0; ii < M; ii += 8) {
        float32_t *A0 = A + ii * LK;
        float32_t *A1 = A0 + LK;
        float32_t *A2 = A1 + LK;
        float32_t *A3 = A2 + LK;
        float32_t *A4 = A3 + LK;
        float32_t *A5 = A4 + LK;
        float32_t *A6 = A5 + LK;
        float32_t *A7 = A6 + LK;

        for (kk = 0; kk < K; kk += 4) {
            // Load upper 4 vec
            v40.val[0] = vld1q_f32(A0); A0 += 4;  // v0 = A0[0:3]
            v40.val[1] = vld1q_f32(A1); A1 += 4;  // v1 = A1[0:3]
            v40.val[2] = vld1q_f32(A2); A2 += 4;
            v40.val[3] = vld1q_f32(A3); A3 += 4;

            // Store upper 4 vec + Load downer 4 vec
            vst4q_lane_f32(pkA0, v40, 0);  // pkA0[0:3] = {v0[0], v1[0]..}
            pkA0 += 4;  // &pkA0[4:7]

            v41.val[0] = vld1q_f32(A4); A4 += 4;

            vst4q_lane_f32(pkA1, v40, 1);  // pkA1[0:3] = {v0[1], v1[1]..}
            pkA1 += 4;
            
            v41.val[1] = vld1q_f32(A5); A5 += 4;

            vst4q_lane_f32(pkA2, v40, 2); 
            pkA2 += 4;

            v41.val[2] = vld1q_f32(A6); A6 += 4;

            vst4q_lane_f32(pkA3, v40, 3); 
            pkA3 += 4;

            v41.val[3] = vld1q_f32(A7); A7 += 4;

            // Store downer 4 vec
            vst4q_lane_f32(pkA0, v41, 0);
            pkA0 += 32;  // Next loop-K
            vst4q_lane_f32(pkA1, v41, 1);
            pkA1 += 32;
            vst4q_lane_f32(pkA2, v41, 2);
            pkA2 += 32;
            vst4q_lane_f32(pkA3, v41, 3);
            pkA3 += 32;
        }
    }
}


void packSMM_f32_A_k8(float32_t* A, float32_t* pkA, 
                    size_t M, size_t K, size_t LK) 
{
    size_t ii = 0, kk = 0;
    float32x4x4_t v40, v41;

    float32_t* pkA0 = pkA;
    float32_t* pkA1 = pkA0 + 8;
    float32_t* pkA2 = pkA1 + 8;
    float32_t* pkA3 = pkA2 + 8;

    for (ii = 0; ii < M; ii += 8) {
        float32_t *A0 = A + ii * LK;
        float32_t *A1 = A0 + LK;
        float32_t *A2 = A1 + LK;
        float32_t *A3 = A2 + LK;
        float32_t *A4 = A3 + LK;
        float32_t *A5 = A4 + LK;
        float32_t *A6 = A5 + LK;
        float32_t *A7 = A6 + LK;

        for (kk = 0; kk < K; kk += 8) {
            // Load upper 4 vec(k0:3)
            v40.val[0] = vld1q_f32(A0); A0 += 4;  // v0 = A0[0:3]
            v40.val[1] = vld1q_f32(A1); A1 += 4;  // v1 = A1[0:3]
            v40.val[2] = vld1q_f32(A2); A2 += 4;
            v40.val[3] = vld1q_f32(A3); A3 += 4;

            // Store upper 4 vec(k0:3)
            // Load lower 4 vec(k0:3)
            vst4q_lane_f32(pkA0, v40, 0);  // pkA0[0:3] = {v0[0], v1[0]..}
            pkA0 += 4;  // &pkA0[4:7]
            v41.val[0] = vld1q_f32(A4); A4 += 4;

            vst4q_lane_f32(pkA1, v40, 1);  // pkA1[0:3] = {v0[1], v1[1]..}
            pkA1 += 4;
            v41.val[1] = vld1q_f32(A5); A5 += 4;

            vst4q_lane_f32(pkA2, v40, 2); 
            pkA2 += 4;
            v41.val[2] = vld1q_f32(A6); A6 += 4;

            vst4q_lane_f32(pkA3, v40, 3); 
            pkA3 += 4;
            v41.val[3] = vld1q_f32(A7); A7 += 4;

            // Store lower 4 vec(k0:3)
            // Load upper 4 vec(k4:7)
            vst4q_lane_f32(pkA0, v41, 0);
            pkA0 += 32;
            v40.val[0] = vld1q_f32(A0); A0 += 4;

            vst4q_lane_f32(pkA1, v41, 1);
            pkA1 += 32;
            v40.val[1] = vld1q_f32(A1); A1 += 4;
            
            vst4q_lane_f32(pkA2, v41, 2);
            pkA2 += 32;
            v40.val[2] = vld1q_f32(A2); A2 += 4;
            
            vst4q_lane_f32(pkA3, v41, 3);
            pkA3 += 32;
            v40.val[3] = vld1q_f32(A3); A3 += 4;

            // Store upper 4 vec(k4:7)
            // Load lower 4 vec(k4:7)
            vst4q_lane_f32(pkA0, v40, 0); 
            pkA0 += 4;
            v41.val[0] = vld1q_f32(A4); A4 += 4;

            vst4q_lane_f32(pkA1, v40, 1); 
            pkA1 += 4;
            v41.val[1] = vld1q_f32(A5); A5 += 4;

            vst4q_lane_f32(pkA2, v40, 2); 
            pkA2 += 4;
            v41.val[2] = vld1q_f32(A6); A6 += 4;

            vst4q_lane_f32(pkA3, v40, 3); 
            pkA3 += 4;
            v41.val[3] = vld1q_f32(A7); A7 += 4;

            // Store lower 4 vec(k4:7)
            vst4q_lane_f32(pkA0, v41, 0);
            pkA0 += 32;
            vst4q_lane_f32(pkA1, v41, 1);
            pkA1 += 32;
            vst4q_lane_f32(pkA2, v41, 2);
            pkA2 += 32;
            vst4q_lane_f32(pkA3, v41, 3);
            pkA3 += 32;
        }
    }
}

// TODO 考虑 Begin, End，循环时就可以完全重叠