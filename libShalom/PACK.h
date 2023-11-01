#include <stdio.h>
// #include <cblas.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>

#define num 1

#define GEMM_K 320
#define GEMM_M 256

void PACKA(float* A, float* Ac, long M, long K, long LK)
{
    long ii, jj = 0, kk = 0;

    for( ii = 0 ; ii < M; ii = ii + 8)
    {
        float *temp = A + ii * LK + kk;

        asm volatile(
                "   ldr     x0, %[Ac]               \n"
                "   ldr     x1, %[K]                \n"
                "   ldr     x2, %[temp]             \n"
                "   ldr     x30, %[LK]              \n"

                // x2-x9: temp[0-7][]
                "   add     x3, x2, x30, lsl #2     \n"  // x3 = temp + (LK * fp32)                row[1]  
                "   add     x4, x2, x30, lsl #3     \n"  // x4 = temp + (LK * 2 fp32)                row[2]
                "   add     x5, x3, x30, lsl #3     \n"  // x5 = temp + (LK * 4) + (LK * 8)     row[3]
                "   add     x6, x4, x30, lsl #3     \n"  // x6 = temp + (LK * 8) + (LK * 8)
                "   add     x7, x5, x30, lsl #3     \n"  
                "   add     x8, x6, x30, lsl #3     \n"
                "   add     x9, x7, x30, lsl #3     \n"

                "   lsr     x21, x1, #3             \n"  // x21 = K / 8     (kk)
                "   cmp     x21, #0                 \n"
                "   beq     PACKA_END               \n"  // K/8 == 0, end?

                "PACKA:                             \n"

                "   prfm    PLDL1KEEP, [x2, #128]   \n"  // prefetch temp[0][128]
                "   prfm    PLDL1KEEP, [x3, #128]   \n"  // prefetch temp[1][128]

                "   ldr     q0, [x2], #16           \n"  // v0 = [0+i][0 1 2 3] +k
                "   ldr     q1, [x3], #16           \n"  // v1 = [1+i][0 1 2 3]
                "   ldr     q2, [x4], #16           \n"  // v2 = [2+i][0 1 2 3]
                "   ldr     q3, [x5], #16           \n"  // v3 = [3+i][0 1 2 3]

                "   prfm    PLDL1KEEP, [x4, #128]   \n"
                "   prfm    PLDL1KEEP, [x5, #128]   \n"

                // Ac[0][0-3] = { [0+i][0], [1+i][0], [2+i][0], [3+i][0] }  Transposed Col-0(8) as an Ac Row-0(8)
                "   st4     {v0.s, v1.s, v2.s, v3.s}[0], [x0], #16  \n"  


                "   ldr     q4, [x6], #16           \n"  // v4 = [4+i][0 1 2 3]
                "   ldr     q5, [x7], #16           \n"
                "   ldr     q6, [x8], #16           \n"
                "   ldr     q7, [x9], #16           \n"

                // Ac[0][4-7] = { [0+i][0], [1+i][0], [2+i][0], [3+i][0] }  Transposed Col-0 (2)
                "   st4     {v4.s, v5.s, v6.s, v7.s}[0], [x0], #16  \n"

                "   prfm    PLDL1KEEP, [x6, #128]   \n"
                "   prfm    PLDL1KEEP, [x7, #128]   \n"

                "   ldr     q8, [x2], #16                           \n"  // v8 = [0+i][4 5 6 7] 交叉 load store
                "   st4     {v0.s, v1.s, v2.s, v3.s}[1], [x0], #16  \n"  // Ac[1][0-3] = { [0+i][1], [1+i][1]... }
                "   ldr     q9, [x3], #16                           \n"  
                "   st4     {v4.s, v5.s, v6.s, v7.s}[1], [x0], #16  \n"  // Ac[1][5-7] = { [4+i][1], [5+i][1]... }

                "   ldr     q10, [x4], #16                          \n" 
                "   st4     {v0.s, v1.s, v2.s, v3.s}[2], [x0], #16  \n"  // Ac[2]
                "   ldr     q11, [x5], #16                          \n"
                "   st4     {v4.s, v5.s, v6.s, v7.s}[2], [x0], #16  \n"  // Ac[2]

                "   prfm    PLDL1KEEP, [x8, #128]   \n"
                "   prfm    PLDL1KEEP, [x9, #128]   \n"


                "   ldr     q12, [x6], #16                              \n"
                "   st4     {v0.s, v1.s, v2.s, v3.s}[3], [x0], #16      \n"  // Ac[3]
                "   ldr     q13, [x7], #16                              \n"
                "   st4     {v4.s, v5.s, v6.s, v7.s}[3], [x0], #16      \n"  // Ac[3]
                "   ldr     q14, [x8], #16                              \n"
                "   st4     {v8.s, v9.s, v10.s, v11.s}[0], [x0], #16    \n"  // Ac[4][0-3] = { [0+i][4], [1+i][4], [2+i][4], [3+i][4] }
                "   ldr     q15, [x9], #16                              \n"
                "   st4     {v12.s, v13.s, v14.s, v15.s}[0], [x0], #16  \n"  // Ac[4][4-7] = { [4+i][4], [5+i][4], [6+i][4], [7+i][4] }

                "   subs    x21, x21, #1            \n"  // kk ++

                "   st4     {v8.s, v9.s, v10.s, v11.s}[1], [x0], #16    \n"  // Ac[5]
                "   st4     {v12.s, v13.s, v14.s, v15.s}[1], [x0], #16  \n"  
                "   st4     {v8.s, v9.s, v10.s, v11.s}[2], [x0], #16    \n"  // Ac[6]
                "   st4     {v12.s, v13.s, v14.s, v15.s}[2], [x0], #16  \n"
                "   st4     {v8.s, v9.s, v10.s, v11.s}[3], [x0], #16    \n"  // Ac[7]
                "   st4     {v12.s, v13.s, v14.s, v15.s}[3], [x0], #16  \n"

                /*  以上循环体
                    Ac[kk+0 : kk+7][0 : 7] = A[ii+0 : ii+7][kk+0 : kk+7] (8*8) Transpose 

                    循环
                    kk = 0 : K/8
                */ 

                "   bgt     PACKA                   \n"

                "   ands    x22, x1, #7             \n"  // x22 = K % 8
                "   beq     PACKA_END               \n"  

                "   cmp     x22, #4                 \n"
                "   blt     K1_PACKA                \n"  // if (x22 < 4) then PACK by KK=1
                                                         // else PACK by KK=4 first

                "K4_PACKA:                          \n"

                "   ldr     q0, [x2], #16           \n"  // [0][0:3]
                "   ldr     q1, [x3], #16           \n"  // [1][0:3]
                "   ldr     q2, [x4], #16           \n"  // ..
                "   ldr     q3, [x5], #16           \n"

                "   st4     {v0.s, v1.s, v2.s, v3.s}[0], [x0], #16  \n"  // Ac[0][0:3] = { [0][0], [1][0], [2][0], [3][0] }
                "   ldr     q4, [x6], #16                           \n"  
                "   st4     {v0.s, v1.s, v2.s, v3.s}[1], [x0], #16  \n"  // Ac[0][4:7] = { [0][1], [1][1], ..             } 
                "   ldr     q5, [x7], #16                           \n"
                "   st4     {v0.s, v1.s, v2.s, v3.s}[2], [x0], #16  \n"  //              { [0][2] ..                      }
                "   ldr     q6, [x8], #16                           \n"
                "   st4     {v0.s, v1.s, v2.s, v3.s}[3], [x0], #16  \n"  //              { [0][3] ..                      }
                "   ldr     q7, [x9], #16                           \n"

                // ?? 与之前的排布不同，是 4x8 转置 x2

                "   st4     {v4.s, v5.s, v6.s, v7.s}[0], [x0], #16  \n"  // [4][0] [5][0]
                "   st4     {v4.s, v5.s, v6.s, v7.s}[1], [x0], #16  \n"  
                "   st4     {v4.s, v5.s, v6.s, v7.s}[2], [x0], #16  \n"
                "   st4     {v4.s, v5.s, v6.s, v7.s}[3], [x0], #16  \n"

                "   subs    x22, x22, #4            \n"  // x22 -= 4
                "   beq     PACKA_END               \n"

                "K1_PACKA:                          \n"

                "   ldr     s0, [x2], #4            \n"
                "   ldr     s1, [x3], #4            \n"
                "   ldr     s2, [x4], #4            \n"
                "   ldr     s3, [x5], #4            \n"

                "   subs    x22, x22, #1            \n"

                "   st4     {v0.s, v1.s, v2.s, v3.s}[0], [x0], #16  \n"  // Ac[0][0:3]
                "   ldr     s4, [x6], #4            \n"
                "   ldr     s5, [x7], #4            \n"
                "   ldr     s6, [x8], #4            \n"
                "   ldr     s7, [x9], #4            \n"

                "   st4     {v4.s, v5.s, v6.s, v7.s}[0], [x0], #16  \n"  // Ac[0][4:7]

                "   bgt     K1_PACKA                \n"

                "PACKA_END:                         \n"

                :
                :
                [temp] "m" (temp),
                [Ac] "m" (Ac),
                [K] "m" (K),
                [LK] "m" (LK)
                :
                "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
                "x9", "x10", "x11", "x12", "x13","x14", "x15", "x16",
                "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24","x25",
                "x26", "x27", "x28", "x30",  // , "x29"
                "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );

        Ac = Ac + K * 8;
    }
} 