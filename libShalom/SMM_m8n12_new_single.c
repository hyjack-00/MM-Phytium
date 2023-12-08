#include "PACK.h"

static double gtod_ref_time_sec = 0.0;

double dclock()
{
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday( &tv, NULL );

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = ( double ) tv.tv_sec;

    norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}

void random_matrix( int m, int n, float *a)
{
    double drand48();
    int i,j;

    for ( i=0; i<m; i++ )
        for ( j=0; j<n; j++ )
        {
            a[i*n+j]= 2.0 * (float)drand48() - 1.0 + 0.000001 * (i+j);
        }
}

void transpose( int m, int n, float *a)
{
    float *temp_a = ( float * ) malloc( m* n * sizeof( float ) );

    int i, j;

    for( i =0 ;i< m; i++)
    {
        for( j= 0;j < n; j++)
        {
            temp_a[j * m + i] = a[i*n + j];
        }
    }


    for( i =0 ; i< n; i++)
    {
        for( j =0; j< m; j++)
        {
            a[ i * m+j] = temp_a[i*m+j];
        }
    }
}


void random_matrix1( int m, int n, float *a)
{
    double drand48(); 
    int i,j;

    for ( i=0; i<m; i++ )
        for ( j=0; j<n; j++ )
            a[i*n+j]= 1.0 ;
}


void SMM(float *C, float *A, float *B, 
            long M, long N, long K, 
            float *temp, 
            long LN, long LK, 
            float *SB, long k_tag)
{
    asm volatile(

        ".macro PACK_KERNEL8x12_BEGIN_K         \n" 
        // 列优先的 A[8][K]
        // 行优先的原始 B[KK][NN]  
        //  -> 重排到行优先的 SB[K][12]

        "   ldr     q0, [x11], #16              \n"  // v0 = A[0:3][0]  A increase
    

        "   prfm    PLDL1KEEP, [x11, #512]      \n"

        // A[0][0] * B0
        "   ldp     q2, q3, [x12]               \n"  //  v2 = B0[0:3]
                                                     //  v3 = B0[4:7]
        "   fmul    v8.4s, v2.4s, v0.s[0]       \n"  //  v8 = A00 * B0[0:3]
        "   ldr     q4, [x12, #32]              \n"  //  v4 = B0[8:11]
        "   fmul    v9.4s, v3.4s, v0.s[0]       \n"  //  v9 = A00 * B0[4:7]
        "   ldr     q1, [x11], #16              \n"  //  v1 = A[4:7][0]  A increase
        "   fmul    v10.4s, v4.4s, v0.s[0]      \n"  // v10 = A00 * B0[8:11]

        "   prfm    PLDL1KEEP, [x13, #64]       \n"

        // A[1][0] * B0
        "   add     x12, x12, x9, lsl #3        \n"  // x12 = B0 + LN * 2 * fp32 = B2 
        "   fmul    v11.4s, v2.4s, v0.s[1]      \n"  // v11 = A10 * B0[0:3]
        "   fmul    v12.4s, v3.4s, v0.s[1]      \n"  // ..
        "   fmul    v13.4s, v4.4s, v0.s[1]      \n"

        "   prfm    PLDL2KEEP, [x12, x30]       \n"
        "   ldr     q5, [x13]                   \n"  //  v5 = B1[0:3]

        // A[2][0] * B0
        "   fmul    v14.4s, v2.4s, v0.s[2]      \n" 
        "   fmul    v15.4s, v3.4s, v0.s[2]      \n"
        "   fmul    v16.4s, v4.4s, v0.s[2]      \n"

        "   prfm    PLDL1KEEP, [x12, #64]       \n"
        "   ldr     q6, [x13, #16]              \n"  //  v6 = B1[4:7]

        // A[3][0] * B0
        "   fmul    v17.4s, v2.4s, v0.s[3]      \n"
        "   fmul    v18.4s, v3.4s, v0.s[3]      \n"
        "   fmul    v19.4s, v4.4s, v0.s[3]      \n"

        "   ldr     q0, [x11], #16              \n"  //  v0 = A[0:3][1]  A increase

        // A[4][0] * B0
        "   fmul    v20.4s, v2.4s, v1.s[0]      \n" 
        "   fmul    v21.4s, v3.4s, v1.s[0]      \n"
        "   fmul    v22.4s, v4.4s, v1.s[0]      \n"

        "   ldr     q7, [x13, #32]              \n"  //  v7 = B1[8:11]

        // A[5][0] * B0
        "   fmul    v23.4s, v2.4s, v1.s[1]      \n"
        "   fmul    v24.4s, v3.4s, v1.s[1]      \n"
        "   fmul    v25.4s, v4.4s, v1.s[1]      \n"

        "   stp     q2, q3, [x24], #32          \n"  // SB[0][0:3] = v2 = B0[0:3]
                                                     // SB[0][4:7] = v3 = B0[4:7]
        "   add     x13, x13, x9, lsl #3        \n"  // x13 = B3

        // A[6][0] * B0
        "   fmul    v26.4s, v2.4s, v1.s[2]      \n"
        "   fmul    v27.4s, v3.4s, v1.s[2]      \n"
        "   fmul    v28.4s, v4.4s, v1.s[2]      \n"

        "   str     q4, [x24], #16              \n"  // SB[0][0] = v4 = B0[8:11]  SB increase
        "   prfm    PLDL2KEEP, [x13, x30]       \n"

        // A[7][0] * B0
        "   fmul    v29.4s, v2.4s, v1.s[3]      \n"
        "   fmul    v30.4s, v3.4s, v1.s[3]      \n"
        "   fmul    v31.4s, v4.4s, v1.s[3]      \n"

        "   ldr     q1, [x11], #16              \n"  // v1 = A[4:7][1]  A increase to A[][2]
                                                     // now v0:v1 loaded A[][1]

        ".endm                                                                          \n"



        ".macro PACK_KERNEL8x12_K0              \n"


        "   prfm PLDL1KEEP, [x11, #512]         \n"
        "   ldp     q5, q6, [x13]               \n"  // B3

        // A02 * B2
        "   fmla    v8.4s, v2.4s, v0.s[0]                           \n"
        "   fmla    v9.4s, v3.4s, v0.s[0]                           \n"
        "   fmla    v10.4s, v4.4s, v0.s[0]                          \n"

        "   ldr     q7, [x13, #32]              \n"  // B3

        // A12 * B2
        "   fmla    v11.4s, v2.4s, v0.s[1]                          \n"
        "   fmla    v12.4s, v3.4s, v0.s[1]                          \n"
        "   fmla    v13.4s, v4.4s, v0.s[1]                          \n"

        "   add     x13, x13, x9, lsl #3        \n"  // x13 = B5
        "   prfm PLDL2KEEP, [x13, x30]          \n"

        // A22 
        "   fmla    v14.4s, v2.4s, v0.s[2]                          \n"
        "   fmla    v15.4s, v3.4s, v0.s[2]                          \n"
        "   fmla    v16.4s, v4.4s, v0.s[2]                          \n"

        "   prfm PLDL1KEEP, [x13, #64]          \n"

        // A32
        "   fmla    v17.4s, v2.4s, v0.s[3]                          \n"
        "   fmla    v18.4s, v3.4s, v0.s[3]                          \n"
        "   fmla    v19.4s, v4.4s, v0.s[3]                          \n"

        "   ldr     q0, [x11], #16              \n"  // A[][2] inc

        // A42
        "   fmla    v20.4s, v2.4s, v1.s[0]                          \n"
        "   fmla    v21.4s, v3.4s, v1.s[0]                          \n"
        "   fmla    v22.4s, v4.4s, v1.s[0]                          \n"

        "   stp     q2, q3, [x24], #32          \n"  // SB

        // A52
        "   fmla    v23.4s, v2.4s, v1.s[1]                          \n"
        "   fmla    v24.4s, v3.4s, v1.s[1]                          \n"
        "   fmla    v25.4s, v4.4s, v1.s[1]                          \n"

        "   str     q4, [x24], #16              \n"  // SB

        // A62
        "   fmla    v26.4s, v2.4s, v1.s[2]                          \n"
        "   fmla    v27.4s, v3.4s, v1.s[2]                          \n"
        "   fmla    v28.4s, v4.4s, v1.s[2]                          \n"

        "   prfm PLDL1KEEP, [x11, #576]                             \n"

        // A72
        "   fmla    v29.4s, v2.4s, v1.s[3]                          \n"
        "   fmla    v30.4s, v3.4s, v1.s[3]                          \n"
        "   fmla    v31.4s, v4.4s, v1.s[3]                          \n"

        "   ldr     q1, [x11], #16              \n"  // A[][2] inc

        ".endm                                                                          \n"




        ".macro PACK_KERNEL8x12_K1                                  \n"
        // v0:v1 = A[0:7][1]
        // v5:v7 = B1[0:11]
        // x11 = &A[][2], x12 = B2, x13 = B3

        "   ldp     q2, q3, [x12]               \n"  // v2,v3 = B2[0:7]
        
        // A01 * B1
        "   fmla    v8.4s, v5.4s, v0.s[0]       \n"  // v8 += A01 * B1[0:3]
        "   fmla    v9.4s, v6.4s, v0.s[0]       \n"  // ..
        "   fmla    v10.4s, v7.4s, v0.s[0]      \n"

        "   ldr     q4, [x12, #32]              \n"  // v4 = B2[8:11]

        // A11 * B1
        "   fmla    v11.4s, v5.4s, v0.s[1]                          \n"
        "   fmla    v12.4s, v6.4s, v0.s[1]                          \n"
        "   fmla    v13.4s, v7.4s, v0.s[1]                          \n"

        
        "   add     x12, x12, x9, lsl #3        \n"  // x12 = B4
        "   prfm PLDL2KEEP, [x12, x30]                              \n"

        // A21 * B1
        "   fmla    v14.4s, v5.4s, v0.s[2]                          \n"
        "   fmla    v15.4s, v6.4s, v0.s[2]                          \n"
        "   fmla    v16.4s, v7.4s, v0.s[2]                          \n"

        "   prfm PLDL1KEEP, [x12, #64]                              \n"

        // A31 * B1
        "   fmla    v17.4s, v5.4s, v0.s[3]                          \n"
        "   fmla    v18.4s, v6.4s, v0.s[3]                          \n"
        "   fmla    v19.4s, v7.4s, v0.s[3]                          \n"

        "   ldr     q0, [x11], #16              \n"  // v0 = A[0:3][2]  increase

        // A41 * B1 
        "   fmla    v20.4s, v5.4s, v1.s[0]                          \n"
        "   fmla    v21.4s, v6.4s, v1.s[0]                          \n"
        "   fmla    v22.4s, v7.4s, v1.s[0]                          \n"

        "   stp     q5, q6, [x24], #32          \n"  // SB[1][0:3] = v5 = B1[0:3]
                                                     // SB[1][4:7] = v6 = B1[4:7]

        // A51
        "   fmla    v23.4s, v5.4s, v1.s[1]                          \n"
        "   fmla    v24.4s, v6.4s, v1.s[1]                          \n"
        "   fmla    v25.4s, v7.4s, v1.s[1]                          \n"

        "   str     q7, [x24], #16              \n"  // SB[1][8:11] = v7 = B1[8:11]  
                                                     // increase to SB[2][]

        // A61
        "   fmla    v26.4s, v5.4s, v1.s[2]                          \n"
        "   fmla    v27.4s, v6.4s, v1.s[2]                          \n"
        "   fmla    v28.4s, v7.4s, v1.s[2]                          \n"
        // A71
        "   fmla    v29.4s, v5.4s, v1.s[3]                          \n"
        "   fmla    v30.4s, v6.4s, v1.s[3]                          \n"
        "   fmla    v31.4s, v7.4s, v1.s[3]                          \n"

        "   ldr     q1, [x11], #16              \n"  // v1 = A[4:7][2]  increase

        ".endm                                                                          \n"



        ".macro PACK_KERNEL8x12_END_K                               \n"


        "   fmla    v8.4s, v5.4s, v0.s[0]                           \n"
        "   fmla    v9.4s, v6.4s, v0.s[0]                           \n"
        "   fmla    v10.4s, v7.4s, v0.s[0]                          \n"

        "   fmla    v11.4s, v5.4s, v0.s[1]                          \n"
        "   fmla    v12.4s, v6.4s, v0.s[1]                          \n"
        "   fmla    v13.4s, v7.4s, v0.s[1]                          \n"

        "   fmla    v14.4s, v5.4s, v0.s[2]                          \n"
        "   fmla    v15.4s, v6.4s, v0.s[2]                          \n"
        "   fmla    v16.4s, v7.4s, v0.s[2]                          \n"

        "   fmla    v17.4s, v5.4s, v0.s[3]                          \n"
        "   fmla    v18.4s, v6.4s, v0.s[3]                          \n"
        "   fmla    v19.4s, v7.4s, v0.s[3]                          \n"

        "   fmla    v20.4s, v5.4s, v1.s[0]                          \n"
        "   fmla    v21.4s, v6.4s, v1.s[0]                          \n"
        "   fmla    v22.4s, v7.4s, v1.s[0]                          \n"

        "   stp     q5, q6, [x24], #32          \n"  // SB

        "   fmla    v23.4s, v5.4s, v1.s[1]                          \n"
        "   fmla    v24.4s, v6.4s, v1.s[1]                          \n"
        "   fmla    v25.4s, v7.4s, v1.s[1]                          \n"

        "   str     q7, [x24], #16              \n"  // SB

        "   fmla    v26.4s, v5.4s, v1.s[2]                          \n"
        "   fmla    v27.4s, v6.4s, v1.s[2]                          \n"
        "   fmla    v28.4s, v7.4s, v1.s[2]                          \n"

        "   fmla    v29.4s, v5.4s, v1.s[3]                          \n"
        "   fmla    v30.4s, v6.4s, v1.s[3]                          \n"
        "   fmla    v31.4s, v7.4s, v1.s[3]                          \n"

        ".endm                                                                          \n"



        ".macro M8N12_PACK_ADD_C                                        \n"


        "   prfm    PLDL1KEEP, [x25, #64]                               \n"
        "   prfm    PLDL1KEEP, [x26, #64]                               \n"

        "   ldp     q0, q1, [x25]                                           \n"
        "   fadd    v8.4s, v8.4s, v0.4s                                 \n"
        "   prfm    PLDL1KEEP, [x27, #64]                               \n"
        "   ldr     q2, [x25, #32]                                          \n"
        "   fadd    v9.4s, v9.4s, v1.4s                                 \n"
        "   ldp     q3, q4, [x26]                                           \n"
        "   prfm    PLDL1KEEP, [x28, #64]                               \n"
        "   fadd    v10.4s, v10.4s, v2.4s                           \n"

        "   prfm    PLDL1KEEP, [x15, #64]                               \n"
        "   ldr     q5, [x26, #32]                                          \n"
        "   fadd    v11.4s, v11.4s, v3.4s                           \n"
        "   prfm    PLDL1KEEP, [x16, #64]                               \n"
        "   ldp     q6, q7, [x27]                                           \n"
        "   fadd    v12.4s, v12.4s, v4.4s                           \n"
        "   prfm    PLDL1KEEP, [x17, #64]                               \n"
        "   ldr     q0, [x27, #32]                                          \n"
        "   fadd    v13.4s, v13.4s, v5.4s                           \n"

        "   prfm    PLDL1KEEP, [x18, #64]                               \n"

        "   ldp     q1, q2, [x28]                                           \n"
        "   fadd    v14.4s, v14.4s, v6.4s                           \n"
        "   ldr     q3, [x28, #32]                                          \n"
        "   fadd    v15.4s, v15.4s, v7.4s                           \n"
        "   fadd    v16.4s, v16.4s, v0.4s                           \n"

        "   ldp     q4, q5, [x15]                                           \n"
        "   fadd    v17.4s, v17.4s, v1.4s                           \n"
        "   ldr     q6, [x15, #32]                                          \n"
        "   fadd    v18.4s, v18.4s, v2.4s                           \n"
        "   ldr     q7, [x16]                                                   \n"
        "   fadd    v19.4s, v19.4s, v3.4s                           \n"

        "   ldp     q0, q1, [x16, #16]                                  \n"
        "   fadd    v20.4s, v20.4s, v4.4s                           \n"
        "   ldp     q2, q3, [x17]                                           \n"
        "   fadd    v21.4s, v21.4s, v5.4s                           \n"
        "   ldr     q4, [x17, #32]                                          \n"
        "   fadd    v22.4s, v22.4s, v6.4s                           \n"


        "   ldp     q5, q6, [x18]                                           \n"
        "   fadd    v23.4s, v23.4s, v7.4s                           \n"
        "   ldr     q7, [x18, #32]                                          \n"
        "   fadd    v24.4s, v24.4s, v0.4s                           \n"
        "   fadd    v25.4s, v25.4s, v1.4s                           \n"

        "   fadd    v26.4s, v26.4s, v2.4s                           \n"
        "   fadd    v27.4s, v27.4s, v3.4s                           \n"
        "   fadd    v28.4s, v28.4s, v4.4s                           \n"

        "   fadd    v29.4s, v29.4s, v5.4s                           \n"
        "   fadd    v30.4s, v30.4s, v6.4s                           \n"
        "   fadd    v31.4s, v31.4s, v7.4s                           \n"

        ".endm                                                                          \n"



        ".macro SAVE8x12                                                        \n"


        "   subs    x21, x21, #1                \n"  // N12

        "   stp     q8, q9, [x25]               \n"  // C0[0:8]
        "   str     q10, [x25, #32]             \n"  // C0[8:11]
        "   add     x25, x25, x9, lsl #5        \n"  // x25 = C0 + LN * 8 * fp32 = C9

        "   prfm    PLDL2KEEP, [x25, x30]       \n"

        "   stp     q11, q12, [x26]             \n"  // C1
        "   str     q13, [x26, #32]             \n"
        "   add     x26, x26, x9, lsl #5        \n"

        "   prfm    PLDL2KEEP, [x26, #64]       \n"

        "   stp     q14, q15, [x27]             \n"  // C2
        "   str     q16, [x27, #32]             \n"
        "   add     x27, x27, x9, lsl #5        \n"

        "   prfm    PLDL2KEEP, [x27, #64]       \n"

        "   stp     q17, q18, [x28]             \n"  // C3
        "   str     q19, [x28, #32]             \n"
        "   add     x28, x28, x9, lsl #5        \n"

        "   prfm    PLDL2KEEP, [x28, #64]       \n"

        "   stp     q20, q21, [x15]             \n"  // C4
        "   str     q22, [x15, #32]             \n"
        "   add     x15, x15, x9, lsl #5        \n"

        "   prfm    PLDL2KEEP, [x15, #64]       \n"

        "   stp     q23, q24, [x16]             \n"  // C5
        "   str     q25, [x16, #32]             \n"
        "   add     x16, x16, x9, lsl #5        \n"

        "   prfm    PLDL2KEEP, [x16, #64]       \n"

        "   stp     q26, q27, [x17]             \n"  // C6
        "   str     q28, [x17, #32]             \n"
        "   add     x17, x17, x9, lsl #5        \n"

        "   prfm    PLDL2KEEP, [x17, #64]       \n"

        "   stp     q29, q30, [x18]             \n"  // C7
        "   str     q31, [x18, #32]             \n"
        "   add     x18, x18, x9, lsl #5        \n"

        "   prfm    PLDL2KEEP, [x18, #64]       \n"

        ".endm                                                                          \n"



        ".macro KERNEL8x12_BEGIN_K                                  \n"


        "   ldp     q0, q1, [x11], #32                                  \n"
        "   prfm    PLDL1KEEP, [x11, #2560]                         \n"
        "   ldr     q2, [x24]                                                       \n"

        "   fmul    v8.4s,  v2.4s, v0.s[0]                          \n"
        "   fmul    v11.4s, v2.4s, v0.s[1]                          \n"
        "   ldr     q3, [x24, #16]                                          \n"
        "   fmul    v14.4s, v2.4s, v0.s[2]                          \n"
        "   fmul    v17.4s, v2.4s, v0.s[3]                          \n"
        "   ldr     q4, [x24, #32]                                          \n"
        "   fmul    v20.4s, v2.4s, v1.s[0]                          \n"
        "   fmul    v23.4s, v2.4s, v1.s[1]                          \n"
        "   ldr     q7, [x24, #48]                                          \n"
        "   fmul    v26.4s, v2.4s, v1.s[2]                          \n"
        "   fmul    v29.4s, v2.4s, v1.s[3]                          \n"

        "   ldr     q2, [x24, #64]                                          \n"

        "   fmul    v9.4s,  v3.4s, v0.s[0]                          \n"
        "   fmul    v12.4s, v3.4s, v0.s[1]                          \n"
        "   ldr     q5, [x11], #16                                          \n"
        "   fmul    v15.4s, v3.4s, v0.s[2]                          \n"
        "   fmul    v18.4s, v3.4s, v0.s[3]                          \n"
        "   ldr     q6, [x11], #16                                          \n"
        "   fmul    v21.4s, v3.4s, v1.s[0]                          \n"
        "   fmul    v24.4s, v3.4s, v1.s[1]                          \n"
        "   fmul    v27.4s, v3.4s, v1.s[2]                          \n"
        "   fmul    v30.4s, v3.4s, v1.s[3]                          \n"

        "   ldr     q3, [x24, #80]                                          \n"

        "   fmul    v10.4s, v4.4s, v0.s[0]                          \n"
        "   fmul    v13.4s, v4.4s, v0.s[1]                          \n"
        "   fmul    v16.4s, v4.4s, v0.s[2]                          \n"
        "   prfm    PLDL1KEEP, [x11, #2560]                         \n"
        "   fmul    v19.4s, v4.4s, v0.s[3]                          \n"
        "   fmul    v22.4s, v4.4s, v1.s[0]                          \n"
        "   fmul    v25.4s, v4.4s, v1.s[1]                          \n"
        "   fmul    v28.4s, v4.4s, v1.s[2]                          \n"
        "   fmul    v31.4s, v4.4s, v1.s[3]                          \n"

        "   add     x24, x24, #96                                           \n"

        ".endm                                                                          \n"



        ".macro KERNEL8x12_K0                   \n"

        // A[0:7][K0] * B[K0][0:3]
        "   ldr     q5, [x11], #16              \n"  
        "   fmla    v8.4s,  v2.4s, v0.s[0]      \n"  
        "   fmla    v11.4s, v2.4s, v0.s[1]      \n"
        "   ldr     q6, [x11], #16              \n"
        "   fmla    v14.4s, v2.4s, v0.s[2]      \n"
        "   fmla    v17.4s, v2.4s, v0.s[3]      \n"
        "   ldr     q7, [x24]                   \n"
        "   fmla    v20.4s, v2.4s, v1.s[0]      \n"
        "   fmla    v23.4s, v2.4s, v1.s[1]      \n"
        "   fmla    v26.4s, v2.4s, v1.s[2]      \n"
        "   fmla    v29.4s, v2.4s, v1.s[3]      \n"

        "   ldr     q2, [x24, #16]              \n"

        // A[0:7][K0] * B[K0][4:7]
        "   fmla    v9.4s,  v3.4s, v0.s[0]      \n"
        "   fmla    v12.4s, v3.4s, v0.s[1]      \n"
        "   fmla    v15.4s, v3.4s, v0.s[2]      \n"
        "   fmla    v18.4s, v3.4s, v0.s[3]      \n"
        "   prfm PLDL1KEEP, [x11, #2560]        \n"
        "   fmla    v21.4s, v3.4s, v1.s[0]      \n"
        "   fmla    v24.4s, v3.4s, v1.s[1]      \n"
        "   fmla    v27.4s, v3.4s, v1.s[2]      \n"
        "   fmla    v30.4s, v3.4s, v1.s[3]      \n"
        
        "   ldr     q3, [x24, #32]              \n"

        // A[0:7][K0] * B[K0][8:11]
        "   fmla    v10.4s, v4.4s, v0.s[0]      \n"
        "   fmla    v13.4s, v4.4s, v0.s[1]      \n"
        "   fmla    v16.4s, v4.4s, v0.s[2]      \n"
        "   fmla    v19.4s, v4.4s, v0.s[3]      \n"
        "   prfm PLDL1KEEP, [x11, #2624]        \n"
        "   fmla    v22.4s, v4.4s, v1.s[0]      \n"
        "   fmla    v25.4s, v4.4s, v1.s[1]      \n"
        "   fmla    v28.4s, v4.4s, v1.s[2]      \n"
        "   fmla    v31.4s, v4.4s, v1.s[3]      \n"

        "   add     x24, x24, #48               \n"

        ".endm                                                                          \n"



        ".macro KERNEL8x12_K1                                               \n"

        "   ldr     q0, [x11], #16                                          \n"
        "   fmla    v8.4s,  v7.4s, v5.s[0]                          \n"
        "   fmla    v11.4s, v7.4s, v5.s[1]                          \n"
        "   ldr     q1, [x11], #16                                          \n"
        "   fmla    v14.4s, v7.4s, v5.s[2]                          \n"
        "   fmla    v17.4s, v7.4s, v5.s[3]                          \n"
        "   ldr     q4, [x24]                                                   \n"
        "   fmla    v20.4s, v7.4s, v6.s[0]                          \n"
        "   fmla    v23.4s, v7.4s, v6.s[1]                          \n"
        "   fmla    v26.4s, v7.4s, v6.s[2]                          \n"
        "   fmla    v29.4s, v7.4s, v6.s[3]                          \n"

        "   ldr     q7, [x24, #16]                                          \n"

        "   fmla    v9.4s,  v2.4s, v5.s[0]                          \n"
        "   fmla    v12.4s, v2.4s, v5.s[1]                          \n"
        "   fmla    v15.4s, v2.4s, v5.s[2]                          \n"
        "   fmla    v18.4s, v2.4s, v5.s[3]                          \n"
        "   prfm PLDL1KEEP, [x24, #256]                             \n"
        "   fmla    v21.4s, v2.4s, v6.s[0]                          \n"
        "   fmla    v24.4s, v2.4s, v6.s[1]                          \n"
        "   fmla    v27.4s, v2.4s, v6.s[2]                          \n"
        "   fmla    v30.4s, v2.4s, v6.s[3]                          \n"

        "   ldr     q2, [x24, #32]                                          \n"

        "   fmla    v10.4s, v3.4s, v5.s[0]                          \n"
        "   fmla    v13.4s, v3.4s, v5.s[1]                          \n"
        "   fmla    v16.4s, v3.4s, v5.s[2]                          \n"
        "   fmla    v19.4s, v3.4s, v5.s[3]                          \n"
        "   fmla    v22.4s, v3.4s, v6.s[0]                          \n"
        "   fmla    v25.4s, v3.4s, v6.s[1]                          \n"
        "   fmla    v28.4s, v3.4s, v6.s[2]                          \n"
        "   fmla    v31.4s, v3.4s, v6.s[3]                          \n"

        "   add     x24, x24, #48                                           \n"

        ".endm                                                                          \n"



        ".macro KERNEL8x12_K2                                               \n"

        "   ldr     q5, [x11], #16                                          \n"
        "   fmla    v8.4s,  v4.4s, v0.s[0]                          \n"
        "   fmla    v11.4s, v4.4s, v0.s[1]                          \n"
        "   ldr     q6, [x11], #16                                          \n"
        "   fmla    v14.4s, v4.4s, v0.s[2]                          \n"
        "   fmla    v17.4s, v4.4s, v0.s[3]                          \n"
        "   ldr     q3, [x24]                                                   \n"
        "   fmla    v20.4s, v4.4s, v1.s[0]                          \n"
        "   fmla    v23.4s, v4.4s, v1.s[1]                          \n"
        "   fmla    v26.4s, v4.4s, v1.s[2]                          \n"
        "   fmla    v29.4s, v4.4s, v1.s[3]                          \n"

        "   ldr     q4, [x24, #16]                                          \n"

        "   fmla    v9.4s,  v7.4s, v0.s[0]                          \n"
        "   fmla    v12.4s, v7.4s, v0.s[1]                          \n"
        "   fmla    v15.4s, v7.4s, v0.s[2]                          \n"
        "   fmla    v18.4s, v7.4s, v0.s[3]                          \n"
        "   prfm PLDL1KEEP, [x11, #2560]                            \n"
        "   fmla    v21.4s, v7.4s, v1.s[0]                          \n"
        "   fmla    v24.4s, v7.4s, v1.s[1]                          \n"
        "   fmla    v27.4s, v7.4s, v1.s[2]                          \n"
        "   fmla    v30.4s, v7.4s, v1.s[3]                          \n"

        "   ldr     q7, [x24, #32]                                          \n"

        "   fmla    v10.4s, v2.4s, v0.s[0]                          \n"
        "   fmla    v13.4s, v2.4s, v0.s[1]                          \n"
        "   fmla    v16.4s, v2.4s, v0.s[2]                          \n"
        "   prfm PLDL1KEEP, [x11, #2624]                            \n"
        "   fmla    v19.4s, v2.4s, v0.s[3]                          \n"
        "   fmla    v22.4s, v2.4s, v1.s[0]                          \n"
        "   fmla    v25.4s, v2.4s, v1.s[1]                          \n"
        "   fmla    v28.4s, v2.4s, v1.s[2]                          \n"
        "   fmla    v31.4s, v2.4s, v1.s[3]                          \n"

        "   add     x24, x24, #48                                           \n"

        ".endm                                                                          \n"



        ".macro KERNEL8x12_K3                                               \n"

        "   ldr     q0, [x11], #16                                          \n"
        "   fmla    v8.4s,  v3.4s, v5.s[0]                          \n"
        "   fmla    v11.4s, v3.4s, v5.s[1]                          \n"
        "   ldr     q1, [x11], #16                                          \n"
        "   fmla    v14.4s, v3.4s, v5.s[2]                          \n"
        "   fmla    v17.4s, v3.4s, v5.s[3]                          \n"
        "   ldr     q2, [x24]                                                   \n"
        "   fmla    v20.4s, v3.4s, v6.s[0]                          \n"
        "   fmla    v23.4s, v3.4s, v6.s[1]                          \n"
        "   fmla    v26.4s, v3.4s, v6.s[2]                          \n"
        "   fmla    v29.4s, v3.4s, v6.s[3]                          \n"

        "   ldr     q3, [x24, #16]                                          \n"

        "   fmla    v9.4s,  v4.4s, v5.s[0]                          \n"
        "   fmla    v12.4s, v4.4s, v5.s[1]                          \n"
        "   fmla    v15.4s, v4.4s, v5.s[2]                          \n"
        "   fmla    v18.4s, v4.4s, v5.s[3]                          \n"
        "   prfm PLDL1KEEP, [x24, #256]                             \n"
        "   fmla    v21.4s, v4.4s, v6.s[0]                          \n"
        "   fmla    v24.4s, v4.4s, v6.s[1]                          \n"
        "   fmla    v27.4s, v4.4s, v6.s[2]                          \n"
        "   fmla    v30.4s, v4.4s, v6.s[3]                          \n"

        "   ldr     q4, [x24, #32]                                          \n"

        "   fmla    v10.4s, v7.4s, v5.s[0]                          \n"
        "   fmla    v13.4s, v7.4s, v5.s[1]                          \n"
        "   fmla    v16.4s, v7.4s, v5.s[2]                          \n"
        "   fmla    v19.4s, v7.4s, v5.s[3]                          \n"
        "   fmla    v22.4s, v7.4s, v6.s[0]                          \n"
        "   fmla    v25.4s, v7.4s, v6.s[1]                          \n"
        "   fmla    v28.4s, v7.4s, v6.s[2]                          \n"
        "   fmla    v31.4s, v7.4s, v6.s[3]                          \n"

        "   add     x24, x24, #48                                           \n"

        ".endm                                                                          \n"



        ".macro KERNEL8x12_END_K                                        \n"

        "   fmla    v8.4s,  v3.4s, v5.s[0]                          \n"
        "   fmla    v11.4s, v3.4s, v5.s[1]                          \n"
        "   fmla    v14.4s, v3.4s, v5.s[2]                          \n"
        "   fmla    v17.4s, v3.4s, v5.s[3]                          \n"
        "   fmla    v20.4s, v3.4s, v6.s[0]                          \n"
        "   fmla    v23.4s, v3.4s, v6.s[1]                          \n"
        "   fmla    v26.4s, v3.4s, v6.s[2]                          \n"
        "   fmla    v29.4s, v3.4s, v6.s[3]                          \n"

        "   fmla    v9.4s,  v4.4s, v5.s[0]                          \n"
        "   fmla    v12.4s, v4.4s, v5.s[1]                          \n"
        "   fmla    v15.4s, v4.4s, v5.s[2]                          \n"
        "   fmla    v18.4s, v4.4s, v5.s[3]                          \n"
        "   fmla    v21.4s, v4.4s, v6.s[0]                          \n"
        "   fmla    v24.4s, v4.4s, v6.s[1]                          \n"
        "   fmla    v27.4s, v4.4s, v6.s[2]                          \n"
        "   fmla    v30.4s, v4.4s, v6.s[3]                          \n"

        "   fmla    v10.4s, v7.4s, v5.s[0]                          \n"
        "   fmla    v13.4s, v7.4s, v5.s[1]                          \n"
        "   fmla    v16.4s, v7.4s, v5.s[2]                          \n"
        "   fmla    v19.4s, v7.4s, v5.s[3]                          \n"
        "   fmla    v22.4s, v7.4s, v6.s[0]                          \n"
        "   fmla    v25.4s, v7.4s, v6.s[1]                          \n"
        "   fmla    v28.4s, v7.4s, v6.s[2]                          \n"
        "   fmla    v31.4s, v7.4s, v6.s[3]                          \n"

        ".endm                                  \n"



        //----------------------------------------------------

        "SMM_NN:                                \n"

        "   ldr     x0, %[C]                    \n"
        "   ldr     x1, %[A]                    \n"
        "   ldr     x2, %[B]                    \n"

        "   ldr     x3, %[M]                    \n"
        "   ldr     x4, %[N]                    \n"
        "   ldr     x5, %[K]                    \n"
        "   ldr     x7, %[temp]                 \n"
        "   ldr     x9, %[LN]                   \n"
        "   ldr     x6, %[LK]                   \n"

        "   prfm    PLDL1KEEP, [x1, #512]       \n"
        "   prfm    PLDL1KEEP, [x2, #64]        \n"

        "   ldr     x10, %[SB]                  \n" 
        "   ldr     x8, %[k_tag]                \n"


        "   mov     x21, #12                    \n"
        "   udiv    x20, x4, x21                \n"          // N / 12


        //---------------------------------------------------- loop N12

        "BEGIN_N12:                             \n"

        "   mov     x25, x0                     \n"   //C0*
        "   prfm    PLDL2KEEP, [x25, #64]       \n"
        "   add     x26, x25, x9, lsl #2        \n"   //C1*
        "   prfm    PLDL2KEEP, [x26, #64]       \n"
        "   add     x27, x25, x9, lsl #3        \n"   //C2*
        "   prfm    PLDL2KEEP, [x27, #64]       \n"
        "   add     x28, x26, x9, lsl #3        \n"   //C3*
        "   prfm    PLDL2KEEP, [x28, #64]       \n"

        "   add     x15, x27, x9, lsl #3        \n"   //C4*
        "   prfm    PLDL2KEEP, [x15, #64]       \n"
        "   add     x16, x28, x9, lsl #3        \n"   //C5*
        "   prfm    PLDL2KEEP, [x16, #64]       \n"
        "   add     x17, x15, x9, lsl #3        \n"   //C6*
        "   prfm    PLDL2KEEP, [x17, #64]       \n"
        "   add     x18, x16, x9, lsl #3        \n"   //C7*
        "   prfm    PLDL2KEEP, [x18, #64]       \n"

        "   mov     x11, x1                     \n"   // x11 = A*
        "   lsr     x21, x3, #3                 \n"   // x21 = M / 8

        "   mov     x30, #128                   \n"   // prfm  bytes to L2
        "   cmp     x20, #1                     \n"
        "   bgt     BEGIN_PACKB                 \n"
        "   mov     x30, #0                     \n"

        //----------------------------------------------------- PACK B

        "BEGIN_PACKB:                           \n"

        "   mov     x24, x10                    \n"   // x24 = SB*

        "   mov     x12, x2                     \n"   // x12 = B0*
        "   add     x13, x12, x9, lsl #2        \n"   // x13 = B0* + LN * fp32 = B1

        "   prfm    PLDL1KEEP, [x12, #64]       \n" 
        "   prfm    PLDL1KEEP, [x13, #64]       \n"
        "   prfm    PLDL1KEEP, [x11, #256]      \n"


        "PACK_Body_K:                           \n"

        "   lsr     x22, x5, #3                 \n"   // x22 = K / 8

        "   PACK_KERNEL8x12_BEGIN_K             \n"

        "   subs    x22, x22, #1                \n"   
        "   b       PACK_K1_7                   \n"

        "PACK_K:                                \n"
        
        "   PACK_KERNEL8x12_K0                  \n"

        "PACK_K1_7:                             \n"

        "   PACK_KERNEL8x12_K1                  \n"
        "   PACK_KERNEL8x12_K0                  \n"
        "   PACK_KERNEL8x12_K1                  \n"
        "   PACK_KERNEL8x12_K0                  \n"
        "   PACK_KERNEL8x12_K1                  \n"
        "   PACK_KERNEL8x12_K0                  \n"

        "   beq     PACK_Edge_K                 \n"  // 判的是 x22

        "   PACK_KERNEL8x12_K1                  \n"
        
        "   subs    x22, x22, #1                \n"
        "   b       PACK_K                      \n" 

        "PACK_Edge_K:                           \n"

        "   PACK_KERNEL8x12_END_K               \n"

        "   cmp     x8, #0                      \n"
        "   beq     M8N12_PACK_SAVE             \n"

        "   M8N12_PACK_ADD_C                    \n"  // if (k_tag != 0)

        "M8N12_PACK_SAVE:                       \n"

        "   SAVE8x12                            \n"

        "   beq     M8_END                      \n"  // if (x21 == 0) end



        //---------------------------------------------------------- loop M8

        "BEGIN_M8:                              \n" 
        // x11 = &A[8][0]
        // x21 = M / 8
        // x20 = N / 12 ...

        "   mov     x24, x10                    \n"  // x24 = SB 
        "   prfm    PLDL1KEEP, [x24, #128]      \n"
        "   prfm    PLDL1KEEP, [x11, #256]      \n"

        "Body_K:                                \n"

        "   lsr     x22, x5, #3                 \n"  // x22 = K / 8
        "   KERNEL8x12_BEGIN_K                  \n"
        "   subs    x22, x22, #1                \n"  
        "   b       K1_7                        \n"

        "Main_K:                                \n"
        
        "   KERNEL8x12_K0                       \n"

        "K1_7:                                  \n"
            
        "   KERNEL8x12_K1                       \n"
        "   KERNEL8x12_K2                       \n"
        "   KERNEL8x12_K3                       \n"
        "   KERNEL8x12_K0                       \n"
        "   KERNEL8x12_K1                       \n"
        "   KERNEL8x12_K2                       \n"

        "   beq     Edge_K                      \n"  // x22

        "   KERNEL8x12_K3                       \n" 
    
        "   subs    x22, x22, #1                \n"  
        "   b       Main_K                      \n"  

        "Edge_K:                                \n"

        "   KERNEL8x12_END_K                    \n"  

        "   cmp     x8, #0                      \n"  // k_tag
        "   beq     M8N12_SAVE                  \n"
        "   M8N12_PACK_ADD_C                    \n"

        "M8N12_SAVE:                            \n"

        "   SAVE8x12                            \n"
        
        "   bgt     BEGIN_M8                    \n"  // if (x21 != 0) loop-M8

        "M8_END:                                \n"

        "   subs    x20, x20, #1                \n"
        "   add     x0, x0, #48                 \n"  // C* += 12 * fp32
        "   add     x2, x2, #48                 \n"  // B* += 12 * fp32
        "   bgt     BEGIN_N12                   \n"  // if (x20 != 0) loop-N12

        :
        :   
        [C] "m" (C),
        [A] "m" (A),
        [B] "m" (B), 
        [M] "m" (M),
        [N] "m" (N),
        [K] "m" (K),
        [temp] "m" (temp),
        [LN] "m" (LN),
        [LK] "m" (LK),
        [SB] "m" (SB),
        [k_tag] "m" (k_tag)
        : 
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "x9", "x10", "x11", "x12", "x13","x14", "x15", "x16",
        "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24","x25",
        "x26", "x27", "x28", "x30",  // , "x29" 改用 x30
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );

}


void Small_SGEMM(float *C, float *A, float *B, 
                    long M, long N, long K, 
                    float *temp, long LN)
{
    omp_set_num_threads(num);
    long i, j, k, kc, mc;
    long LK = K;
    void *ptr, *ptr1;
    posix_memalign(&ptr, 64, num * GEMM_K * 12 * sizeof( float ));  // Pack B: Kc x 12
    posix_memalign(&ptr1, 64, GEMM_K * M * sizeof( float ));        // Pack A: M x Kc  这里好像有点问题
    float *SSB = (float *)ptr;
    float *Ac = (float *)ptr1;

    for(i = 0; i < LN; i = i + N)
    {
        for(k = 0; k < LK; k = k + kc)      // B block: GK x N (N不分块)
        { 
            kc = GEMM_K;
            if(LK - k < GEMM_K)
                kc = LK - k;

            float *BB = B + k * LN + i;     // B[k][i]

            for(j = 0; j < M; j = j + mc)   // A block: GM x GK
            {
                mc = GEMM_M;
                if(M - j < GEMM_M)
                    mc= M - j;
                float *AA = A + k + j * LK; // A[j][k]
                float *CC = C + i + j * LN; // C[j][i]

                PACKA(AA, Ac, mc, kc, LK);

                SMM(CC, Ac, BB, 
                    mc, N, kc, 
                    temp, 
                    LN, LK, 
                    &SSB[i/N * GEMM_K * 12],
                    k);
            }
        }
    }

    free(SSB);
    free(Ac);
}


int main()
{
    openblas_set_num_threads(num);
    int i,j,k;
    int loop = 5;
    long M, N, K;
    double start, cost;
    double gflops;
    long lda, ldb, ldc;
    int flag = 0 ;
    float temp = -1;
    float *A, *B, *C, *D;

    M = 256;
    N = 2400 * 2;
    K = 4000;
    lda = K;
    ldb = N;
    ldc = N;

    A = ( float * ) malloc( K* M * sizeof( float ) );
    B = ( float * ) malloc( K* N * sizeof( float ) );
    C = ( float * ) malloc( M* N * sizeof( float ) );
    D = ( float * ) malloc( M* N * sizeof( float ) );

    double ops = (double)M *N *K * 1.0e-09 * 2;

    random_matrix(M,K,A);
    random_matrix(K,N,B);

    for( i = 0; i < 2; i++)  // warmup
        Small_SGEMM(C, A, B, M, N/num, K, &temp, N);

    start = dclock();
    for( i = 0; i < loop ; i++)
        Small_SGEMM(C, A, B, M, N/num, K, &temp, N);
    cost =(dclock()-start)/loop; 

    printf("\nN_SMM:  M= %d N=%d K=%d flops = %lf effic= %.3lf %\n", 
        M, N, K, ops/cost, ops/cost/17.6 * 100/num);


    // for( i = 0; i < 2; i++)
    //     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, D, ldc);

    // start = dclock();
    // for( i = 0; i < loop ; i++)
    //     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, D, ldc);

    // cost=(dclock()-start)/loop;
    // printf("OpenBLAS:  M= %d N=%d K=%d cost = %lfms flops = %lf effic= %.3lf %\n\n", 
    //                         M, N, K, cost * 1000, ops/cost, ops/cost/17.6 * 100);


    // for( i= 0; i< M; i++)
    // {
    //     for( j= 0 ;j < N;j++)
    //     {
    //         if((C[i*N+j]- D[i*N+j] > 0.001 || C[i*N+j]- D[i*N+j] < -0.001) )
    //         {
    //             printf("i = %d, j= %d\n",i ,j );
    //             printf("C= %lf , D= %lf\n", C[i*N+j], D[i*N+j]);
    //             flag =1;
    //             printf("error\n");
    //             break;
    //         }
    //     }
    //     if(flag ==1)
    //         break;
    // }

    // if(flag == 0)
    //     printf("结果正确\n");

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}