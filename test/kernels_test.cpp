#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <queue>

#include <omp.h>

#include "test_helpers.h"
#include "microkernels.h"

#define ANS_CHECK false  // 是否进行答案检查
#define OPTI_BLOCKING_MODE false  // 是否为分块大小测试模式

#if OPTI_BLOCKING_MODE == true
struct block_t {
    size_t Ti, Tj, Tk;
    float time;
};
bool operator<(const block_t &b1, const block_t &b2) {
    return b1.time < b2.time;
}
#endif



static size_t Ti = 256;
static size_t Tj = 256;
static size_t Tk = 256;
#define MIN(x,y) (((x)<(y))?(x):(y))

void kernel_s32_packAB(
    int32_t *A, int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    mks32_pAB_t mk, packs32_A_t pkA, packs32_B_t pkB) 
{
    int32_t *Apack = (int32_t *) malloc(sizeof(int32_t) * Ti * Tk);
    int32_t *Bpack = (int32_t *) malloc(sizeof(int32_t) * Tk * Tj);

    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        size_t it = MIN(ni-i0, Ti);
        for (size_t k0 = 0; k0 < nk; k0 += Tk) {
            size_t kt = MIN(nk-k0, Tk);
            for (size_t j0 = 0; j0 < nj; j0 += Tj) {
                size_t jt = MIN(nj-j0, Tj);
            
                pkA(A+i0*nk+k0, Apack, it, kt, nk);
                pkB(B+k0*nj+j0, Bpack, kt, jt, nj);
                // print_mat(Apack, it*kt/32, 32, "Apack");
                // print_mat(Bpack, kt*jt/64, 64, "Bpack");
                mk(Apack, Bpack, C+i0*nj+j0, it, jt, kt, nj);
            }
        }
    }
}
void kernel_s32_packABC(
    int32_t *A, int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    mks32_pABC_t mk, packs32_A_t pkA, packs32_B_t pkB, 
    packs32_C_t pkC, unpacks32_C_t upkC) 
{
    int32_t *Apack = (int32_t *) malloc(sizeof(int32_t) * Ti * Tk);
    int32_t *Bpack = (int32_t *) malloc(sizeof(int32_t) * Tk * Tj);
    int32_t *Cpack = (int32_t *) malloc(sizeof(int32_t) * Ti * Tj);

    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        for (size_t j0 = 0; j0 < nj; j0 += Tj) {
            size_t it = MIN(ni-i0, Ti);
            size_t jt = MIN(nj-j0, Tj);
            for (size_t k0 = 0; k0 < nk; k0 += Tk) {
                size_t kt = MIN(nk-k0, Tk);
                pkA(A+i0*nk+k0, Apack, it, kt, nk);
                pkB(B+k0*nj+j0, Bpack, kt, jt, nj);
                pkC(C+i0*nj+j0, Cpack, kt, jt, nj);

                mk(Apack, Bpack, Cpack, it, jt, kt);
                
                upkC(C+i0*nj+j0, Cpack, kt, jt, nj);
            }
        }
    }
}
void kernel_s32(
    int32_t *A, int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk, mks32_t mk) 
{
    // #pragma omp parallel for
    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        size_t it = MIN(ni-i0, Ti);
        for (size_t j0 = 0; j0 < nj; j0 += Tj) {
            size_t jt = MIN(nj-j0, Tj);
            for (size_t k0 = 0; k0 < nk; k0 += Tk) {
                size_t kt = MIN(nk-k0, Tk);
                mk(A+i0*nk+k0, B+k0*nj+j0, C+i0*nj+j0, it, jt, kt, nk, nj);
            }
        }
    }
}


#define TEST_N_S32 1024

void test_s32() {
    const int data_loop = 10;
    const int compute_loop = 5;
    // const int ni = 4, nj = 8, nk = 8;
    const int ni = TEST_N_S32, nj = TEST_N_S32, nk = TEST_N_S32;

    #if OPTI_BLOCKING_MODE == false
    // 性能测试模式 ==============================
    cout << "Standard Test start" << endl;
    cout << "Loop: " << data_loop << "x" << compute_loop
       << ", Size: i" << ni << " j" << nj << " k" << nk << endl;
    #if FILE_OUTPUT == true
    cout << "File output: " << ouput_file << endl; 
    #endif

    double total_time2 = 0;
    for (int data = 0; data < data_loop; data ++) {
        int32_t *A = (int32_t *) malloc(sizeof(int32_t) * ni * nk);
        int32_t *B = (int32_t *) malloc(sizeof(int32_t) * nk * nj);
        int32_t *C = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
        rand_mat_s32(A, ni * nk, 1234);
        rand_mat_s32(B, nk * nj, 5678);
        // rand_mat_s32(A, ni * nk, time(0));
        // rand_mat_s32(B, nk * nj, time(0)+1);

        // 连续计算 ==============================
        double total_time1 = 0;
        for (int compute = 0; compute < compute_loop; compute ++) {
            zeros_s32(C, ni * nj);
            auto start = Clock::now();

            // kernel_s32(A, B, C, ni, nj, nk, mks32_4x8k8_ldB_fchC);
            // kernel_s32_packAB(A, B, C, ni, nj, nk, 
            //     mks32_4x8k8_ldB_fchC_pkAB,
            //     packs32_4x8k8_A,
            //     packs32_4x8k8_B);
            // kernel_packABC_s32(A, B, C, ni, nj, nk, 
            //     mks32_4x8k8_ldB_fchC_pkABC, 
            //     packs32_4x8k8_A,
            //     packs32_4x8k8_B,
            //     packs32_4x8k8_C,
            //     unpacks32_4x8k8_C);

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000.0;
            total_time1 += dur; 
            cout << "compute time: " << dur << " msecs" << endl;
        }
        cout << "  avg time1: " << total_time1/compute_loop << " msecs for data: " << data << endl;
        total_time2 += total_time1/compute_loop;

        // 答案检查 ==============================
        #if ANS_CHECK == true  
            int32_t *D = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
            zeros_s32(D, ni * nj);
            kernel_s32(A, B, D, ni, nj, nk, mks32_0);
            ans_check_s32(C, D, ni, nj);

            print_mat(A, ni, nk, "A");
            print_mat(B, nk, nj, "B");
            print_mat(C, ni, nj, "C");
            print_mat(D, ni, nj, "D");
            free(D);
        #endif

        free(A);
        free(B);
        free(C);
    }
    cout << "    total avg time2: " << total_time2/data_loop << " msecs" << endl;



    #else  // OPTI_BLOCKING_MODE
    // 寻找最优分块模式 ==============================
    const size_t blk_unit = 32;
    const size_t blk_lb = 64 / blk_unit;
    const size_t blk_ub = ni / blk_unit;
    const int rand_loop = 20;
    const int topN = 40;
    srand(time(0));

    cout << "Block Size Optimizing Test start" << endl;
    cout << "Loop: " << data_loop << "x" << rand_loop << "x" << compute_loop << endl;
    cout << "Size: i" << ni << " j" << nj << " k" << nk << endl;
    #if FILE_OUTPUT == true
    cout << "File output: " << ouput_file << endl; 
    #endif


    
    std::priority_queue<block_t> pq_block;
    for (int i = 0; i < topN; i ++) pq_block.push({0,0,0,10000});

    float *record = (float *) malloc(sizeof(float) * 4 * data_loop * rand_loop);
    int idx_record = 0;

    for (int data = 0; data < data_loop; data ++) {
        int32_t *A = (int32_t *) malloc(sizeof(int32_t) * ni * nk);
        int32_t *B = (int32_t *) malloc(sizeof(int32_t) * nk * nj);
        int32_t *C = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
        rand_mat_s32(A, ni * nk, time(0));
        rand_mat_s32(B, nk * nj, time(0)+1);

        for (int randT = 0; randT < rand_loop; randT ++) {
            zeros_s32(C, ni * nj);
            Ti = (rand() % (blk_ub-blk_lb) + blk_lb) * blk_unit;
            Tj = (rand() % (blk_ub-blk_lb) + blk_lb) * blk_unit;
            Tk = (rand() % (blk_ub-blk_lb) + blk_lb) * blk_unit;

            double total_time = 0.;
            for (int compute = 0; compute < compute_loop; compute ++) {
                auto start = Clock::now();

                // kernel_s32(A, B, C, ni, nj, nk, mks32_8x4k8_ldA_fchC);
                kernel_s32_packAB(A, B, C, ni, nj, nk, 
                    mks32_4x8k8_ldB_fchC_pkAB, 
                    packs32_4x8k8_A,
                    packs32_4x8k8_B);
                // kernel_s32_packABC(A, B, C, ni, nj, nk, 
                //     mks32_4x8k8_ldB_fchC_pkABC, 
                //     packs32_4x8k8_A,
                //     packs32_4x8k8_B,
                //     packs32_4x8k8_C,
                //     unpacks32_4x8k8_C);

                auto end = Clock::now();
                double dur = Dur(start, end);
                total_time += dur / 1000.;
            }
            double t = total_time / compute_loop;
            cout << Ti << " " << Tj << " " << Tk << " " << t << " msecs" << endl;

            record[idx_record+0] = (float) Ti;
            record[idx_record+1] = (float) Tj;
            record[idx_record+2] = (float) Tk;
            record[idx_record+3] = (float) t;
            idx_record += 4;
            if (t < pq_block.top().time) {
                pq_block.pop();
                block_t blk = {Ti, Tj, Tk, (float)t};
                pq_block.push(blk);
            }
        }
        free(A);
        free(B);
        free(C);
    }
    print_mat(record, data_loop * rand_loop, 4, "blocking record");
    cout << endl << "============================ result ============================" << endl;
    while (!pq_block.empty()) {
        auto block = pq_block.top();
        pq_block.pop();
        cout << block.Ti << " " << block.Tj << " " << block.Tk << " " << block.time << " msecs" << endl;
    }
    free(record);
    #endif 



    // 收尾 ==============================
    char cur_time[50];
    time_t now = time(NULL);
    strftime(cur_time, 50, "%x %X", localtime(&now));
    cout << "Test Finished at " << cur_time << endl;

    #if FILE_OUTPUT == true
        ofs.close();
    #endif
}


#define TEST_N_F32 1024

void test_f32() {
    const int data_loop = 5;
    const int compute_loop = 5;
    const int ni = 4, nj = 8, nk = 8;
    // const int ni = TEST_N_F32, nj = TEST_N_F32, nk = TEST_N_F32;

    cout << "Loop: " << data_loop << "x" << compute_loop << endl;
    cout << "Size: i" << ni << " j" << nj << " k" << nk << endl;

    double total_time = 0;
    for (int data_i = 0; data_i < data_loop; data_i ++) {
        float32_t *A = (float32_t *) malloc(sizeof(float32_t) * ni * nk);
        float32_t *B = (float32_t *) malloc(sizeof(float32_t) * nk * nj);
        float32_t *C = (float32_t *) malloc(sizeof(float32_t) * ni * nj);
        rand_mat_f32(A, ni * nk, 1234);
        rand_mat_f32(B, nk * nj, 5678);
        // rand_mat_f32(A, ni * nk, time(0));
        // rand_mat_f32(B, nk * nj, time(0)+1);

        double time = 0;
        for (int compute_i = 0; compute_i < compute_loop; compute_i ++) {
            zeros_f32(C, ni * nj);
            auto start = Clock::now();

            /* Timing Zone in -- */

            kernelSMM_f32_pkAB_single(C, A, B, 
                                    ni, nj, nk, 
                                    nj);

            /* -- Timing Zone out */

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000.0;
            time += dur; 
            cout << "compute time: " << dur << " msecs" << endl;
        }
        cout << "  avg time: " << time/compute_loop << " msecs for data: " << data_i << endl;
        total_time += time/compute_loop;

        // Answer Check 
            float32_t *D = (float32_t *) malloc(sizeof(float32_t) * ni * nj);
            zeros_f32(D, ni * nj);
            naive(A, B, D, ni, nj, nk);
            ans_check_f32(C, D, ni, nj); 
            free(D);

        free(A);
        free(B);
        free(C);
    }
    cout << "    total avg time: " << total_time/data_loop << " msecs" << endl;
}


int main() {
    // test_s32();
    test_f32();
}