#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <queue>

#include <omp.h>

#include "microkernels.h"

using std::string;
using std::cout;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;
#define Dur(start,end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

#define FILE_OUTPUT false  // 是否输出从 stdout 到 文件
#define ANS_CHECK false  // 是否进行答案检查
#define OPTI_BLOCKING_MODE false  // 是否为分块大小测试模式

string ouput_file = "output/output.dat";
#if FILE_OUTPUT == true
    #define OS ofs
    std::ofstream ofs(ouput_file, std::ios::app);
#else
    #define OS cout
#endif

// 输入矩阵的随机数范围
#define RAND_UB 10  // [LB, UB)
#define RAND_LB -10

template <typename T>
void print_mat(T *mat, int ni, int nj, string msg="") {
    std::ofstream fout("output/matrix.dat", std::ios::app);
    if (!fout.is_open()) {
        cout << "Error opening file for result" << endl;
        exit(1);
    }
    fout << msg << endl;
    for (int i = 0; i < ni; i ++) {
        for (int j = 0; j < nj; j ++) {
            fout << mat[i*nj + j] << " ";
        }
        fout << endl;
    }
    fout << endl;
    fout.close();
}
template <typename T>
void naive(T *A, T *B, T *C, size_t ni, size_t nj, size_t nk) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < ni; i ++) {
        for (size_t j = 0; j < nj; j ++) {
            T cij = 0;
            for (size_t k = 0; k < nk; k ++) {
                cij += A[i*nk + k] * B[k*nj + j];
            }
            C[i*nj + j] = cij;
        }
    }
}
void ans_check_s32(int *mat, int *ans, int ni, int nj) {
    int flag = 1;
    for (int i = 0; i < ni*nj; i ++)
        if (mat[i] != ans[i]) {
            OS << "Ans Wrong at [" << i/nj << "," << i%nj << "]  mat:" << mat[i] << " ans:" << ans[i] << endl;
            flag = 0;
        }
    if (flag) OS << "Ans Correct" << endl;
}
void rand_mat_s32(int *mat, int length_1d, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < length_1d; i ++) 
        mat[i] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}
void rand_mat_f32(float *mat, int length_1d, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < length_1d; i ++) 
        mat[i] = ((float)rand() / ((float)RAND_MAX / (RAND_UB - RAND_LB))) + RAND_LB;
}
template <typename T>
void zeros(T *mat, int length_1d) {
    for (int i = 0; i < length_1d; i ++) mat[i] = 0;
}


static size_t Ti = 256;
static size_t Tj = 256;
static size_t Tk = 256;
#define MIN(x,y) (((x)<(y))?(x):(y))

void outer_kernel_packAB(
    int32_t *A, int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    mks32_pAB_t mk, packs32_A_t pkA, packs32_B_t pkB) 
{
    int32_t *Apack = (int32_t *) malloc(sizeof(int32_t) * Ti * Tk);
    int32_t *Bpack = (int32_t *) malloc(sizeof(int32_t) * Tk * Tj);

    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        for (size_t j0 = 0; j0 < nj; j0 += Tj) {
            size_t it = MIN(ni-i0, Ti);
            size_t jt = MIN(nj-j0, Tj);
            for (size_t k0 = 0; k0 < nk; k0 += Tk) {
                size_t kt = MIN(nk-k0, Tk);
                pkA(A+i0*nk+k0, Apack, it, kt, nk);
                pkB(B+k0*nj+j0, Bpack, kt, jt, nj);
                // print_mat(Apack, it*kt/32, 32, "Apack");
                // print_mat(Bpack, kt*jt/64, 64, "Bpack");
                mk(Apack, Bpack, C+i0*nj+j0, it, jt, kt, nj);
            }
        }
    }
}
void outer_kernel_packABC(
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
void outer_kernel(
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

#define TEST_N 2048

int main() {
    const int input_loop = 10;
    const int compute_loop = 10;
    // const int ni = 4, nj = 8, nk = 8;
    const int ni = TEST_N, nj = TEST_N, nk = TEST_N;


    #if OPTI_BLOCKING_MODE == false
    // 性能测试模式 ==============================
    OS << "Standard Test start" << endl;
    OS << "Loop: " << input_loop << "x" << compute_loop
       << ", Size: i" << ni << " j" << nj << " k" << nk << endl;
    #if FILE_OUTPUT == true
    cout << "File output: " << ouput_file << endl; 
    #endif

    double total_time2 = 0;
    for (int input = 0; input < input_loop; input ++) {
        int32_t *A = (int32_t *) malloc(sizeof(int32_t) * ni * nk);
        int32_t *B = (int32_t *) malloc(sizeof(int32_t) * nk * nj);
        int32_t *C = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
        int32_t *D = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
        // rand_mat_s32(A, ni * nk, 1234);
        // rand_mat_s32(B, nk * nj, 5678);
        rand_mat_s32(A, ni * nk, time(0));
        rand_mat_s32(B, nk * nj, time(0)+1);

        // 连续计算 ==============================
        double total_time1 = 0;
        for (int compute = 0; compute < compute_loop; compute ++) {
            zeros(C, ni * nj);
            auto start = Clock::now();

            // outer_kernel(A, B, C, ni, nj, nk, mks32_8x4k8_ldA_fchC);
            outer_kernel_packAB(A, B, C, ni, nj, nk, 
                mks32_4x8k8_ldB_fchC_pkAB, 
                packs32_4x8k8_A,
                packs32_4x8k8_B);
            // outer_kernel_packABC(A, B, C, ni, nj, nk, 
            //     mks32_4x8k8_ldB_fchC_pkABC, 
            //     packs32_4x8k8_A,
            //     packs32_4x8k8_B,
            //     packs32_4x8k8_C,
            //     unpacks32_4x8k8_C);

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000.0;
            total_time1 += dur; 
            OS << "compute time: " << dur << " msecs" << endl;
        }
        OS << "  avg time1: " << total_time1/compute_loop << " msecs for input: " << input << endl;
        total_time2 += total_time1/compute_loop;

        // 答案检查 ==============================
        #if ANS_CHECK == true  
        zeros(D, ni * nj);
        outer_kernel(A, B, D, ni, nj, nk, mks32_0);
        ans_check_s32(C, D, ni, nj);

        print_mat(A, ni, nk, "A");
        print_mat(B, nk, nj, "B");
        print_mat(C, ni, nj, "C");
        print_mat(D, ni, nj, "D");
        #endif

        free(A);
        free(B);
        free(C);
        free(D);
    }
    OS << "    total avg time2: " << total_time2/input_loop << " msecs" << endl;



    #else  // OPTI_BLOCKING_MODE
    // 寻找最优分块模式 ==============================
    const int blk_unit = 32;
    const int blk_lb = 64 / blk_unit;
    const int blk_ub = ni / blk_unit;
    const int rand_loop = 50;
    const int topN = 40;
    srand(time(0));

    OS << "Block Size Optimizing Test start" << endl;
    OS << "Loop: " << input_loop << "x" << rand_loop << "x" << compute_loop << endl;
    OS << ", Size: i" << ni << " j" << nj << " k" << nk << endl;
    #if FILE_OUTPUT == true
    cout << "File output: " << ouput_file << endl; 
    #endif

    struct block_t {
        int Ti, Tj, Tk;
        float time;
        bool operator<(block_t b) {
            return this.time < b.time;
        }
    };
    std::priority_queue<block_t> pq_block;
    for (int i = 0; i < topN; i ++) pq_block.push({-1,-1,-1,10000});

    float *record = (float *) malloc(sizeof(float) * 4 * input_loop * rand_loop);
    int idx_record = 0;

    for (int input = 0; input < input_loop; input ++) {
        int32_t *A = (int32_t *) malloc(sizeof(int32_t) * ni * nk);
        int32_t *B = (int32_t *) malloc(sizeof(int32_t) * nk * nj);
        int32_t *C = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
        rand_mat_s32(A, ni * nk, time(0));
        rand_mat_s32(B, nk * nj, time(0)+1);

        for (int rand = 0; rand < rand_loop; rand ++) {
            zeros(C, ni * nj);
            Ti = (rand() % (blk_ub-blk_lb) + blk_lb) * blk_unit;
            Tj = (rand() % (blk_ub-blk_lb) + blk_lb) * blk_unit;
            Tk = (rand() % (blk_ub-blk_lb) + blk_lb) * blk_unit;

            double total_time = 0.;
            for (int compute = 0; compute < compute_loop; compute ++) {
                auto start = Clock::now();

                // outer_kernel(A, B, C, ni, nj, nk, mks32_8x4k8_ldA_fchC);
                outer_kernel_packAB(A, B, C, ni, nj, nk, 
                    mks32_4x8k8_ldB_fchC_pkAB, 
                    packs32_4x8k8_A,
                    packs32_4x8k8_B);
                // outer_kernel_packABC(A, B, C, ni, nj, nk, 
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
            OS << Ti << " " << Tj << " " << Tk << " " << t << " msecs" << endl;

            if (t < pq_block.top().time) {
                pq_block.pop();
                pq_block.push({(float)Ti, (float)Tj, (float)Tk, t});
            }
        }

        free(A);
        free(B);
        free(C);
    }
    print_mat(record, input_loop * rand_loop, 4, "blocking record");
    free(record);
    #endif 



    // 收尾 ==============================
    char cur_time[50];
    time_t now = time(NULL);
    strftime(cur_time, 50, "%x %X", localtime(&now));
    OS << "Test Finished at " << cur_time << endl;

    #if FILE_OUTPUT == true
        ofs.close();
    #endif
}
