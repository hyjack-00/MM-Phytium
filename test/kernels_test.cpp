#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <arm_neon.h>
#include <omp.h>

#include "microkernerls.h"

using std::string;
using std::cout;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;
#define Dur(start,end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// #define FILE_OUTPUT true  // 输出从 stdout 到 文件
#define FILE_OUTPUT false
string ouput_file = "output/output.dat";
#if FILE_OUTPUT == true
    #define OS ofs
    std::ofstream ofs;
#else
    #define OS cout
#endif

// 输入矩阵的随机数范围
#define RAND_UB 100  // [LB, UB)
#define RAND_LB -100

template <typename T>
void print_mat(T *mat, int ni, int nj) {
    std::ofstream fout;
    fout.open("output/matrix.dat");
    if (!fout.is_open()) {
        cout << "Error opening file for result" << endl;
        exit(1);
    }
    for (int i = 0; i < ni; i ++) {
        for (int j = 0; j < nj; j ++) {
            fout << mat[i*nj + j] << " ";
        }
        fout << endl;
    }
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



int main() {
    const int input_loop = 10;
    const int compute_loop = 10;
    const int ni = 1024, nj = 1024, nk = 1024;

    int32_t *A = (int32_t *) malloc(sizeof(int32_t) * ni * nk);
    int32_t *B = (int32_t *) malloc(sizeof(int32_t) * nk * nj);
    int32_t *C = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
    // float32_t *A = (float32_t *) malloc(sizeof(float32_t) * ni * nk);
    // float32_t *B = (float32_t *) malloc(sizeof(float32_t) * nk * nj);
    // float32_t *C = (float32_t *) malloc(sizeof(float32_t) * ni * nj);

    double total_time2 = 0;
    for (int input = 0; input < input_loop; input ++) {
        rand_mat_s32(A, ni * nk, 1234);
        rand_mat_s32(B, nk * nj, 5678);
        // rand_mat_f32(A, ni * nk, 1234);
        // rand_mat_f32(B, nk * nj, 5678);

        double total_time1 = 0;
        for (int compute = 0; compute < compute_loop; compute ++) {
            rand_mat_s32(C, ni * nj, 1357);
            auto start = Clock::now();

            hello();

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000000.0;
            total_time1 += dur; 
            OS << "compute time: " << dur << " secs" << endl;
        }
        OS << "  avg time1: " << total_time1/compute_loop << " secs (for this input)" << endl;
        total_time2 += total_time1;
    }
    OS << "    total avg time2: " << total_time2/input_loop << " secs" << endl;

    // print_mat(C, ni, nj);
    free(A);
    free(B);
    free(C);

    char cur_time[50];
    time_t now = time(NULL);
    strftime(cur_time, 50, "%x %X", localtime(&now));
    OS << "Finished at " << cur_time << endl;
}
