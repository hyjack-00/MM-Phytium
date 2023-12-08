#include <iostream> 
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "test_helpers.h"
#include "DCmult.h" 
#include "Cmult.h"

#define ANS_CHECK false  // 是否进行答案检查

int main() {
    // 参数 2 =======================================
    const int data_loop = 1;     // 生成新矩阵数据的次数
    const int compute_loop = 1;  // 每个生成矩阵的计算次数
    // const int ni = 4, nj = 5, nk = 3;
    const int ni = 400, nj = 300, nk = 500;

    cout << "Test start" << endl;
    cout << "Loop: " << data_loop << "x" << compute_loop << endl;
    cout << "Size: i" << ni << " j" << nj << " k" << nk << endl;
    #if FILE_OUTPUT == true
    cout << "File output: " << ouput_file << endl; 
    #endif

    double total_time2 = 0;
    for (int data = 0; data < data_loop; data ++) {
        int *A = (int *) malloc(sizeof(int) * ni * nk);
        int *B = (int *) malloc(sizeof(int) * nk * nj);
        int *C = (int *) malloc(sizeof(int) * ni * nj);
        rand_sparce_array_s32(A, ni, nk, 0.2, 1234);
        rand_sparce_array_s32(B, nk, nj, 0.2, 5678);
        zeros_s32(C, ni*nj);
        // rand_sparce_array_s32(A, ni, nk, 0.2, time(0));
        // rand_sparce_array_s32(B, nk, nj, 0.2, time(0)+1);
        // rand_sparce_array_s32(C, ni, nj, 0.2, time(0)+2);

        print_mat(A, ni, nk, "A");
        print_mat(B, nk, nj, "B");

        sparce_matrix<int> A_sp(A, ni, nk, 0);
        sparce_matrix<int> B_sp(B, nk, nj, 1);
        dc_sparce_matrix<int> A_dcsp0(A_sp);
        dc_sparce_matrix<int> B_dcsp0(B_sp);
        dc_sparce_matrix<int> A_dcsp(A_dcsp0, 1);
        dc_sparce_matrix<int> B_dcsp(B_dcsp0, 1);

        // C 暂时采用堆内存（不合理），见 transfer.h @55
        sparce_matrix<int> *C_sp;
        dc_sparce_matrix<int> *C_dcsp;

        // 连续计算 ==============================
        double total_time1 = 0;
        for (int compute = 0; compute < compute_loop; compute ++) {
            C_sp = new sparce_matrix<int>(C, ni, nj, 1);
            dc_sparce_matrix<int> C_dcsp0(*C_sp);
            C_dcsp = new dc_sparce_matrix<int>(C_dcsp0, 0);
            auto start = Clock::now();

            dcgemm(&A_dcsp, &B_dcsp, C_dcsp);

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
            int *D = (int *) malloc(sizeof(int) * ni * nj);
            copy_mat(C, D, ni*nj);
            naive(A, B, C, ni, nj, nk);

            // 未实现对比：dcsp 转 array / dcsp 实现 == 
            // ans_check_s32(C, D, ni, nj);

            print_mat(A, ni, nk, "A");
            print_mat(B, nk, nj, "B");
            print_mat(C_dcsp, "C");
            print_mat(D, ni, nj, "D");
            free(D);
        #endif 

        free(A);
        free(B);
        free(C);
    }
    cout << "    total avg time2: " << total_time2/data_loop << " msecs" << endl;

    char cur_time[50];
    time_t now = time(NULL);
    strftime(cur_time, 50, "%x %X", localtime(&now));
    cout << "Test Finished at " << cur_time << endl;
}