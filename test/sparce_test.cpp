#include <iostream> 
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "test.h"
#include "DCmult.h"
#include "Cmult.h"

using std::cout;
using std::endl;
using std::string;

typedef std::chrono::high_resolution_clock Clock;
#define Dur(start,end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// 参数 1 =======================================
string ouput_file = "output/output.dat";
#define FILE_OUTPUT false  // 是否输出从 stdout 转到 文件
#define ANS_CHECK true  // 是否进行答案检查

#define RAND_UB 100  // 随机整数属于 [LB, UB)
#define RAND_LB -100


#if FILE_OUTPUT == true
    #define OS ofs
    std::ofstream ofs(ouput_file, std::ios::app);
#else
    #define OS cout
#endif

template <typename T>
void print_mat(T *mat, int ni, int nj, string msg="") {
    std::ofstream fout("output/sparce_matrix.dat", std::ios::app);
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
void ans_check_s32(int *mat, int *ans, int ni, int nj) {
    int flag = 1;
    for (int i = 0; i < ni*nj; i ++)
        if (mat[i] != ans[i]) {
            OS << "Ans Wrong at [" << i/nj << "," << i%nj << "]  mat:" << mat[i] << " ans:" << ans[i] << endl;
            flag = 0;
        }
    if (flag) OS << "Ans Correct" << endl;
}
void zeros_mat_s32(int *mat, int length_1d) {
    for (int i = 0; i < length_1d; i ++) mat[i] = 0;
}
void rand_sparce_mat_s32(int *mat, int ni, int nj, double density, unsigned int seed) {
    srand(seed);
    zeros_mat_s32(mat, ni*nj);
    int nnz = ni * nj * density;
    for (int elem = 0; elem < nnz; elem ++) {
        int i = rand() % ni;
        int j = rand() % nj;
        mat[i*nj + j] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
    }
}

int main() {
    // 参数 2 =======================================
    const int data_loop = 10;     // 生成矩阵数据的次数
    const int compute_loop = 10;  // 每个生成矩阵的计算次数
    const int ni = 4, nj = 5, nk = 3;

    OS << "Test start" << endl;
    OS << "Loop: " << data_loop << "x" << compute_loop << endl;
    OS << "Size: i" << ni << " j" << nj << " k" << nk << endl;
    #if FILE_OUTPUT == true
    cout << "File output: " << ouput_file << endl; 
    #endif

    double total_time2 = 0;
    for (int data = 0; data < data_loop; data ++) {
        int *A = (int *) malloc(sizeof(int) * ni * nk);
        int *B = (int *) malloc(sizeof(int) * ni * nk);
        int *C = (int *) malloc(sizeof(int) * ni * nk);
        int *D = (int *) malloc(sizeof(int) * ni * nk);
        rand_sparce_mat_s32(A, ni, nk, 0.2, 1234);
        rand_sparce_mat_s32(B, nk, nj, 0.2, 5678);

        #if ANS_CHECK == true
        zeros_mat_s32(D, ni*nj);
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


    char cur_time[50];
    time_t now = time(NULL);
    strftime(cur_time, 50, "%x %X", localtime(&now));
    OS << "Test Finished at " << cur_time << endl;
}