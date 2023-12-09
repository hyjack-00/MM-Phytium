#include <iostream>
#include <armpl.h>

int main() {
    // 设置矩阵大小
    const int M = 3;
    const int N = 3;
    const int K = 3;

    // 分配矩阵
    double* A = new double[M * K];
    double* B = new double[K * N];
    double* C = new double[M * N];

    // 初始化矩阵数据（示例数据）
    for (int i = 0; i < M * K; ++i) {
        A[i] = i + 1;
    }

    for (int i = 0; i < K * N; ++i) {
        B[i] = i + 1;
    }

    // 执行 GEMM
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);

    // 输出结果
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放内存
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
