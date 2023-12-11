#include <Eigen/Eigen>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std;

const int SIZE = 1024;
const int FOR = 5;
const double ops = (double) SIZE * SIZE * SIZE * 2;

void int32() {
    MatrixXi A = MatrixXi::Random(SIZE, SIZE);
    MatrixXi B = MatrixXi::Random(SIZE, SIZE);
    MatrixXi C = MatrixXi::Zero(SIZE, SIZE);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < FOR; ++i) 
        C += A * B;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    double time = time_span.count() / FOR;
    cout << "Eigen int32: " << time << "s " 
        << ops / time / 1e9 << " GOPS" << endl;

    // Optional: Print result to prevent compiler optimization
    // cout << C(0, 0) << endl;
}

void fp32() {
    MatrixXf A = MatrixXf::Random(SIZE, SIZE);
    MatrixXf B = MatrixXf::Random(SIZE, SIZE);
    MatrixXf C = MatrixXf::Zero(SIZE, SIZE);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < FOR; ++i) 
        C += A * B;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    double time = time_span.count() / FOR;
    cout << "Eigen fp32: " << time << "s " 
        << ops / time / 1e9 << " GFLOPS" << endl;

    // Optional: Print result to prevent compiler optimization
    // cout << C(0, 0) << endl;
}

void fp64() {
    MatrixXd A = MatrixXd::Random(SIZE, SIZE);
    MatrixXd B = MatrixXd::Random(SIZE, SIZE);
    MatrixXd C = MatrixXd::Zero(SIZE, SIZE);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < FOR; ++i) 
        C += A * B;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    double time = time_span.count() / FOR;
    cout << "Eigen fp64: " << time << "s " 
        << ops / time / 1e9 << " GFLOPS" << endl;


    // Optional: Print result to prevent compiler optimization
    // cout << C(0, 0) << endl;
}

int main() {
    int32();
    fp32();
    fp64();
}
