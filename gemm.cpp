#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

using namespace std;
typedef chrono::high_resolution_clock Clock;

#define A(i,j) A[(i) + (j) * LDA]
#define B(i,j) B[(i) + (j) * LDB]
#define C(i,j) C[(i) + (j) * LDC]

static void init_array(int ni, int nj, int nk,
                       double *alpha, double *beta,
                       double *C, int LDC,
                       double *A, int LDA,
                       double *B, int LDB) {
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C(i,j) = (double)((i * j + 1) % ni) / ni;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A(i,j) = (double)(i * (j + 1) % nk) / nk;

  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B(i,j) = (double)(i * (j + 2) % nj) / nj;
}

static void print_array(int ni, int nj, double *C, int LDC) {

    int i, j;
    ofstream fout;
    fout.open("result.dat");
    if (!fout.is_open()) {
      cout << "Error opening file for result" << endl;
      exit(1);
    }
    for (i = 0; i < ni; i++) {
      for (j = 0; j < nj; j++) {
        fout << C(i,j) << " ";
      }
      fout << endl;
    }

}


void scale_c(double *C, int M, int N, int LDC, double scalar)
{
    int i,j;

    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){

            C(i, j) *= scalar;
        }
    }
}

void mydgemm_cpu_opt(int M, int N, int K,
                     double alpha, double *A, int LDA, double *B, int LDB,
                     double beta, double *C, int LDC)
{
    int i,j,k;

    if (beta != 1.0) scale_c(C, M, N, LDC, beta);

    for (i = 0; i < M; i++){

        for (j = 0; j < N; j++){

            double tmp = C(i,j);

            for (k = 0; k < K; k++){
                tmp += alpha * A(i,k) * B(k,j);
            }

            C(i,j) = tmp;
        }
    }
}

void mydgemm_cpu(int M, int N, int K,
                 double alpha, double *A, int LDA, double *B, int LDB,
                 double beta, double *C, int LDC)
{

    int i,j,k;

    if (beta != 1.0) scale_c(C, M, N, LDC, beta);

    int M4 = M & -4, N4 = N & -4;

    for (i = 0; i < M4; i += 4)
    {
        for (j = 0; j < N4; j += 4)
        {
            double c00 = C(i  ,j  );
            double c01 = C(i  ,j+1);
            double c02 = C(i  ,j+2);
            double c03 = C(i  ,j+3);

            double c10 = C(i+1,j  );
            double c11 = C(i+1,j+1);
            double c12 = C(i+1,j+2);
            double c13 = C(i+1,j+3);

            double c20 = C(i+2,j  );
            double c21 = C(i+2,j+1);
            double c22 = C(i+2,j+2);
            double c23 = C(i+2,j+3);

            double c30 = C(i+3,j  );
            double c31 = C(i+3,j+1);
            double c32 = C(i+3,j+2);
            double c33 = C(i+3,j+3);

            for (k = 0; k < K; k++)
            {
                double a0 = alpha * A(i  , k);
                double a1 = alpha * A(i+1, k);
                double a2 = alpha * A(i+2, k);
                double a3 = alpha * A(i+3, k);

                double b0 = B(k, j  );
                double b1 = B(k, j+1);
                double b2 = B(k, j+2);
                double b3 = B(k, j+3);

                c00 += a0 * b0;
                c01 += a0 * b1;
                c02 += a0 * b2;
                c03 += a0 * b3;

                c10 += a1 * b0;
                c11 += a1 * b1;
                c12 += a1 * b2;
                c13 += a1 * b3;

                c20 += a2 * b0;
                c21 += a2 * b1;
                c22 += a2 * b2;
                c23 += a2 * b3;

                c30 += a3 * b0;
                c31 += a3 * b1;
                c32 += a3 * b2;
                c33 += a3 * b3;
            }

            C(i  ,j  ) = c00;
            C(i  ,j+1) = c01;
            C(i  ,j+2) = c02;
            C(i  ,j+3) = c03;

            C(i+1,j  ) = c10;
            C(i+1,j+1) = c11;
            C(i+1,j+2) = c12;
            C(i+1,j+3) = c13;

            C(i+2,j  ) = c20;
            C(i+2,j+1) = c21;
            C(i+2,j+2) = c22;
            C(i+2,j+3) = c23;

            C(i+3,j  ) = c30;
            C(i+3,j+1) = c31;
            C(i+3,j+2) = c32;
            C(i+3,j+3) = c33;

        }
    }

    if (M4==M && N4==N){
        return;
    }

    // boundary conditions
    if (M4 != M){
        mydgemm_cpu_opt(M - M4, N, K,
                        alpha, A + M4, LDA, B, LDB,
                        1.0, &C(M4,0), LDC);
    }

    if (N4 != N){
        mydgemm_cpu_opt(M4, N - N4, K,
                        alpha, A, LDA, &B(0,N4), LDB,
                        1.0, &C(0,N4), LDC);
    }
}


int main(int argc, char **argv)
{
    int ni = 1000;
    int nj = 1300;
    int nk = 1600;

    double alpha;
    double beta;

    double *A = (double *) malloc(sizeof(double *) * ni * nk);

    double *B = (double *)malloc (sizeof(double *) * nk * nj);

    double *C = (double *)malloc (sizeof(double *) * ni * nj);

    init_array(ni, nj, nk, &alpha, &beta, C, ni, A, ni, B, nk);


    cout << "Start computing..." << endl;

    auto startTime = Clock::now();

    mydgemm_cpu(ni, nj, nk,
                alpha, A, ni, B, nk,
                beta, C, ni);

    auto endTime = Clock::now();

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

    cout << "End computing..." << endl;
    cout << "Compute time=  " << compTime.count() << " microseconds" << endl;

    print_array(ni, nj, C, ni);

    free(A);
    free(B);
    free(C);

    return 0;
}