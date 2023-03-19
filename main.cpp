#include<iostream> 
#include<vector>
using namespace std;

#include "mult.h"
#define M 4
#define N 5
#define L 3
//M| N- * N| L- 

int main() {
	/*
	1, 0, 0, 0, 0;
	0, 0, 0, 0, 0;
	0, 2, 0, 0, 0;
	4, 0, 0, 8, 0;
	*/
	int a[M * N] = { 1, 0, 0, 0, 0,	0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 8, 0 };

	/*
	1, 2, 0;
	0, 0, 0;
	4, 0, 0;
	0, 8, 0;
	0, 0, 0;
	*/
	int b[N * L] = { 1, 2, 0, 0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0 };

	/*
	1, 0, 2;
	0, 0, 0;
	4, 0, 0;
	0, 0, 8;
	*/
	int c[M * L] = { 1,0,2,0,0,0,4,0,0,0,0,8 };
	//init_random(a, ROW, COL);//4ÐÐ5ÁÐ
	sparce_matrix<int> sa1(a, M, N, 0);
	sparce_matrix<int> sa2(b, N, L, 1);
	sparce_matrix<int> sa3(c, M, L, 1);
	dc_sparce_matrix<int> d1(sa1);
	dc_sparce_matrix<int> d2(sa2);
	dc_sparce_matrix<int> d3(sa3);
	dc_sparce_matrix<int> da1(d1, 1);
	dc_sparce_matrix<int> da2(d2, 1);
	dc_sparce_matrix<int>* da3 = new dc_sparce_matrix<int>(d3, 0);
	gemm(&da1, &da2, da3);
	cout << da1 << da2 << *da3;

}