#pragma once
#ifndef TRANSFER
#define TRANSFER

#include <vector>
#include <iostream>
#include <random>
using namespace std;

typedef unsigned int uint;
typedef const unsigned int cui;

//注意：rd()是一个无符号整数
template<typename tp>
void init_random(tp* a, cui row, cui col) {
	random_device rd;
	uint size = row * col;
	uint zerorow = rd() % row;
	uint zerocol = rd() % col;
	//初始化两个二维数组
	for (uint i = 0; i < row; ++i) {
		for (uint j = 0; j < col; j++) {
			a[i * col + j] = (((i - zerorow) * (j - zerocol)) ? (tp)rd() : 0);
		}
	}
	uint zero = 0;
	for (uint i = 0; i < size; i += 2) {
		zero = rd() % size;
		a[zero] = 0;
	}
}

template<typename tp>
ostream& operator<<(ostream& output, vector<tp>* a)
{
	output << "[";
	for (size_t i = 0; i < a->size(); i++) {
		output << (*a)[i] << ", ";
	}
	output << "]";
	return output;
}

template<typename tp>
struct sparce_matrix {
	vector<tp>* data;
	vector<unsigned int>* row_index;//同一列里的行号
	vector<unsigned int>* col_range;//哪些是同一列的
	bool trans; // 如果是0，就是csc，否则是csr（相当于存储转置的csc）

	// init from a dense matrix
	sparce_matrix(const tp* A, cui row, cui col, bool tr): trans(tr) {
		if (trans) {
			data = new vector<tp>;
			row_index = new vector<uint>;
			col_range = new vector<uint>;
			for (uint j = 0; j < row; j++) {
				col_range->push_back(row_index->size());
				for (uint i = 0; i < col; i++) {
					if (A[j * col + i]) {
						row_index->push_back(i);
						data->push_back(A[j * col + i]);
					}
				}
			}
			col_range->push_back(row_index->size());
			data->shrink_to_fit();
			row_index->shrink_to_fit();
			col_range->shrink_to_fit();
			return;
		}
		else {
			data = new vector<tp>;
			row_index = new vector<uint>;
			col_range = new vector<uint>;
			for (uint i = 0; i < col; i++) {
				col_range->push_back(row_index->size());
				for (uint j = 0; j < row; j++) {
					if (A[j * col + i]) {
						row_index->push_back(j);
						data->push_back(A[j * col + i]);
					}
				}
			}
			data->shrink_to_fit();
			row_index->shrink_to_fit();
			col_range->shrink_to_fit();
			return;
		}
	}

	~sparce_matrix() {}

	friend ostream& operator<<(ostream& output, const sparce_matrix& a)
	{
		output << "data:" << a.data << endl << "row_index:" << a.row_index << endl << "col_range:" << a.col_range << endl;
		return output;
	}

};

template<typename tp>
struct dc_sparce_matrix {
	vector<tp>* data;
	vector<uint>* row_index;//同一列里的行号
	vector<uint>* col_range;//行号里哪些是同一列的
	vector<uint>* col_index;//上面的范围是哪些列的
	bool trans; // 如果是0，就是dcsc，否则是dcsr（相当于存储转置的csc）

	dc_sparce_matrix() {
		trans = 0;
		data = new vector<tp>;
		row_index = new vector<uint>;
		col_range = new vector<uint>;
		col_index = new vector<uint>;
	}

	// init from a dense matrix
	dc_sparce_matrix(const sparce_matrix<tp>& a) {
		data = new vector<tp>(a.data->begin(), a.data->end());
		row_index = new vector<uint>(a.row_index->begin(), a.row_index->end());
		col_range = new vector<uint>;
		col_index = new vector<uint>;
		trans = a.trans;
		col_range->push_back(0);
		for (uint i = 0; i < a.col_range->size() - 1; i++) {
			if ((*(a.col_range))[i] != (*(a.col_range))[i + 1]) {
				col_range->push_back((*(a.col_range))[i + 1]);
				col_index->push_back(i);
			}
		}
	}

	void transpose() {

	}

	unsigned int nnz() {
		return data->size();
	}

	unsigned int nzc() {
		return col_index->size();
	}

	~dc_sparce_matrix() {}

	friend ostream& operator<<(ostream& output, const dc_sparce_matrix& a)
	{
		output << "transposed?  " << a.trans << endl << "data:" << a.data << endl << "row_index:" << a.row_index << endl << "col_range:" << a.col_range << endl << "col_index:" << a.col_index << endl;
		return output;
	}

};


#endif // ! TRANSFER

