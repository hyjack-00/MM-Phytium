#pragma once
#ifndef TRANSFER
#define TRANSFER

#include <vector>
#include <iostream>
#include <random>
#include <map>
#include <unordered_map>
using namespace std;

typedef unsigned int uint;
typedef const unsigned int cui;



template<typename tp>
vector<tp>* init_with_reserve(uint size) {
	vector<tp>* a = new vector<tp>;
	a->reserve(size);
	return a;
}

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

/* 关于内存管理
	1. 矩阵中的数组数据是堆数据，但矩阵对象本身是无论以栈 / 堆存储的
		- dcgemm 函数为例，如果 C 是栈上对象会错误
		  传入的 C 指针会被改变为指向新的堆内存，但这个指针只是表示栈位置的临时值
		- 应当：
			- 实现好矩阵对象的赋值运算符，做好旧内存的释放
			- 矩阵对象应全部以 直接引用 & 传入
			- dcgemm 内部的中间矩阵对象全部使用局部栈上内存，运算结果赋值给 C 的引用

	2. 大部分情况下，对象本身的内存由用户方处理，输入的内存不应当由库函数释放
	  	- 通过其他类型的输入矩阵的 “转换构造函数”，const 输入矩阵 √
		- 需要 “赋值运算符”，const 输入矩阵 !
		- 可能需要 “类型转换运算符”，const 输入矩阵 !
		- 可能需要 类型转移转换函数（友元），需要销毁输入矩阵 !
		- 销毁矩阵 .clear() 应当是清空对象指向的堆数据，但对象本身的内存不能被清除
	
	3. gemm 函数究竟是使用哪种形式？
		- `C1 = gemm(const A, const B, const C); `
		- `gemm(const A, const B, C); `  √ 只要处理好 C 的内存

	-- Huangyj 
*/

template<typename tp>
struct sparce_matrix {
	/* 使用 const 值会导致赋值构造默认被禁止 -- Huangyj */
	uint row;//只要大于row_index里的所有值就可以了
	vector<tp>* data;
	vector<uint>* row_index;//同一列里的行号
	vector<uint>* col_range;//哪些是同一列的
	bool trans; // 如果是0，就是csc，否则是csr（相当于存储转置的csc）

	sparce_matrix() {
		row = 0;
		data = nullptr;
		row_index = nullptr;
		col_range = nullptr;
		trans = 0;
	}

	//一个空的矩阵
	sparce_matrix(cui row_num, const bool& transport):row(row_num),trans(transport) {
		data = new vector<tp>;
		row_index = new vector<uint>;
		col_range = new vector<uint>(1,0);
	}

	// init from a dense matrix
	sparce_matrix(const tp* A, cui r, cui c, bool tr) : trans(tr), row(tr ? c : r) {
		if (trans) {
			data = new vector<tp>;
			row_index = new vector<uint>;
			col_range = new vector<uint>;
			for (uint j = 0; j < r; j++) {
				col_range->push_back(row_index->size());
				for (uint i = 0; i < c; i++) {
					if (A[j * c + i]) {
						row_index->push_back(i);
						data->push_back(A[j * c + i]);
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
			for (uint i = 0; i < c; i++) {
				col_range->push_back(row_index->size());
				for (uint j = 0; j < r; j++) {
					if (A[j * c + i]) {
						row_index->push_back(j);
						data->push_back(A[j * c + i]);
					}
				}
			}
			col_range->push_back(row_index->size());
			data->shrink_to_fit();
			row_index->shrink_to_fit();
			col_range->shrink_to_fit();
			return;
		}
	}

	//注：trans==1意味着它会以另一种存储方式存储。但是两个矩阵逻辑上是相等的。
	sparce_matrix(const sparce_matrix<tp>* a,const bool& transpose) :row(transpose ? a->col() : a->row) {
		if (transpose) {
			data = new vector<tp>(a->nnz(), 0);
			trans = !a->trans;
			row_index = new vector<uint>(a->nnz(), 0);

			//创建一些临时数组:col_num代表A转置的这一列上有多少元素
			vector<uint>* col_num = new vector<uint>(a->row, 0);
			for (uint ri = 0; ri < a->nnz(); ri++) {
				col_num->at(a->row_index->at(ri)) += 1;
			}
			uint* col_now = new uint[a->row + 1];//这个是为了等会记录每个data插入的位置。
			uint tmp = 0;
			for (uint it = 0; it != a->row; it++) {
				col_now[it] = tmp;
				tmp += col_num->at(it);
			}
			col_now[a->row] = tmp;
			col_range = new vector<uint>(col_now, col_now + a->row + 1);


			//插入data
			for (uint col = 0; col < a->col(); col++) {
				for (uint row = a->col_range->at(col); row < a->col_range->at(col + 1); row++) {
					data->at(col_now[a->row_index->at(row)]) = a->data->at(row);
					row_index->at(col_now[a->row_index->at(row)]) = col;
					col_now[a->row_index->at(row)]++;
				}
			}
		}
		else {
			data = new vector<tp>(a->data->begin(), a->data->end());
			trans = a->trans;
			row_index = new vector<uint>(a->row_index->begin(), a->row_index->end());
			col_range = new vector<uint>(a->col_range->begin(), a->col_range->end());
		}
	}

	//注意：默认行号递增，且直接填充进下一列。但不要求非空。
	template<class InputIterator>
	void append_col(InputIterator begin, InputIterator end) {
		while (begin != end) {
			row_index->push_back((*begin)->row);
			data->push_back((*begin)->data);
			begin++;
		}
		col_range->push_back(data->size());
	}

	uint nnz() const {
		return data->size();
	}

	void transpose() const {
		!trans;
		return;
	}

	void clean() {
		delete data;
		delete row_index;
		delete col_range;
		data = nullptr;
		row_index = nullptr;
		col_range = nullptr;
	}

	~sparce_matrix() {
		delete data;
		delete row_index;
		delete col_range;
	}

	uint col() const {
		return col_range->size() - 1;
	}

	friend ostream& operator<<(ostream& output, const sparce_matrix& a)
	{
		output << "transposed? " << a.trans << endl << "data:" << a.data << endl << "row_index : " << a.row_index << endl << "col_range : " << a.col_range << endl;
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
		data = nullptr;
		row_index = nullptr;
		col_range = nullptr;
		col_index = nullptr;
	}

	// init from a normal dense matrix
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
		data->shrink_to_fit();
		row_index->shrink_to_fit();
		col_range->shrink_to_fit();
		col_index->shrink_to_fit();
	}

	// init from a double conpressed matrix. 
	// notice: "transpose == 1" means you want to store this matrix in a different form, but the two are logically equal.
	dc_sparce_matrix(const dc_sparce_matrix<tp>& a, const bool& transpose = false) {
		if (transpose) {
			data = new vector<tp>(a.nnz(), 0);
			trans = !a.trans;
			row_index = new vector<uint>(a.nnz(), 0);

			//创建一些临时数组
			map<uint, uint>* num = new map<uint, uint>;
			for (uint ri = 0; ri < a.nnz(); ri++) {
				if (num->find(a.row_index->at(ri)) == num->end()) {
					num->insert(pair<uint,uint>(a.row_index->at(ri), 1));
				}
				else {
					num->at(a.row_index->at(ri)) += 1;
				}
			}
			col_index = init_with_reserve<uint>(num->size());
			col_range = init_with_reserve<uint>(num->size() + 1);
			unordered_map<uint, uint>* col_now = new unordered_map<uint, uint>;//这个是为了等会记录每个data插入的位置。
			col_range->push_back(0);
			for (auto it = num->begin(); it != num->end(); it++) {
				col_now->insert(pair<uint, uint>(it->first, col_range->back()));
				col_index->push_back(it->first);
				col_range->push_back(col_range->back() + it->second);
				
			}
			
			//插入data
			for (uint col = 0; col < a.col_index->size(); col++) {
				for (uint row = a.col_range->at(col); row < a.col_range->at(col + 1); row++) {
					data->at(col_now->at(a.row_index->at(row))) = a.data->at(row);
					row_index->at(col_now->at(a.row_index->at(row))) = a.col_index->at(col);
					col_now->at(a.row_index->at(row))++;
				}
			}
		}
		else {
			data = new vector<tp>(a.data->begin(), a.data->end());
			trans = a.trans;
			row_index = new vector<uint>(a.row_index->begin(), a.row_index->end());
			col_range = new vector<uint>(a.col_range->begin(), a.col_range->end());
			col_index = new vector<uint>(a.col_index->begin(), a.col_index->end());
		}
	}

	//通过输入的符合字典序的坐标三元组构建。
	template<class InputIterator>
	dc_sparce_matrix(InputIterator begin, InputIterator end, const bool& transpose = false) {
		trans = transpose;
		if (trans) {
			data = new vector<tp>;
			row_index = new vector<uint>;
			col_range = new vector<uint>;
			col_index = new vector<uint>;
			uint index = (*begin)->row;
			col_range->push_back(0);
			col_index->push_back(index);
			for (auto i = begin; i != end; i++) {
				if ((*i)->row != index) {
					index = (*i)->row;
					col_index->push_back((*i)->row);
					col_range->push_back(data->size());
				}
				row_index->push_back((*i)->col);
				data->push_back((*i)->data);
			}
			col_range->push_back(data->size());
		}
		else {
			data = new vector<tp>;
			row_index = new vector<uint>;
			col_range = new vector<uint>;
			col_index = new vector<uint>;
			uint index = (*begin)->col;
			col_range->push_back(0);
			col_index->push_back(index);
			for (auto i = begin; i != end; i++) {
				if ((*i)->col != index) {
					index = (*i)->col;
					col_index->push_back((*i)->col);
					col_range->push_back(data->size());
				}
				row_index->push_back((*i)->row);
				data->push_back((*i)->data);
			}
			col_range->push_back(data->size());
		}
		data->shrink_to_fit();
		row_index->shrink_to_fit();
		col_range->shrink_to_fit();
		col_index->shrink_to_fit();
	}

	uint nnz() const {
		return data->size();
	}

	uint nzc() const {
		return col_index->size();
	}

	void transpose() const {
		!trans;
		return;
	}

	void clean() {
		delete data;
		delete row_index;
		delete col_index;
		delete col_range;
		data = nullptr;
		row_index = nullptr;
		col_range = nullptr;
		col_index = nullptr;
	}

	~dc_sparce_matrix() {
		delete data;
		delete row_index;
		delete col_index;
		delete col_range;
	}

	friend ostream& operator<<(ostream& output, const dc_sparce_matrix& a)
	{
		output << "transposed?  " << a.trans << endl << "data:" << a.data << endl << "row_index:" << a.row_index << endl << "col_range:" << a.col_range << endl << "col_index:" << a.col_index << endl;
		return output;
	}

};


#endif // ! TRANSFER

