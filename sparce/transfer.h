#pragma once
#ifndef TRANSFER
#define TRANSFER

#include <vector>
#include <iostream>
#include <random>
#include <map>
#include <unordered_map>

#include <string.h>  // memset

using std::vector;
using std::ostream;
using std::map;
using std::unordered_map;
using std::pair;

typedef unsigned int uint;


template<typename tp>
vector<tp>* init_with_reserve(uint size) {
	vector<tp>* a = new vector<tp>;
	a->reserve(size);
	return a;
}

//注意：rd()是一个无符号整数
template<typename tp>
void init_random(tp* a, const uint row, const uint col) {
	std::random_device rd;
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

// 实现 vector 的输出
template<typename tp>
ostream& operator<<(ostream& output, const vector<tp>& a) {
	output << "[";
	for (size_t i = 0; i < a->size(); i++) {
		output << a[i] << ", ";
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

// 默认 CSC，被压缩的 outer 是 col
#define outer_size col_size
#define inner_size row_size
#define outer_range col_range
#define inner_index row_index


template<typename tp>
struct sparce_matrix {
	uint row_size;            // 实际行数，只要大于 row_index 里的所有值就可以了
	uint col_size;            // === col_range.size() - 1
	vector<tp> data;          // data[i]      表示第 i 个元素的值
	vector<uint> row_index;   // row_index[i] 表示第 i 个元素的行号
	vector<uint> col_range;   // col_range[j] 表示 j 列的第一个元素的索引

	bool csr;                 // 默认 0 表示 csc 存储
	                          // 转置时可 csr=1：数据存储不变，对象变成另一个矩阵的 csr 表示，此时 col-row 含义相反

	sparce_matrix() : 
		row_size(0), col_size(0), csr(0) {}  // vector 自动初始化

	/** Empty matrix */
	sparce_matrix(uint rows, uint cols, bool is_csr=0) : csr(is_csr) {
		if (is_csr) {
			row_size = cols;
			col_size = rows;
		}
		else {
			row_size = rows;
			col_size = cols;
		}
		col_range = vector<uint>(1,0);
	}

	/** Init from a dense matrix (Row-Major)
	 * @param transpose 为 1 时表示稠密矩阵输入前，增加转置操作
	*/
	sparce_matrix(const tp* M, uint M_rows, uint M_cols, bool is_csr=0, bool transpose=0) {
		data = vector<tp>();
		inner_index = vector<uint>();
		outer_range = vector<uint>();
		csr = is_csr;

		bool M_col_compress = !(is_csr ^ transpose);
		/*
		    csr  trans  |  M_col_compress （是否选择 M_col 维度压缩输入为 outer）
			 0     0    |    1
			 0     1    |    0
			 1     1    |    1
			 1     0    |    0
		*/

		// outer=M_col -> inner=M_row
		inner_size = M_col_compress ? M_rows : M_cols;
		outer_size = M_col_compress ? M_cols : M_rows;

		if (M_col_compress) {
			for (uint c = 0; c < M_cols; c++) {
				outer_range.push_back(inner_index.size());
				for (uint r = 0; r < M_rows; r++) {
					if (M[r * M_cols + c]) {
						inner_index.push_back(c);
						data.push_back(M[r * M_cols + c]);
					}
				}
			}
			outer_range.push_back(inner_index.size());
			
			data.shrink_to_fit();
			inner_index.shrink_to_fit();
			outer_range.shrink_to_fit();
			return;
		}
		else {
			for (uint r = 0; r < M_rows; r++) {
				outer_range.push_back(inner_index.size());
				for (uint c = 0; c < M_cols; c++) {
					if (M[r * M_cols + c]) {
						inner_index.push_back(r);
						data.push_back(M[r * M_cols + c]);
					}
				}
			}
			outer_range.push_back(inner_index.size());

			data.shrink_to_fit();
			inner_index.shrink_to_fit();
			outer_range.shrink_to_fit();
			return;
		}
	}

	/** @param transpose 为 1 时将会改变存储数据，但不会改变数学表示 */
	sparce_matrix(const sparce_matrix<tp>& a, bool convert_fmt=0) {
		if (convert_fmt) {
			/* 以下编程假设 a 是 csc
				a.outer = "a.col", 
				a.inner = "a.row",
				b.outer = a.inner = "a.row", 
				b.inner = a.outer = "a.col"
			*/ 
			uint nnz = a.non_zeros();

			csr = !a.csr;
			row_size = a.col_size;
			col_size = a.row_size;
			row_index = vector<uint>(nnz, 0);
			data      = vector<tp>(nnz, 0);

			// outer/col_count[i] 统计 A 的第 i 行(inner) 元素数量
			vector<uint> col_count = vector<uint>(a.row_size, 0);
			for (uint ri = 0; ri < nnz; ri++) {
				uint r = a.row_index[ri];
				col_count[r] += 1;
			}

			vector<uint> col_now = vector<uint>(a.row_size + 1);  // 累加得到 outer/col_range
			uint acc = 0;
			for (uint it = 0; it < a.row_size; it++) {
				col_now[it] = acc;
				acc += col_count[it];
			}
			col_now[a.row_size] = acc;
			col_range = vector<uint>(col_now);

			// 插入data
			// (a)i 表示元素索引，(a)r 表示行号，(a)c 表示列号
			for (uint ac = 0; ac < a.col_size; ac++) {
				for (uint ar = a.col_range[ac]; ar < a.col_range[ac+1]; ar++) {
					uint ai = a.row_index[ar];
					uint i = col_now[ai];

					data[i] = a.data[ai];
					row_index[i] = ac;
					col_now[ai]++;
				}
			}
		}
		else {
			row_size = a.row_size;
			col_size = a.col_size;
			data 	  = vector<tp>(a.data.begin(), a.data.end());
			row_index = vector<uint>(a.row_index.begin(), a.row_index.end());
			col_range = vector<uint>(a.col_range.begin(), a.col_range.end());
			csr = a.csr;
		}
	}

	/** 仅作测试用
	 * @return Dense matrix (Row-Major) form of the matrix, memory independent
	 * @warning Raw pointer, should be deleted by user!!
	 * */
	tp * to_dense_mat() const {
		tp *mat = new tp[row_size * col_size];
		memset(mat, 0, sizeof(tp) * row_size * col_size);
		
		if (csr) {  // outer = row
			uint it = 0;
			for (uint r = 0; r < outer_size; r++) {
				for ( ; it < outer_range[r+1]; it++) {
					uint c = inner_index[it];
					mat[r * inner_size + c] = data[it];
				}
			}
		}
		else {  // outer = col
			uint it = 0;
			for (uint c = 0; c < outer_size; c++) {
				for ( ; it < outer_range[c+1]; it++) {
					uint r = inner_index[it];
					mat[r * inner_size + c] = data[it];
				}
			}
		}
	}

	// /** csc 时补充一列？ */
	// template<class InputIterator>
	// void append_outer(InputIterator begin, InputIterator end) {
	// 	while (begin != end) {
	// 		inner_index.push_back((*begin).row_size);
	// 		data.push_back((*begin).data);
	// 		begin++;
	// 	}
	// 	outer_range.push_back(data.size());
	// }

	/** @warning 将会改变数据组织方式，但不会改变数学表示 */
	void convert_format() {
		transpose();
		quick_transpose();
	}

	/** @warning 将会改变数据组织方式，但不改变 csr，这会变成另一个数学矩阵的表示 */
	void transpose() {
		/* 以下编程假设 a 是 csc
			a.outer = "a.col", 
			a.inner = "a.row",
			b.outer = a.inner = "a.row", 
			b.inner = a.outer = "a.col"
		*/ 
		uint nnz = non_zeros();

		std::swap(col_size, row_size);
		vector<uint> _row_index = std::move(row_index);
		vector<uint> _col_range = std::move(col_range);
		vector<uint> _data      = std::move(data);

		row_index = vector<uint>(nnz, 0);
		data      = vector<tp>(nnz, 0);

		// outer/col_count[i] 统计 A 的第 i 行(inner) 元素数量
		vector<uint> col_count = vector<uint>(col_size, 0);
		for (uint ri = 0; ri < nnz; ri++) {
			uint r = _row_index[ri];
			col_count[r] += 1;
		}

		vector<uint> col_now = vector<uint>(col_size + 1);  // 累加得到 outer/col_range
		uint acc = 0;
		for (uint it = 0; it < col_size; it++) {
			col_now[it] = acc;
			acc += col_count[it];
		}
		col_now[col_size] = acc;
		col_range = vector<uint>(col_now);

		// 插入data
		// (a)i 表示元素索引，(a)r 表示行号，(a)c 表示列号
		for (uint ac = 0; ac < row_size; ac++) {
			for (uint ar = _col_range[ac]; ar < _col_range[ac+1]; ar++) {
				uint ai = _row_index[ar];
				uint i = col_now[ai];

				data[i] = _data[ai];
				row_index[i] = ac;
				col_now[ai]++;
			}
		}
	}

	void quick_transpose() {
		csr = !csr;
		return;
	}

	void clear() {
		col_range.clear();
		row_index.clear();
		data.clear();
		col_size = 0;
		row_size = 0;
	}

	~sparce_matrix() = default;

	uint cols() const {
		if (csr) return inner_size;
		else     return outer_size;  // col_range->size() - 1;
	}
	uint rows() const {
		if (csr) return outer_size;  // col_range->size() - 1;
		else     return inner_size;
	}

	/** number of non-zero elements */
	uint non_zeros() const {
		return data.size();
	}

	friend ostream& operator<<(ostream& output, const sparce_matrix& a)
	{
		output << "csr? " << a.csr << endl << "data:" << a.data << endl << "row_index : " << a.row_index << endl << "col_range : " << a.col_range << endl;
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
		data = new vector<tp>(a.data.begin(), a.data.end());
		row_index = new vector<uint>(a.row_index.begin(), a.row_index.end());
		col_range = new vector<uint>;
		col_index = new vector<uint>;
		trans = a.csr;
		col_range->push_back(0);
		for (uint i = 0; i < a.col_range.size() - 1; i++) {
			if (a.col_range[i] != a.col_range[i + 1]) {
				col_range->push_back(a.col_range[i + 1]);
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

