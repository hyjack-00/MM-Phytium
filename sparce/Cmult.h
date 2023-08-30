#pragma once
#ifndef COMPRESSED_SPARSE_MULT
#define COMPRESSED_SPARSE_MULT
#define USE_ORIGINAL false

#include "transfer.h"
#include <iostream>
#include <cassert>
#include <queue>
#include <map>
#include <deque>

/* 关于之后的改进
	1. gemm 函数究竟是使用哪种形式？
		- `C1 = gemm(const A, const B, const C); `
		- `gemm(const A, const B, C); `  √ 只要处理好 C 的内存

	2. 指标和数分离计算
		- 指标在乘的过程中被生成，需要被排序。
		- 数据由乘法运算得到，根据指标情况可能需要相加。

	3. 创建数组C的过程(把多列合到一起)还无法并行

	-- yzr
*/

template<typename tp>
struct halfCord {
	uint row, sign;//sign是为了在merge的时候，提示这个元素来自哪次结果
	tp data;
	halfCord(cui row_index, const tp d, cui s) :row(row_index), data(d), sign(s) {}
};

template<typename tp>
struct cmp2 {
	bool reverse;
	cmp2(const bool& revparam = false) {
		reverse = revparam;
	}
	bool operator() (const halfCord<tp>* lhs, const halfCord<tp>* rhs) const {
		if (lhs->row > rhs->row) {
			return !reverse;
		}
		else {
			return reverse;
		}
	}
};



// 比较第一个元素的大小
// 注意: 每次待比较队列pop了以后, 也要让这个队列从优先队列里pop再重新加入
struct cmp_deque {
	bool reverse;
	cmp_deque(const bool& revparam = false) {
		reverse = revparam;
	}
	bool operator() (const deque<uint>* lhs, const deque<uint>* rhs) const {
		if (lhs->front() > rhs->front()) {
			return !reverse;
		}
		else {
			return reverse;
		}
	}
};

template<typename tp>
bool is_equal2(const halfCord<tp>* a, const halfCord<tp>* b) {
	return (a->row == b->row);
}

#if USE_ORIGINAL == true

// 原版的, 会改变C的指向
template<typename tp>
void cgemm(const sparce_matrix<tp>* _A, const sparce_matrix<tp>* _B, sparce_matrix<tp>*& _C) {
	const sparce_matrix<tp>* A, * B, * C;
	if (_A->trans) {
		A = new sparce_matrix<tp>(_A, 1);
	}
	else {
		A = _A;
	}
	if (_B->trans) {
		B = new sparce_matrix<tp>(_B, 1);
	}
	else {
		B = _B;
	}
	if (_C->trans) {
		C = new sparce_matrix<tp>(_C, 1);
	}
	else {
		C = _C;
	}
	assert(!A->trans && !B->trans && !C->trans);

	sparce_matrix<tp>* C1 = new sparce_matrix<tp>(A->row, 0);

	//相当于每次做一个矩阵乘B的一个列向量，最后直接拼起来
	for (uint i = 0; i < B->col(); i++) {
		vector<queue<halfCord<tp>*>*>* mid_results = new vector<queue<halfCord<tp>*>*>;

		// 先把C里的值输进去，最后加
		queue<halfCord<tp>*>* c_ele = new queue<halfCord<tp>*>;
		for (uint row = C->col_range->at(i); row < C->col_range->at(i + 1); row++) {
			halfCord<tp>* c = new halfCord<tp>(C->row_index->at(row), C->data->at(row), 0);
			c_ele->push(c);
		}
		if (!c_ele->empty())mid_results->push_back(c_ele);

		// 对B在这一列上的每一个非零元的行号，找到A中列号与这些行号相同的列，然后这些列被非零元数乘
		for (uint j = B->col_range->at(i); j < B->col_range->at(i + 1); j++) {
			queue<halfCord<tp>*>* jcol = new queue<halfCord<tp>*>;
			for (uint k = A->col_range->at(B->row_index->at(j)); k < A->col_range->at(B->row_index->at(j) + 1); k++) {
				halfCord<tp>* ab = new halfCord<tp>(A->row_index->at(k), A->data->at(k) * B->data->at(j), mid_results->size());
				jcol->push(ab);
			}
			if (!jcol->empty())mid_results->push_back(jcol);
		}

		// 排列
		priority_queue<halfCord<tp>*, vector<halfCord<tp>*>, cmp2<tp>> compare_result;
		deque<halfCord<tp>*> final_result;
		for (uint i = 0; i < mid_results->size(); i++) {
			compare_result.push(mid_results->at(i)->front());
			mid_results->at(i)->pop();
		}

		while (!compare_result.empty()) {
			halfCord<tp>* tmp = compare_result.top();
			compare_result.pop();
			if (!final_result.empty() && is_equal2(final_result.back(), tmp)) {
				final_result.back()->data += tmp->data;
			}
			else {
				final_result.push_back(tmp);
			}

			if (!mid_results->at(tmp->sign)->empty()) {
				compare_result.push(mid_results->at(tmp->sign)->front());
				mid_results->at(tmp->sign)->pop();
			}
		}
		C1->append_col(final_result.begin(), final_result.end());
	}

	_C = C1;
}


#else

// 新版。会改变_C的指向
template<typename tp>
void cgemm(const sparce_matrix<tp>* _A, const sparce_matrix<tp>* _B, sparce_matrix<tp>*& _C) {
	const sparce_matrix<tp>* A, * B, * C;
	if (_A->trans) {
		A = new sparce_matrix<tp>(_A, 1);
	}
	else {
		A = _A;
	}
	if (_B->trans) {
		B = new sparce_matrix<tp>(_B, 1);
	}
	else {
		B = _B;
	}
	if (_C->trans) {
		C = new sparce_matrix<tp>(_C, 1);
	}
	else {
		C = _C;
	}
	assert(!A->trans && !B->trans && !C->trans);

	sparce_matrix<tp>* C1 = new sparce_matrix<tp>(A->row, 0);

	//相当于每次做一个矩阵乘B的一个列向量，最后直接拼起来
	for (uint i = 0; i < B->col(); i++) {
		/*
		position phase
		第一步:	把要用到的位置坐标装到一起(尾部朝上)
		而且需要尾部留一个序号, 这个序号用来生成对数据(value)的操作
		比如, Bi列上, i_1,i_2,...行上是非零元, 那么posi的结果类似:
		(每列是一个posi的元素, 尾部朝上)
		(C_i---A_i1--A_i2--A_i3--...)
		 0     1     2     3     ...
		 5     4     2     7     ...
		 3     1     1     6     ...
		 0                 2     ...
						   1     ...

		1. 这一步只是为了增强局部性. 这一步相当于把k个队列装载到缓存里.
			可能有不需要复制的做法
		2. 序号放在尾部会导致局部性变差的问题, 但放在头部会导致对中间元素删除
			可能需要新建一个数据结构, 或者考虑一次装载一部分序号,
			并且改用uint16数据类型, 使得每次载入的长度适合缓存大小
			(注意在后面会用到-1)
		3. 尺寸过大的时候, 可能分几步合并, 在每一步考虑一次装载一部分,
			最终使得总长度符合缓存大小
			--yzr
		*/
		uint iindex = 0;
		priority_queue<deque<uint>*, vector<deque<uint>*>, cmp_deque>* posi = new priority_queue<deque<uint>*, vector<deque<uint>*>, cmp_deque>;
		deque<uint>* c_posi = new deque<uint>(C->row_index->begin() + C->col_range->at(i), C->row_index->begin() + C->col_range->at(i + 1));
		if (!c_posi->empty()) {
			c_posi->push_back(iindex++);
			posi->push(c_posi);
		}

		for (uint j = B->col_range->at(i); j < B->col_range->at(i + 1); j++) {
			uint col_j = B->row_index->at(j);
			deque<uint>* a_posi = new deque<uint>(A->row_index->begin() + A->col_range->at(col_j), A->row_index->begin() + A->col_range->at(col_j + 1));
			if (!a_posi->empty()) {
				a_posi->push_back(iindex++);
				posi->push(a_posi);
			}
		}

		/*
		第二步: 输出排列, 同时生成C的row_index
		final_result是对数据的操作, 语义如下:
		-1表示轮到下一个元素了, 序号表示要加上对应列的一个元素
		可以一次发一部分给value_phase, 即把push改成进程间通信
		*/
		queue<uint>* final_result = new queue<uint>;
		vector<uint>* col_range = new vector<uint>;
		deque<uint>* min_col = nullptr;
		uint min_row_index = posi->top()->front();
		C1->row_index->push_back(min_row_index);

		while (!posi->empty()) {
			min_col = posi->top();
			posi->pop();

			if (min_row_index != min_col->front()) {
				min_row_index = min_col->front();
				C1->row_index->push_back(min_row_index);
				final_result->push(-1);
			}

			final_result->push(min_col->back());
			min_col->pop_front();

			if (min_col->size() != 1) {
				posi->push(min_col);
			}
			else {
				delete min_col;
			}
		}
		delete posi;
		C1->col_range->push_back(C1->row_index->size());
		/*
		value phase
		先得到每列的数据, 然后按照position phase的结果来加和, 放入.
		*/

		vector<queue<tp>*>* mid_results = new vector<queue<tp>*>;

		// 先把C里的值输进去，最后加
		queue<tp>* c_ele = new queue<tp>;
		for (uint row = C->col_range->at(i); row < C->col_range->at(i + 1); row++) {
			c_ele->push(C->data->at(row));
		}
		if (!c_ele->empty())mid_results->push_back(c_ele);

		// 对B在这一列上的每一个非零元的行号，找到A中列号与这些行号相同的列，然后这些列被非零元数乘
		for (uint j = B->col_range->at(i); j < B->col_range->at(i + 1); j++) {
			queue<tp>* jcol = new queue<tp>;
			for (uint k = A->col_range->at(B->row_index->at(j)); k < A->col_range->at(B->row_index->at(j) + 1); k++) {
				jcol->push(A->data->at(k) * B->data->at(j));
			}
			if (!jcol->empty())mid_results->push_back(jcol);
		}

		// 作和, 放入
		tp sum = 0;
		while (!final_result->empty()) {
			if (~final_result->front()) {
				sum += mid_results->at(final_result->front())->front();
				mid_results->at(final_result->front())->pop();
			}
			else {
				C1->data->push_back(sum);
				sum = 0;
			}
			final_result->pop();
		}
		delete final_result;
		C1->data->push_back(sum);
		for (auto i = mid_results->begin(); i != mid_results->end(); i++) {
			delete *i;
		}
		delete mid_results;
	}
	
	_C = C1;
}
#endif

#endif