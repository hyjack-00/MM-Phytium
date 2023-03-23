#pragma once
#ifndef DOUBLE_COMPRESSED_SPARSE_MULT
#define DOUBLE_COMPRESSED_SPARSE_MULT

#include "transfer.h"
#include <iostream>
#include <cassert>
#include <queue>
#include <map>
#include <deque>

typedef unsigned int uint;
typedef const unsigned int cui;

template<typename tp>
struct Cord {
	uint row, col, sign;//sign是为了在merge的时候，提示这个元素来自哪次结果
	tp data;
	Cord(cui row_index, cui col_index, const tp d, cui s) :row(row_index), col(col_index), data(d), sign(s){}
};

template<typename tp>
struct cmp {
	bool reverse;
	cmp(const bool& revparam = false) {
		reverse = revparam;
	}
	bool operator() (const Cord<tp>* lhs, const Cord<tp>* rhs) const {
		if (lhs->row > rhs->row || (lhs->row == rhs->row && lhs->col > rhs->col)) {
			return !reverse;
		}
		else {
			return reverse;
		}
	}
};

template<typename tp>
bool is_equal(const Cord<tp>* a, const Cord<tp>* b) {
	return (a->col == b->col && a->row == b->row);
}

//这里假定ABC是列，行，行优先
//注意：目前的版本下，调用本函数之后，C作为一个指针会改变其所指的位置。这可能有所不妥？
//当前版本下，一次乘法可能需要四次转换存储方式。可能有更好的方法。
template<typename tp>
void dcgemm(const dc_sparce_matrix<tp>* _A, const dc_sparce_matrix<tp>* _B, dc_sparce_matrix<tp>* _C) {
	// 预处理
	const dc_sparce_matrix<tp>* A, * B, * C;
	if (_A->trans) {
		A = new dc_sparce_matrix<tp>(*_A, 1);
	}
	else {
		A = _A;
	}
	if (!_B->trans) {
		B = new dc_sparce_matrix<tp>(*_B, 1);
	}
	else {
		B = _B;
	}
	if (!_C->trans) {
		C = new dc_sparce_matrix<tp>(*_C, 1);
	}
	else {
		C = _C;
	}
	assert(!A->trans && B->trans && C->trans);
	vector<queue<Cord<tp>*>*>* mid_results = new vector<queue<Cord<tp>*>*>;

	//先把C里的值输进去，最后加
	queue<Cord<tp>*>* c_ele = new queue<Cord<tp>*>;
	for (uint col = 0; col < C->col_index->size(); col++) {
		for (uint row = C->col_range->at(col); row < C->col_range->at(col + 1); row++) {
			Cord<tp>* a = new Cord<tp>(C->col_index->at(col), C->row_index->at(row), C->data->at(row), 0);
			c_ele->push(a);
		}
	}
	mid_results->push_back(c_ele);

	uint a = 0, b = 0;//逻辑指针
	while (a != A->nzc() && b != B->nzc()) {
		if ((*(A->col_index))[a] == (*(B->col_index))[b]) {
			queue<Cord<tp>*>* desc_res = new queue<Cord<tp>*>;
			mid_results->push_back(desc_res);
			//生成A的一列和B的一列乘出来的笛卡尔积。
			for (uint i = (*(A->col_range))[a]; i < (*(A->col_range))[a + 1]; i++) {
				for (uint j = (*(B->col_range))[b]; j < (*(B->col_range))[b + 1]; j++) {
					Cord<tp>* a = new Cord<tp>(A->row_index->at(i), B->row_index->at(j), (A->data->at(i) * B->data->at(j)), mid_results->size()-1);
					desc_res->push(a);
				}
			}
			a++;
			b++;
		}
		else if ((*(A->col_index))[a] > (*(B->col_index))[b]) {
			b++;
		}
		else {
			a++;
		}
	}

	priority_queue<Cord<tp>*, vector<Cord<tp>*>, cmp<tp>> compare_result;
	deque<Cord<tp>*> final_result;
	for (uint i = 0; i < mid_results->size(); i++) {
		compare_result.push(mid_results->at(i)->front());
		(*mid_results)[i]->pop();
	}

	while (!compare_result.empty()) {
		Cord<tp>* tmp = compare_result.top();
		compare_result.pop();
		if (!final_result.empty() && is_equal(final_result.back(), tmp)) {
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

	if (_A != A) delete A;
	if (_B != B) delete B;
	if (_C != C) delete C;

	//把deque里的数据装到_C里。这里会导致_C被改位置。可能没有保持位置的必要，所以先写成这样.
	dc_sparce_matrix<tp>* C1 = new dc_sparce_matrix<tp>(final_result.begin(), final_result.end(), 1);
	if (_C->trans) {
		delete _C;
		_C = C1;
	}else{
		delete _C;
		_C = new dc_sparce_matrix<tp>(*C1, 1);
		delete C1;
	}
	return;

	/* 
		如果采用返回值的写法，输入的 _C 矩阵内存不应该由 gemm 函数释放，而是应当由创建矩阵的用户方处理 
		-- by Huangyj
	*/
	// dc_sparce_matrix<tp>* C1 = new dc_sparce_matrix<tp>(final_result.begin(), final_result.end(), 1);
	// if (_C->trans) {
	// 	return C1;
	// }else{
	// 	dc_sparce_matrix<tp>* C1_temp = new dc_sparce_matrix<tp>(*C1, 1);
	// 	delete C1;
	// 	return C1_temp;
	// }
	// return nullptr;
}

#endif // !DOUBLE_COMPRESSED_SPARSE_MULT

