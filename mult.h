#pragma once
#ifndef SPARSE_MULT
#define SPARSE_MULT

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
		if (lhs->col > rhs->col || (lhs->col == rhs->col && lhs->row > rhs->row) || (lhs->col == rhs->col && lhs->row == rhs->row && lhs->data > rhs->data)) {
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

//这里假定ABC是列，行，列优先，并且C的数据作废。
template<typename tp>
void mult(dc_sparce_matrix<tp>* A, dc_sparce_matrix<tp>* B, dc_sparce_matrix<tp>* C) {
	assert(A->trans == 0 && B->trans == 1 && C->trans == 1);
	vector<queue<Cord<tp>*>*>* mid_results = new vector<queue<Cord<tp>*>*>;
	uint a = 0, b = 0;//逻辑指针
	while (a != A->nzc() && b != B->nzc()) {
		if ((*(A->col_index))[a] == (*(B->col_index))[b]) {
			queue<Cord<tp>*>* desc_res = new queue<Cord<tp>*>;
			mid_results->push_back(desc_res);
			//生成A的一列和B的一列乘出来的笛卡尔积。
			for (uint i = (*(A->col_range))[a]; i < (*(A->col_range))[a + 1]; i++) {
				for (uint j = (*(B->col_range))[b]; j < (*(B->col_range))[b + 1]; j++) {
					Cord<tp>* a = new Cord(A->row_index->at(i), B->row_index->at(j), (A->data->at(i) * B->data->at(j)), mid_results->size() - 1);
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

	//空的就直接返回了，否则装到C里会出一些问题
	if (mid_results->empty()) return;

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

	//把deque里的数据装到C里。
	C->data = new vector<tp>;
	C->row_index = new vector<uint>;
	C->col_range = new vector<uint>;
	C->col_index = new vector<uint>;
	uint index = final_result.front()->row;
	C->col_range->push_back(0);
	C->col_index->push_back(index);
	for (auto i = final_result.begin(); i != final_result.end(); i++) {
		if ((*i)->row != index) {
			index = (*i)->row;
			C->col_index->push_back((*i)->row);
			C->col_range->push_back(C->data->size());
		}
		C->row_index->push_back((*i)->col);
		C->data->push_back((*i)->data);
	}
	C->col_range->push_back(C->data->size());
	return;
}

#endif // !SPARSE_MULT

