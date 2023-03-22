#pragma once
#pragma once
#ifndef COMPRESSED_SPARSE_MULT
#define COMPRESSED_SPARSE_MULT

#include "transfer.h"
#include <iostream>
#include <cassert>
#include <queue>
#include <map>
#include <deque>

template<typename tp>
struct halfCord {
	uint row, sign;//sign��Ϊ����merge��ʱ����ʾ���Ԫ�������Ĵν��
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

template<typename tp>
bool is_equal2(const halfCord<tp>* a, const halfCord<tp>* b) {
	return (a->row == b->row);
}

//��ı�_C��ָ��
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

	//�൱��ÿ����һ�������B��һ�������������ֱ��ƴ����
	for (uint i = 0; i < B->col(); i++) {
		vector<queue<halfCord<tp>*>*>* mid_results = new vector<queue<halfCord<tp>*>*>;

		//�Ȱ�C���ֵ���ȥ������
		queue<halfCord<tp>*>* c_ele = new queue<halfCord<tp>*>;
		for (uint row = C->col_range->at(i); row < C->col_range->at(i + 1); row++) {
			halfCord<tp>* c = new halfCord<tp>(C->row_index->at(row), C->data->at(row), 0);
			c_ele->push(c);
		}
		if (!c_ele->empty())mid_results->push_back(c_ele);

		//��B����һ���ϵ�ÿһ������Ԫ���кţ��ҵ�A���к�����Щ�к���ͬ���У�Ȼ����Щ�б�����Ԫ����
		for (uint j = B->col_range->at(i); j < B->col_range->at(i + 1); j++) {
			queue<halfCord<tp>*>* jcol = new queue<halfCord<tp>*>;
			for (uint k = A->col_range->at(B->row_index->at(j)); k < A->col_range->at(B->row_index->at(j) + 1); k++) {
				halfCord<tp>* ab = new halfCord<tp>(A->row_index->at(k), A->data->at(k) * B->data->at(j), mid_results->size());
				jcol->push(ab);
			}
			if (!jcol->empty())mid_results->push_back(jcol);
		}
		
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


#endif