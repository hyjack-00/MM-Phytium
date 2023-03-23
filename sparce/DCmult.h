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
	uint row, col, sign;//sign��Ϊ����merge��ʱ����ʾ���Ԫ�������Ĵν��
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

//����ٶ�ABC���У��У�������
//ע�⣺Ŀǰ�İ汾�£����ñ�����֮��C��Ϊһ��ָ���ı�����ָ��λ�á�������������ף�
//��ǰ�汾�£�һ�γ˷�������Ҫ�Ĵ�ת���洢��ʽ�������и��õķ�����
template<typename tp>
void dcgemm(const dc_sparce_matrix<tp>* _A, const dc_sparce_matrix<tp>* _B, dc_sparce_matrix<tp>* _C) {
	// Ԥ����
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

	//�Ȱ�C���ֵ���ȥ������
	queue<Cord<tp>*>* c_ele = new queue<Cord<tp>*>;
	for (uint col = 0; col < C->col_index->size(); col++) {
		for (uint row = C->col_range->at(col); row < C->col_range->at(col + 1); row++) {
			Cord<tp>* a = new Cord<tp>(C->col_index->at(col), C->row_index->at(row), C->data->at(row), 0);
			c_ele->push(a);
		}
	}
	mid_results->push_back(c_ele);

	uint a = 0, b = 0;//�߼�ָ��
	while (a != A->nzc() && b != B->nzc()) {
		if ((*(A->col_index))[a] == (*(B->col_index))[b]) {
			queue<Cord<tp>*>* desc_res = new queue<Cord<tp>*>;
			mid_results->push_back(desc_res);
			//����A��һ�к�B��һ�г˳����ĵѿ�������
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

	//��deque�������װ��_C�����ᵼ��_C����λ�á�����û�б���λ�õı�Ҫ��������д������.
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
		������÷���ֵ��д��������� _C �����ڴ治Ӧ���� gemm �����ͷţ�����Ӧ���ɴ���������û������� 
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

