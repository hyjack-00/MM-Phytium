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

//ע�⣺rd()��һ���޷�������
template<typename tp>
void init_random(tp* a, cui row, cui col) {
	random_device rd;
	uint size = row * col;
	uint zerorow = rd() % row;
	uint zerocol = rd() % col;
	//��ʼ��������ά����
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
	cui row;//ֻҪ����row_index�������ֵ�Ϳ�����
	vector<tp>* data;
	vector<uint>* row_index;//ͬһ������к�
	vector<uint>* col_range;//��Щ��ͬһ�е�
	bool trans; // �����0������csc��������csr���൱�ڴ洢ת�õ�csc��

	//һ���յľ���
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

	//ע��trans==1��ζ����������һ�ִ洢��ʽ�洢���������������߼�������ȵġ�
	sparce_matrix(const sparce_matrix<tp>* a,const bool& transpose) :row(transpose ? a->col() : a->row) {
		if (transpose) {
			data = new vector<tp>(a->nnz(), 0);
			trans = !a->trans;
			row_index = new vector<uint>(a->nnz(), 0);

			//����һЩ��ʱ����:col_num����Aת�õ���һ�����ж���Ԫ��
			vector<uint>* col_num = new vector<uint>(a->row, 0);
			for (uint ri = 0; ri < a->nnz(); ri++) {
				col_num->at(a->row_index->at(ri)) += 1;
			}
			uint* col_now = new uint[a->row + 1];//�����Ϊ�˵Ȼ��¼ÿ��data�����λ�á�
			uint tmp = 0;
			for (uint it = 0; it != a->row; it++) {
				col_now[it] = tmp;
				tmp += col_num->at(it);
			}
			col_now[a->row] = tmp;
			col_range = new vector<uint>(col_now, col_now + a->row + 1);


			//����data
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

	//ע�⣺Ĭ���кŵ�������ֱ��������һ�С�����Ҫ��ǿա�
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
	vector<uint>* row_index;//ͬһ������к�
	vector<uint>* col_range;//�к�����Щ��ͬһ�е�
	vector<uint>* col_index;//����ķ�Χ����Щ�е�
	bool trans; // �����0������dcsc��������dcsr���൱�ڴ洢ת�õ�csc��

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

			//����һЩ��ʱ����
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
			unordered_map<uint, uint>* col_now = new unordered_map<uint, uint>;//�����Ϊ�˵Ȼ��¼ÿ��data�����λ�á�
			col_range->push_back(0);
			for (auto it = num->begin(); it != num->end(); it++) {
				col_now->insert(pair<uint, uint>(it->first, col_range->back()));
				col_index->push_back(it->first);
				col_range->push_back(col_range->back() + it->second);
				
			}
			
			//����data
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

	//ͨ������ķ����ֵ����������Ԫ�鹹����
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

