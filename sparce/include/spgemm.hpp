#ifndef SPGEMM_HPP
#define SPGEMM_HPP

#include "spgemm_utils.h"
#include <arm_neon.h>

template<typename CSROrdinal, typename Value, typename HOrdType, typename LOrdType>
void twoLSymbolic(Matrix<CSROrdinal, CSROrdinal, Value> & A,
		Matrix<CSROrdinal, CSROrdinal, Value> & B_org,
		TwoLevelMatrix<CSROrdinal, Value, HOrdType, LOrdType> & B,
		Matrix<CSROrdinal, CSROrdinal, Value> & C,
		UpperBounds<CSROrdinal, Value> & upb,
		CSROrdinal chunk_size) {

	CSROrdinal nrowsA = A.nrows();
	CSROrdinal * ptrA = A.Ptr();
	CSROrdinal * colindicesA = A.ColIndices();

	CSROrdinal nrowsB = B_org.nrows();
	CSROrdinal ncolsB = B_org.ncols();
	CSROrdinal * ptrB = B_org.Ptr();
	CSROrdinal * colindicesB = B_org.ColIndices();

	CSROrdinal widthH = ((long)1<<B.num_bits.higher);
	CSROrdinal widthL = ((long)1<<B.num_bits.lower);

	CSROrdinal * ptrBH = B.H.Ptr();
	HOrdType * colindicesBH = B.H.ColIndices();
	CSROrdinal * valuesBH = B.H.Values();
	LOrdType * colindicesBL = B.L;

	CSROrdinal * ptrC = C.Ptr();
	ptrC[0] = 0;
	if (USE_COMPRESSION && ((nrowsB >> 5) < (L2_CACHE_SIZE/sizeof(uint32_t)))) {
#pragma omp parallel
		{
			DenseHashMap<CSROrdinal, Value, uint32_t, uint16_t> hashmapLC(((uint32_t)(ncolsB + 31))>>5, upb.max_widthComp);
#pragma omp for schedule(dynamic, chunk_size)
			for (CSROrdinal v=0; v<nrowsA; ++v) {
				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v+1] - ptrA[v] == 1) {
					CSROrdinal u=colindicesA[ptrA[v]];
					ptrC[v+1] = ptrB[u+1] - ptrB[u];
					continue;
				} else if (ptrA[v+1] - ptrA[v] == 0) {
					ptrC[v+1] = 0;
					continue;
				}

				for (CSROrdinal u_pos=ptrA[v]; u_pos<ptrA[v+1]; ++u_pos) {
					CSROrdinal u=colindicesA[u_pos];
					CSROrdinal start=B.row_ptr_compress[u];
					CSROrdinal end=B.row_ptr_compress[u+1];
					for (CSROrdinal k=start; k<end; ++k)
						hashmapLC.insertOr(B.LA_compress[k], B.values_compress[k]);
				}
				ptrC[v+1] = hashmapLC.getUsedSizeOr();
				hashmapLC.resetSize();
			}
		}
	} else 
		if (ncolsB < L2_CACHE_SIZE/sizeof(CSROrdinal)) {
#pragma omp parallel
		{
			DenseHashMap<CSROrdinal, Value, bool, CSROrdinal> hashmap(ncolsB, upb.max_width, false);
#pragma omp for schedule(dynamic, chunk_size)
			for (CSROrdinal v=0; v<nrowsA; ++v) {
				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v+1] - ptrA[v] == 1) {
					CSROrdinal u=colindicesA[ptrA[v]];
					ptrC[v+1] = ptrB[u+1] - ptrB[u];
					continue;
				} else if (ptrA[v+1] - ptrA[v] == 0) {
					ptrC[v+1] = 0;
					continue;
				}

				for (CSROrdinal u_pos=ptrA[v]; u_pos<ptrA[v+1]; ++u_pos) {
					CSROrdinal u=colindicesA[u_pos];
					CSROrdinal start=ptrB[u];
					CSROrdinal end=ptrB[u+1];
					for (CSROrdinal w_pos=start; w_pos<end; ++w_pos) {
						CSROrdinal w=colindicesB[w_pos];
						hashmap.insert(w);
					}
				}
				ptrC[v+1] = hashmap.getUsedSizeOr();
				hashmap.reset();
			}
		}
	} 
	// cache is not big enough
	else 
	{
		if (USE_COMPRESSION) {
#pragma omp parallel
			{
				SparseHashMap<CSROrdinal, Value, false, HOrdType> hashmapH(widthH, upb);
				assert(widthL >> 5);
				DenseHashMap<CSROrdinal, Value, uint32_t, LOrdType> hashmapLC(widthL >> 5, widthL >> 5);

#pragma omp for schedule(dynamic, chunk_size)
				for (CSROrdinal v = 0; v < nrowsA; ++v) {
					// if degree of v in A is less than one, then we can just add the degree
					if (ptrA[v + 1] - ptrA[v] == 1) {
						CSROrdinal u = colindicesA[ptrA[v]];
						ptrC[v + 1] = valuesBH[ptrBH[u + 1]] - valuesBH[ptrBH[u]];
						continue;
					}
					else if (ptrA[v + 1] - ptrA[v] == 0) {
						ptrC[v + 1] = 0;
						continue;
					}

					CSROrdinal v_sizeC = 0;
					// gather all the position info
					for (CSROrdinal u_pos = ptrA[v]; u_pos < ptrA[v + 1]; ++u_pos) {
						CSROrdinal u = colindicesA[u_pos];
						for (CSROrdinal wH_pos = ptrBH[u]; wH_pos < ptrBH[u + 1]; ++wH_pos) {
							HOrdType wH = colindicesBH[wH_pos];
							hashmapH.insert(wH, wH_pos);
						}
					}

					CSROrdinal keynum = hashmapH.getKeysNum();

					for (CSROrdinal k = 0; k < keynum; ++k) {
						HOrdType colindex;
						hashmapH.getKey(k, colindex);

						CSROrdinal next;
						CSROrdinal pos;
						bool moreValue = hashmapH.getFirstValue(colindex, pos, next);
						if (!moreValue) {
							v_sizeC += valuesBH[pos + 1] - valuesBH[pos];
						}
						else {
							while (true) {
								CSROrdinal start = B.seg_ptr_compress[pos];
								CSROrdinal end = B.seg_ptr_compress[pos + 1];
								for (CSROrdinal k = start; k < end; ++k)
									hashmapLC.insertOr(B.L_compress[k], B.values_compress[k]);
								if (!hashmapH.getCollisions(pos, next))
									break;
							}

							v_sizeC += hashmapLC.getUsedSizeOr();
							hashmapLC.resetSize();
						}
					}
					hashmapH.resetSize();
					// store the size of row in C
					ptrC[v + 1] = v_sizeC;
				}
			}
		}
		// not_compression
		else
		{
#pragma omp parallel
			{
				SparseHashMap<CSROrdinal, Value, false, HOrdType> hashmapH(widthH, upb);
				DenseHashMap<CSROrdinal, Value, bool, LOrdType> hashmapL(widthL, widthL);

#pragma omp for schedule(dynamic, chunk_size)
				for (CSROrdinal v = 0; v < nrowsA; ++v) {

					// if degree of v in A is less than one, then we can just add the degree
					if (ptrA[v + 1] - ptrA[v] == 1) {
						CSROrdinal u = colindicesA[ptrA[v]];
						ptrC[v + 1] = valuesBH[ptrBH[u + 1]] - valuesBH[ptrBH[u]];

						continue;
					}
					else if (ptrA[v + 1] - ptrA[v] == 0) {

						ptrC[v + 1] = 0;
						continue;
					}

					CSROrdinal v_sizeC = 0;
					// gather all the position info
					for (CSROrdinal u_pos = ptrA[v]; u_pos < ptrA[v + 1]; ++u_pos) {
						CSROrdinal u = colindicesA[u_pos];
						for (CSROrdinal wH_pos = ptrBH[u]; wH_pos < ptrBH[u + 1]; ++wH_pos) {
							HOrdType wH = colindicesBH[wH_pos];
							hashmapH.insert(wH, wH_pos);
						}
					}

					CSROrdinal keynum = hashmapH.getKeysNum();

					for (CSROrdinal k = 0; k < keynum; ++k) {
						HOrdType colindex;
						hashmapH.getKey(k, colindex);

						CSROrdinal next;
						CSROrdinal pos;
						bool moreValue = hashmapH.getFirstValue(colindex, pos, next);
						if (!moreValue) {
							v_sizeC += valuesBH[pos + 1] - valuesBH[pos];
						}
						else {
							while (true) {
								CSROrdinal start = valuesBH[pos];
								CSROrdinal end = valuesBH[pos + 1];
								for (CSROrdinal k = start; k < end; ++k)
									hashmapL.insert(colindicesBL[k]);
								if (!hashmapH.getCollisions(pos, next))
									break;
							}

							v_sizeC += hashmapL.getUsedSize();
							hashmapL.reset();	
						}
					}
					hashmapH.resetSize();
					// store the size of row in C
					ptrC[v + 1] = v_sizeC;
				}
			}
		}
	}
	std::partial_sum(ptrC, ptrC + nrowsA + 1, ptrC);
}

#include "LN-sp.hpp"

template<typename CSROrdinal, typename Value, typename HOrdType, typename LOrdType>
void Multiplication(
		Matrix<CSROrdinal, CSROrdinal, Value> & A,
		Matrix<CSROrdinal, CSROrdinal, Value> & B_org,
		NumBits nbits,
		Matrix<CSROrdinal, CSROrdinal, Value> & C,
		bool verb = false,
		bool statistic_verb = false) {

	// Construct S, and check if use compression and if uses, stores compression
	// results in B as well
	if (verb) HooksRegionBegin("Construct S and get upperbound");
	TwoLevelMatrix<CSROrdinal, Value, HOrdType, LOrdType> B(nbits);
	constructHL<CSROrdinal, Value, HOrdType, LOrdType>(B_org, B, 512, statistic_verb);
	UpperBounds<CSROrdinal, Value> upb = getUpperBounds(A, B_org, B, 512, statistic_verb);
	if (verb) HooksRegionEnd("Construct S and get upperbound", STR_LEN, NUM_LEN);
	if (statistic_verb) upb.print();

	// allocate for ptr of C
	if (verb) HooksRegionBegin("MallocPtr");
	C.AllocatePtr(A.nrows() + 1);
	if (verb) HooksRegionEnd("MallocPtr", STR_LEN, NUM_LEN);

	// symbolic: compute ptr of C
	if (verb) HooksRegionBegin("Symbolic");
	CSROrdinal chunk_size = 0;
#pragma omp parallel
	{
		chunk_size = A.nrows() / omp_get_num_threads() ;
	}
	twoLSymbolic(A, B_org, B, C, upb, chunk_size / TASK_PER_THREAD_SYM > 0 ? chunk_size / TASK_PER_THREAD_SYM : 1);
	if (verb) HooksRegionEnd("Symbolic", STR_LEN, NUM_LEN);

	// allocate for col indices and values of C
	if (verb) HooksRegionBegin("MallocColIndices");
	C.AllocateColIndices(C.Ptr()[C.nrows()]);
	C.AllocateValues(C.Ptr()[C.nrows()]);
	if (verb) HooksRegionEnd("MallocColIndices", STR_LEN, NUM_LEN);

	// numeric: compute for col indices and values in C
	if (verb) HooksRegionBegin("Numeric");
	twoLNumeric(A, B_org, B, C, upb, chunk_size / TASK_PER_THREAD_NUM > 0 ? chunk_size / TASK_PER_THREAD_NUM : 1);
	if (verb) HooksRegionEnd("Numeric", STR_LEN, NUM_LEN);
}

// Multiplication where the maximum degree or A/B is 1.
template<typename CSROrdinal, typename Value>
void Multiplication_trival(
		Matrix<CSROrdinal, CSROrdinal, Value> & A,
		Matrix<CSROrdinal, CSROrdinal, Value> & B,
		Matrix<CSROrdinal, CSROrdinal, Value> & C,
		bool verb = false)
{
	if (verb) HooksRegionBegin("Construct S and get upperbound");
	// for trival case, we don't construct S, but to keep
	if (verb) HooksRegionEnd("Construct S and get upperbound", STR_LEN, NUM_LEN);

	if (verb) HooksRegionBegin("MallocPtr");
	// allocate for ptr
	CSROrdinal * ptrA = A.Ptr();
	CSROrdinal * colindicesA = A.ColIndices();
	Value * valuesA = A.Values();

	CSROrdinal nrowsB = B.nrows();
	CSROrdinal * ptrB = B.Ptr();
	CSROrdinal * colindicesB = B.ColIndices();
	Value * valuesB = B.Values();

	C.AllocatePtr(A.nrows() + 1);
	CSROrdinal * ptrC = C.Ptr();
	if (verb) HooksRegionEnd("MallocPtr", STR_LEN, NUM_LEN);

	if (verb) HooksRegionBegin("Symbolic");
	ptrC[0]=0;
	if (A.nrows() == A.nnz() && B.nrows() == B.nnz()) {
#pragma omp parallel for schedule (static)
		for (CSROrdinal v=0; v<A.nrows(); ++v)
			ptrC[v+1]=v+1;
	} else {
#pragma omp parallel for schedule (static)
		for (CSROrdinal v=0; v<A.nrows(); ++v) {
			if (ptrA[v+1] == ptrA[v]) {
				ptrC[v+1]=0;
				continue;
			}
			CSROrdinal u=colindicesA[ptrA[v]];
			ptrC[v+1]=ptrB[u+1]-ptrB[u];
		}
		std::partial_sum(ptrC, ptrC + C.nrows() + 1, ptrC);
	}
	if (verb) HooksRegionEnd("Symbolic", STR_LEN, NUM_LEN);

	if (verb) HooksRegionBegin("MallocColIndices");
	// allocate for col indices and values
	C.AllocateColIndices(C.Ptr()[C.nrows()]);
	C.AllocateValues(C.Ptr()[C.nrows()]);
	CSROrdinal * colindicesC = C.ColIndices();
	Value * valuesC = C.Values();
	if (verb) HooksRegionEnd("MallocColIndices", STR_LEN, NUM_LEN);

	if (verb) HooksRegionBegin("Numeric");
#pragma omp parallel for schedule (static)
	for (CSROrdinal v=0; v<A.nrows(); ++v) {
		if (ptrC[v+1] == ptrC[v]) continue;
		CSROrdinal u=colindicesA[ptrA[v]];
		colindicesC[ptrC[v]]=colindicesB[ptrB[u]];
		valuesC[ptrC[v]]=valuesA[v]*valuesB[ptrB[u]];
	}
	if (verb) HooksRegionEnd("Numeric", STR_LEN, NUM_LEN);
}

template<typename CSROrdinal, typename Value>
void SpGEMM (
		Matrix<CSROrdinal, CSROrdinal, Value> & A,
		Matrix<CSROrdinal, CSROrdinal, Value> & B,
		Matrix<CSROrdinal, CSROrdinal, Value> & C,
		bool verb = false,
		bool statistic_verb = false) {
	if (A.ncols() != B.nrows()) {
		std::cerr << "Error: the number of columns in A is not equal to the number of rows in B : "<< A.ncols() << ", " << B.nrows() << std::endl;
		std::exit( 1 );
	}

	C.init(A.nrows(), B.ncols());

	if (statistic_verb)
		std::cout << std::setw(STR_LEN) << "Statistics for SpGEMM" << " :" << std::endl;
	if ((A.maxDegree() == 1 && B.maxDegree() == 1) && !statistic_verb) {
		Multiplication_trival(A, B, C, verb);
		return;
	}

	NumBits nbits = getNumBits<CSROrdinal, Value>(B.ncols());
	if (statistic_verb) nbits.print();

	// assign different types for higher bits part and lower bits part
	bool uint16ForH = nbits.higher > 8 && nbits.higher <= 16;
	bool uint8ForH = nbits.higher <= 8;
	bool uint16ForL = nbits.lower > 8 && nbits.lower <= 16;
	bool uint8ForL = nbits.lower <= 8;
	if (uint8ForH && uint8ForL)
		Multiplication<CSROrdinal, Value, uint8_t, uint8_t>(A, B, nbits, C, verb, statistic_verb );
	else if (uint16ForH && uint16ForL)
		Multiplication<CSROrdinal, Value, uint16_t, uint16_t>(A, B, nbits, C, verb, statistic_verb );
	else if (uint16ForH && uint8ForL)
		Multiplication<CSROrdinal, Value, uint16_t, uint8_t>(A, B, nbits, C, verb, statistic_verb );
	else if (uint8ForH && uint16ForL)
		Multiplication<CSROrdinal, Value, uint8_t, uint16_t>(A, B, nbits, C, verb, statistic_verb );
	else if (uint8ForH)
		Multiplication<CSROrdinal, Value, uint8_t, CSROrdinal>(A, B, nbits, C, verb, statistic_verb );
	else if (uint16ForH)
		Multiplication<CSROrdinal, Value, uint16_t, CSROrdinal>(A, B, nbits, C, verb, statistic_verb );
	else if (uint8ForL)
		Multiplication<CSROrdinal, Value, CSROrdinal, uint8_t>(A, B, nbits, C, verb, statistic_verb );
	else if (uint16ForL)
		Multiplication<CSROrdinal, Value, CSROrdinal, uint16_t>(A, B, nbits, C, verb, statistic_verb );
	else
		Multiplication<CSROrdinal, Value, CSROrdinal, CSROrdinal>(A, B, nbits, C, verb, statistic_verb );
}
#endif