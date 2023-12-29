struct Default_tag {};
struct u32_f32_0_u8 {};
struct u32_f32_0_u16 {};
struct u32_f32_0_u32 {};

template<typename CSROrdinal, typename Value, typename HOrdType, typename LOrdType> struct TagDispatch {
	using Tag = Default_tag;
};
template<typename HOrdType> struct TagDispatch<uint32_t, float32_t, HOrdType, uint8_t> {
	using Tag = u32_f32_0_u8;
};
template<typename HOrdType> struct TagDispatch<uint32_t, float32_t, HOrdType, uint16_t> {
	using Tag = u32_f32_0_u16;
};
template<typename HOrdType> struct TagDispatch<uint32_t, float32_t, HOrdType, uint32_t> {
	using Tag = u32_f32_0_u32;
};

template<typename CSROrdinal, typename Value, typename HOrdType, typename LOrdType>
void _2LN(Matrix<CSROrdinal, CSROrdinal, Value>& A,
	Matrix<CSROrdinal, CSROrdinal, Value>& B_org,
	TwoLevelMatrix<CSROrdinal, Value, HOrdType, LOrdType>& B,
	Matrix<CSROrdinal, CSROrdinal, Value>& C,
	UpperBounds<CSROrdinal, Value>& upb,
	CSROrdinal chunk_size, Default_tag) {

	CSROrdinal nrowsA = A.nrows();
	CSROrdinal* ptrA = A.Ptr();
	CSROrdinal* colindicesA = A.ColIndices();
	Value* valuesA = A.Values();

	CSROrdinal nrowsB = B_org.nrows();
	CSROrdinal ncolsB = B_org.ncols();
	CSROrdinal* ptrB = B_org.Ptr();
	CSROrdinal* colindicesB = B_org.ColIndices();
	Value* valuesB = B_org.Values();

	CSROrdinal* ptrC = C.Ptr();
	CSROrdinal* colindicesC = C.ColIndices();
	Value* valuesC = C.Values();

	CSROrdinal widthH = ((long)1 << B.num_bits.higher);
	CSROrdinal widthL = ((long)1 << B.num_bits.lower);

	// 如果一个CACHE能装得下
	if (nrowsA < L2_CACHE_SIZE / sizeof(Value)) {
#pragma omp parallel
		{
			DenseHashMap<CSROrdinal, Value, Value, CSROrdinal> hashmap(ncolsB, upb.max_width, inf<Value>());
#pragma omp for schedule(dynamic, chunk_size)
			for (CSROrdinal v = 0; v < nrowsA; ++v) {
				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v + 1] - ptrA[v] == 1) {
					CSROrdinal u = colindicesA[ptrA[v]];
					Value valvu = valuesA[ptrA[v]];
					CSROrdinal pos_vw = ptrC[v];

					
					for (CSROrdinal pos_w=ptrB[u]; pos_w<ptrB[u+1]; pos_w++, pos_vw++) {
						colindicesC[pos_vw]=colindicesB[pos_w];
						valuesC[pos_vw]=valuesB[pos_w]*valvu;
					}
					
					continue;

				}
				else if (ptrA[v + 1] - ptrA[v] == 0)
					continue;
				// if degree of v in A is more than one
				
				for (CSROrdinal u_pos=ptrA[v]; u_pos<ptrA[v+1]; ++u_pos) {
					CSROrdinal u=colindicesA[u_pos];
					Value valvu=valuesA[u_pos];
					CSROrdinal start=ptrB[u];
					CSROrdinal end=ptrB[u+1];
					for (CSROrdinal w_pos=start; w_pos<end; ++w_pos) {
						CSROrdinal w=colindicesB[w_pos];
						Value valuw = valuesB[w_pos];
						hashmap.insertInc(w, valvu*valuw);
					}
				}

				CSROrdinal nonzerovw;
				Value valuevw;
				CSROrdinal posvw = ptrC[v];
				while (hashmap.getKeyValue(nonzerovw, valuevw)) {
					colindicesC[posvw] = nonzerovw;
					valuesC[posvw++] = valuevw;
				}
				hashmap.resetSize();
			}
		}
	}
	// 如果一个CACHE装不下
	else {
		CSROrdinal* ptrBH = B.H.Ptr();
		HOrdType* colindicesBH = B.H.ColIndices();
		CSROrdinal* valuesBH = B.H.Values();
		Value* valuesBL = B.values;
		LOrdType* colindicesBL = B.L;

#pragma omp parallel
		{
			SparseHashMap<CSROrdinal, Value, true, HOrdType> hashmapH(widthH, upb);
			DenseHashMap<CSROrdinal, Value, Value, LOrdType> hashmapL(widthL, widthL, inf<Value>());

#pragma omp for schedule(dynamic, chunk_size)
			for (CSROrdinal v = 0; v < nrowsA; ++v) {

				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v + 1] - ptrA[v] == 1) {
					CSROrdinal u = colindicesA[ptrA[v]];

					// for each nonzero u-w in B
					Value val = valuesA[ptrA[v]];
					CSROrdinal j = ptrC[v];
					
					for (CSROrdinal w_pos=ptrBH[u]; w_pos<ptrBH[u+1]; ++w_pos) {
						CSROrdinal wH = (CSROrdinal)colindicesBH[w_pos];
						for (CSROrdinal i=valuesBH[w_pos]; i<valuesBH[w_pos+1]; ++i, ++j) {
							valuesC[j] = val * valuesBL[i];
							colindicesC[j] = (wH << B.num_bits.lower) & colindicesBL[i];
						}
					}
					
					continue;
				}
				else if (ptrA[v + 1] - ptrA[v] == 0) {
					continue;
				}
				// if degree of v in A is more than one
				// gather all the position info
				CSROrdinal u_start = ptrA[v];
				CSROrdinal u_end = ptrA[v + 1];
				for (CSROrdinal u_pos = u_start; u_pos < u_end; ++u_pos) {
					CSROrdinal u = colindicesA[u_pos];
					Value val = valuesA[u_pos];
					const CSROrdinal wH_start = ptrBH[u];
					const CSROrdinal wH_end = ptrBH[u + 1];
					for (CSROrdinal wH_pos = wH_start; wH_pos < wH_end; ++wH_pos) {
						HOrdType wH = colindicesBH[wH_pos];
						hashmapH.insert(wH, wH_pos, val);
					}
				}

				CSROrdinal v_sizeC = ptrC[v];
				CSROrdinal keynum = hashmapH.getKeysNum();
				for (CSROrdinal k = 0; k < keynum; ++k) {
					HOrdType nonzero;
					Value      val;
					hashmapH.getKey(k, nonzero, val);
					CSROrdinal next;
					CSROrdinal pos;
					bool moreValue = hashmapH.getFirstValue(nonzero, pos, next);
					CSROrdinal nonzero_wH = (CSROrdinal)nonzero << B.num_bits.lower;
					if (!moreValue) {
						
						CSROrdinal w_pos = valuesBH[pos];
						CSROrdinal end = valuesBH[pos+1];

						for (; w_pos<end; ++w_pos, ++v_sizeC) {
							valuesC[v_sizeC] = val * valuesBL[w_pos];
							colindicesC[v_sizeC] = nonzero_wH & colindicesBL[w_pos];
						}					
					}
					else {
						do {

							CSROrdinal k = valuesBH[pos];
							CSROrdinal end = valuesBH[pos + 1];
							
							for (; k<end; ++k) {
								hashmapL.insertInc(colindicesBL[k], val*valuesBL[k]);
							}
							

						} while (hashmapH.getCollisions(pos, next, val));

						// read from hashmapL and write to C
						LOrdType nonzero_wL;
						Value valuec;
						while (hashmapL.getKeyValue(nonzero_wL, valuec)) {
							colindicesC[v_sizeC] = nonzero_wH & nonzero_wL;
							valuesC[v_sizeC++] = valuec;
						}
						hashmapL.resetSize();
					}
				}
				hashmapH.resetSize();
				assert(v_sizeC == ptrC[v + 1]);
			}
		}
	}
}

template<typename CSROrdinal, typename Value, typename HOrdType, typename LOrdType>
void _2LN(Matrix<CSROrdinal, CSROrdinal, Value>& A,
	Matrix<CSROrdinal, CSROrdinal, Value>& B_org,
	TwoLevelMatrix<CSROrdinal, Value, HOrdType, LOrdType>& B,
	Matrix<CSROrdinal, CSROrdinal, Value>& C,
	UpperBounds<CSROrdinal, Value>& upb,
	CSROrdinal chunk_size, u32_f32_0_u8) {

	CSROrdinal nrowsA = A.nrows();
	CSROrdinal* ptrA = A.Ptr();
	CSROrdinal* colindicesA = A.ColIndices();
	Value* valuesA = A.Values();

	CSROrdinal nrowsB = B_org.nrows();
	CSROrdinal ncolsB = B_org.ncols();
	CSROrdinal* ptrB = B_org.Ptr();
	CSROrdinal* colindicesB = B_org.ColIndices();
	Value* valuesB = B_org.Values();

	CSROrdinal* ptrC = C.Ptr();
	CSROrdinal* colindicesC = C.ColIndices();
	Value* valuesC = C.Values();

	CSROrdinal widthH = ((long)1 << B.num_bits.higher);
	CSROrdinal widthL = ((long)1 << B.num_bits.lower);

	// 如果一个CACHE能装得下
	if (nrowsA < L2_CACHE_SIZE / sizeof(Value)) {
#pragma omp parallel
		{
			DenseHashMap<CSROrdinal, Value, Value, CSROrdinal> hashmap(ncolsB, upb.max_width, inf<Value>());
#pragma omp for schedule(dynamic, chunk_size)
			for (CSROrdinal v = 0; v < nrowsA; ++v) {
				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v + 1] - ptrA[v] == 1) {
					CSROrdinal u = colindicesA[ptrA[v]];
					Value valvu = valuesA[ptrA[v]];
					CSROrdinal pos_vw = ptrC[v];

					/*
					for (CSROrdinal pos_w=ptrB[u]; pos_w<ptrB[u+1]; pos_w++, pos_vw++) {
						colindicesC[pos_vw]=colindicesB[pos_w];
						valuesC[pos_vw]=valuesB[pos_w]*valvu;
					}
					*/
					float32x4_t reg;
					uint32x4_t v;
					CSROrdinal pos_w = ptrB[u];
					for (; (ptrB[u + 1] - pos_w) >> 2; pos_w += 4, pos_vw += 4) {
						reg = vld1q_f32(valuesB + pos_w);
						reg = vmulq_n_f32(reg, valvu);
						vst1q_f32(valuesC + pos_vw, reg);

						v = vld1q_u32(colindicesB + pos_w);
						vst1q_u32(colindicesC + pos_vw, v);
					}

					for (; pos_w < ptrB[u + 1]; pos_w++, pos_vw++) {
						colindicesC[pos_vw] = colindicesB[pos_w];
						valuesC[pos_vw] = valuesB[pos_w] * valvu;
					}
					continue;

				}
				else if (ptrA[v + 1] - ptrA[v] == 0)
					continue;
				// if degree of v in A is more than one
				/*
				for (CSROrdinal u_pos=ptrA[v]; u_pos<ptrA[v+1]; ++u_pos) {
					CSROrdinal u=colindicesA[u_pos];
					Value valvu=valuesA[u_pos];
					CSROrdinal start=ptrB[u];
					CSROrdinal end=ptrB[u+1];
					for (CSROrdinal w_pos=start; w_pos<end; ++w_pos) {
						CSROrdinal w=colindicesB[w_pos];
						Value valuw = valuesB[w_pos];
						hashmap.insertInc(w, valvu*valuw);
					}
				}*/

				for (CSROrdinal u_pos = ptrA[v]; u_pos < ptrA[v + 1]; ++u_pos) {
					CSROrdinal u = colindicesA[u_pos];
					Value valvu = valuesA[u_pos];
					float32x4_t reg;
					CSROrdinal pos_w = ptrB[u];
					for (; (ptrB[u + 1] - pos_w) >> 2; ++pos_w) {
						reg = vld1q_f32(valuesB + pos_w);
						reg = vmulq_n_f32(reg, valvu);
						hashmap.insertInc(colindicesB[pos_w], reg[0]);
						hashmap.insertInc(colindicesB[++pos_w], reg[1]);
						hashmap.insertInc(colindicesB[++pos_w], reg[2]);
						hashmap.insertInc(colindicesB[++pos_w], reg[3]);
					}

					for (; pos_w < ptrB[u + 1]; pos_w++) {
						hashmap.insertInc(colindicesB[pos_w], valvu * valuesB[pos_w]);
					}
				}

				CSROrdinal nonzerovw;
				Value valuevw;
				CSROrdinal posvw = ptrC[v];
				while (hashmap.getKeyValue(nonzerovw, valuevw)) {
					colindicesC[posvw] = nonzerovw;
					valuesC[posvw++] = valuevw;
				}
				hashmap.resetSize();
			}
		}
	}
	// 如果一个CACHE装不下
	else {
		CSROrdinal* ptrBH = B.H.Ptr();
		HOrdType* colindicesBH = B.H.ColIndices();
		CSROrdinal* valuesBH = B.H.Values();
		Value* valuesBL = B.values;
		LOrdType* colindicesBL = B.L;

#pragma omp parallel
		{
			SparseHashMap<CSROrdinal, Value, true, HOrdType> hashmapH(widthH, upb);
			DenseHashMap<CSROrdinal, Value, Value, LOrdType> hashmapL(widthL, widthL, inf<Value>());

#pragma omp for schedule(dynamic, chunk_size)
			for (CSROrdinal v = 0; v < nrowsA; ++v) {

				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v + 1] - ptrA[v] == 1) {
					CSROrdinal u = colindicesA[ptrA[v]];

					// for each nonzero u-w in B
					Value val = valuesA[ptrA[v]];
					CSROrdinal j = ptrC[v];
					/*
					for (CSROrdinal w_pos=ptrBH[u]; w_pos<ptrBH[u+1]; ++w_pos) {
						CSROrdinal wH = (CSROrdinal)colindicesBH[w_pos];
						for (CSROrdinal i=valuesBH[w_pos]; i<valuesBH[w_pos+1]; ++i, ++j) {
							valuesC[j] = val * valuesBL[i];
							colindicesC[j] = (wH << B.num_bits.lower) & colindicesBL[i];
						}
					}
					*/
					for (CSROrdinal pos_w = ptrBH[u]; pos_w < ptrBH[u + 1]; ++pos_w) {
						CSROrdinal wH = (CSROrdinal)colindicesBH[pos_w] << B.num_bits.lower;
						uint32x4_t v;
						uint32x4_t wH4 = vdupq_n_u32(wH);
						float32x4_t reg;
						CSROrdinal i = valuesBH[pos_w], end = valuesBH[pos_w + 1];
						for (; (end - i) >> 3; i += 8, j += 8) {
							reg = vld1q_f32(valuesBL + i);
							reg = vmulq_n_f32(reg, val);
							vst1q_f32(valuesC + j, reg);

							reg = vld1q_f32(valuesBL + i + 4);
							reg = vmulq_n_f32(reg, val);
							vst1q_f32(valuesC + j + 4, reg);
							
							uint8x8_t vec = vld1_u8(colindicesBL + i);
							uint16x8_t vu = vmovl_u8(vec);
							v = vmovl_u16(vget_low_u16(vu));
							v = vandq_u32(v, wH4);
							vst1q_u32(colindicesC + j, v);

							v = vmovl_u16(vget_high_u16(vu));
							v = vandq_u32(v, wH4);
							vst1q_u32(colindicesC + j + 4, v);
						}

						for (; i < end; ++i, ++j) {
							colindicesC[j] = wH & colindicesBL[i];
							valuesC[j] = val * valuesBL[i];
						}
					}
					continue;
				}
				else if (ptrA[v + 1] - ptrA[v] == 0) {
					continue;
				}
				// if degree of v in A is more than one
				// gather all the position info
				CSROrdinal u_start = ptrA[v];
				CSROrdinal u_end = ptrA[v + 1];
				for (CSROrdinal u_pos = u_start; u_pos < u_end; ++u_pos) {
					CSROrdinal u = colindicesA[u_pos];
					Value val = valuesA[u_pos];
					const CSROrdinal wH_start = ptrBH[u];
					const CSROrdinal wH_end = ptrBH[u + 1];
					for (CSROrdinal wH_pos = wH_start; wH_pos < wH_end; ++wH_pos) {
						HOrdType wH = colindicesBH[wH_pos];
						hashmapH.insert(wH, wH_pos, val);
					}
				}

				CSROrdinal v_sizeC = ptrC[v];
				CSROrdinal keynum = hashmapH.getKeysNum();
				for (CSROrdinal k = 0; k < keynum; ++k) {
					HOrdType nonzero;
					Value      val;
					hashmapH.getKey(k, nonzero, val);
					CSROrdinal next;
					CSROrdinal pos;
					bool moreValue = hashmapH.getFirstValue(nonzero, pos, next);
					CSROrdinal nonzero_wH = (CSROrdinal)nonzero << B.num_bits.lower;
					if (!moreValue) {
						uint32x4_t v;
						uint32x4_t wH4 = vdupq_n_u32(nonzero_wH);
						float32x4_t reg;
						CSROrdinal i = valuesBH[pos], end = valuesBH[pos + 1];
						for (; (end - i) >> 3; i += 8, v_sizeC += 8) {
							reg = vld1q_f32(valuesBL + i);
							reg = vmulq_n_f32(reg, val);
							vst1q_f32(valuesC + v_sizeC, reg);

							reg = vld1q_f32(valuesBL + i + 4);
							reg = vmulq_n_f32(reg, val);
							vst1q_f32(valuesC + v_sizeC + 4, reg);
														
							uint8x8_t vec = vld1_u8(colindicesBL + i);
							uint16x8_t vu = vmovl_u8(vec);
							v = vmovl_u16(vget_low_u16(vu));
							v = vandq_u32(v, wH4);
							vst1q_u32(colindicesC + v_sizeC, v);

							v = vmovl_u16(vget_high_u16(vu));
							v = vandq_u32(v, wH4);
							vst1q_u32(colindicesC + v_sizeC + 4, v);
						}

						for (; i < end; ++i, ++v_sizeC) {
							valuesC[v_sizeC] = val * valuesBL[i];
							colindicesC[v_sizeC] = nonzero_wH & colindicesBL[i];
						}

					}
					else {
						do {

							CSROrdinal k = valuesBH[pos];
							CSROrdinal end = valuesBH[pos + 1];
							/*
							for (; k<end; ++k) {
								hashmapL.insertInc(colindicesBL[k], val*valuesBL[k]);
							}
							*/
							float32x4_t reg;
							for (; (end - k) >> 2; ++k) {
								reg = vld1q_f32(valuesBL + k);
								reg = vmulq_n_f32(reg, val);
								hashmapL.insertInc(colindicesBL[k], reg[0]);
								hashmapL.insertInc(colindicesBL[++k], reg[1]);
								hashmapL.insertInc(colindicesBL[++k], reg[2]);
								hashmapL.insertInc(colindicesBL[++k], reg[3]);
							}
							for (; k < end; ++k) {
								hashmapL.insertInc(colindicesBL[k], val * valuesBL[k]);
							}

						} while (hashmapH.getCollisions(pos, next, val));

						// read from hashmapL and write to C
						LOrdType nonzero_wL;
						Value valuec;
						while (hashmapL.getKeyValue(nonzero_wL, valuec)) {
							colindicesC[v_sizeC] = nonzero_wH & nonzero_wL;
							valuesC[v_sizeC++] = valuec;
						}
						hashmapL.resetSize();
					}
				}
				hashmapH.resetSize();
				assert(v_sizeC == ptrC[v + 1]);
			}
		}
	}
}

template<typename CSROrdinal, typename Value, typename HOrdType, typename LOrdType>
void _2LN(Matrix<CSROrdinal, CSROrdinal, Value>& A,
	Matrix<CSROrdinal, CSROrdinal, Value>& B_org,
	TwoLevelMatrix<CSROrdinal, Value, HOrdType, LOrdType>& B,
	Matrix<CSROrdinal, CSROrdinal, Value>& C,
	UpperBounds<CSROrdinal, Value>& upb,
	CSROrdinal chunk_size, u32_f32_0_u16) {

	uint32_t nrowsA = A.nrows();
	uint32_t* ptrA = A.Ptr();
	uint32_t* colindicesA = A.ColIndices();
	float32_t* valuesA = A.Values();

	uint32_t nrowsB = B_org.nrows();
	uint32_t ncolsB = B_org.ncols();
	uint32_t* ptrB = B_org.Ptr();
	uint32_t* colindicesB = B_org.ColIndices();
	float32_t* valuesB = B_org.Values();

	uint32_t* ptrC = C.Ptr();
	uint32_t* colindicesC = C.ColIndices();
	float32_t* valuesC = C.Values();

	uint32_t widthH = ((long)1 << B.num_bits.higher);
	uint32_t widthL = ((long)1 << B.num_bits.lower);

	// 如果一个CACHE能装得下
	if (nrowsA < L2_CACHE_SIZE / sizeof(float32_t)) {
#pragma omp parallel
		{
			DenseHashMap<uint32_t, float32_t, float32_t, uint32_t> hashmap(ncolsB, upb.max_width, inf<float32_t>());
#pragma omp for schedule(dynamic, chunk_size)
			for (uint32_t v = 0; v < nrowsA; ++v) {
				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v + 1] - ptrA[v] == 1) {
					uint32_t u = colindicesA[ptrA[v]];
					float32_t valvu = valuesA[ptrA[v]];
					uint32_t pos_vw = ptrC[v];

					/*
					for (uint32_t pos_w=ptrB[u]; pos_w<ptrB[u+1]; pos_w++, pos_vw++) {
						colindicesC[pos_vw]=colindicesB[pos_w];
						valuesC[pos_vw]=valuesB[pos_w]*valvu;
					}
					*/
					float32x4_t reg;
					uint32x4_t v;
					uint32_t pos_w = ptrB[u];
					for (; (ptrB[u + 1] - pos_w) >> 2; pos_w += 4, pos_vw += 4) {
						reg = vld1q_f32(valuesB + pos_w);
						reg = vmulq_n_f32(reg, valvu);
						vst1q_f32(valuesC + pos_vw, reg);

						v = vld1q_u32(colindicesB + pos_w);
						vst1q_u32(colindicesC + pos_vw, v);
					}

					for (; pos_w < ptrB[u + 1]; pos_w++, pos_vw++) {
						colindicesC[pos_vw] = colindicesB[pos_w];
						valuesC[pos_vw] = valuesB[pos_w] * valvu;
					}
					continue;

				}
				else if (ptrA[v + 1] - ptrA[v] == 0)
					continue;
				// if degree of v in A is more than one
				/*
				for (uint32_t u_pos=ptrA[v]; u_pos<ptrA[v+1]; ++u_pos) {
					uint32_t u=colindicesA[u_pos];
					float32_t valvu=valuesA[u_pos];
					uint32_t start=ptrB[u];
					uint32_t end=ptrB[u+1];
					for (uint32_t w_pos=start; w_pos<end; ++w_pos) {
						uint32_t w=colindicesB[w_pos];
						float32_t valuw = valuesB[w_pos];
						hashmap.insertInc(w, valvu*valuw);
					}
				}*/

				for (uint32_t u_pos = ptrA[v]; u_pos < ptrA[v + 1]; ++u_pos) {
					uint32_t u = colindicesA[u_pos];
					float32_t valvu = valuesA[u_pos];
					float32x4_t reg;
					uint32_t pos_w = ptrB[u];
					for (; (ptrB[u + 1] - pos_w) >> 2; ++pos_w) {
						reg = vld1q_f32(valuesB + pos_w);
						reg = vmulq_n_f32(reg, valvu);
						hashmap.insertInc(colindicesB[pos_w], reg[0]);
						hashmap.insertInc(colindicesB[++pos_w], reg[1]);
						hashmap.insertInc(colindicesB[++pos_w], reg[2]);
						hashmap.insertInc(colindicesB[++pos_w], reg[3]);
					}

					for (; pos_w < ptrB[u + 1]; pos_w++) {
						hashmap.insertInc(colindicesB[pos_w], valvu * valuesB[pos_w]);
					}
				}

				uint32_t nonzerovw;
				float32_t valuevw;
				uint32_t posvw = ptrC[v];
				while (hashmap.getKeyValue(nonzerovw, valuevw)) {
					colindicesC[posvw] = nonzerovw;
					valuesC[posvw++] = valuevw;
				}
				hashmap.resetSize();
			}
		}
	}
	// 如果一个CACHE装不下
	else {
		uint32_t* ptrBH = B.H.Ptr();
		HOrdType* colindicesBH = B.H.ColIndices();
		uint32_t* valuesBH = B.H.Values();
		float32_t* valuesBL = B.values;
		uint16_t* colindicesBL = B.L;

#pragma omp parallel
		{
			SparseHashMap<uint32_t, float32_t, true, HOrdType> hashmapH(widthH, upb);
			DenseHashMap<uint32_t, float32_t, float32_t, uint16_t> hashmapL(widthL, widthL, inf<float32_t>());

#pragma omp for schedule(dynamic, chunk_size)
			for (uint32_t v = 0; v < nrowsA; ++v) {

				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v + 1] - ptrA[v] == 1) {
					uint32_t u = colindicesA[ptrA[v]];

					// for each nonzero u-w in B
					float32_t val = valuesA[ptrA[v]];
					uint32_t j = ptrC[v];
					/*
					for (uint32_t w_pos=ptrBH[u]; w_pos<ptrBH[u+1]; ++w_pos) {
						uint32_t wH = (uint32_t)colindicesBH[w_pos];
						for (uint32_t i=valuesBH[w_pos]; i<valuesBH[w_pos+1]; ++i, ++j) {
							valuesC[j] = val * valuesBL[i];
							colindicesC[j] = (wH << B.num_bits.lower) & colindicesBL[i];
						}
					}
					*/
					for (uint32_t pos_w = ptrBH[u]; pos_w < ptrBH[u + 1]; ++pos_w) {
						uint32_t wH = (uint32_t)colindicesBH[pos_w] << B.num_bits.lower;
						uint32x4_t v;
						uint32x4_t wH4 = vdupq_n_u32(wH);
						float32x4_t reg;
						uint32_t i = valuesBH[pos_w], end = valuesBH[pos_w + 1];
						for (; (end - i) >> 2; i += 4, j += 4) {
							reg = vld1q_f32(valuesBL + i);
							reg = vmulq_n_f32(reg, val);
							vst1q_f32(valuesC + j, reg);

							v = vmovl_u16(vld1_u16((uint16_t*)colindicesBL + i));

							v = vandq_u32(v, wH4);
							vst1q_u32(colindicesC + j, v);
						}

						for (; i < end; ++i, ++j) {
							colindicesC[j] = wH & colindicesBL[i];
							valuesC[j] = val * valuesBL[i];
						}
					}
					continue;
				}
				else if (ptrA[v + 1] - ptrA[v] == 0) {
					continue;
				}
				// if degree of v in A is more than one
				// gather all the position info
				uint32_t u_start = ptrA[v];
				uint32_t u_end = ptrA[v + 1];
				for (uint32_t u_pos = u_start; u_pos < u_end; ++u_pos) {
					uint32_t u = colindicesA[u_pos];
					float32_t val = valuesA[u_pos];
					const uint32_t wH_start = ptrBH[u];
					const uint32_t wH_end = ptrBH[u + 1];
					for (uint32_t wH_pos = wH_start; wH_pos < wH_end; ++wH_pos) {
						HOrdType wH = colindicesBH[wH_pos];
						hashmapH.insert(wH, wH_pos, val);
					}
				}

				uint32_t v_sizeC = ptrC[v];
				uint32_t keynum = hashmapH.getKeysNum();
				for (uint32_t k = 0; k < keynum; ++k) {
					HOrdType nonzero;
					float32_t      val;
					hashmapH.getKey(k, nonzero, val);
					uint32_t next;
					uint32_t pos;
					bool moreValue = hashmapH.getFirstValue(nonzero, pos, next);
					uint32_t nonzero_wH = (uint32_t)nonzero << B.num_bits.lower;
					uint32x4_t wH4 = vdupq_n_u32(nonzero_wH);
					if (!moreValue) {
						/*
						uint32_t start = valuesBH[pos];
						uint32_t end = valuesBH[pos+1];

						for (uint32_t w_pos=start; w_pos<end; ++w_pos, ++v_sizeC) {
							valuesC[v_sizeC] = val * valuesBL[w_pos];
							colindicesC[v_sizeC] = nonzero_wH & colindicesBL[w_pos];
						}
						*/
						uint32x4_t v;
						float32x4_t reg;
						uint32_t i = valuesBH[pos], end = valuesBH[pos + 1];
						for (; (end - i) >> 2; i += 4, v_sizeC += 4) {
							reg = vld1q_f32(valuesBL + i);
							reg = vmulq_n_f32(reg, val);
							vst1q_f32(valuesC + v_sizeC, reg);

							v = vmovl_u16(vld1_u16((uint16_t*)colindicesBL + i));

							v = vandq_u32(v, wH4);
							vst1q_u32(colindicesC + v_sizeC, v);
						}

						for (; i < end; ++i, ++v_sizeC) {
							valuesC[v_sizeC] = val * valuesBL[i];
							colindicesC[v_sizeC] = nonzero_wH & colindicesBL[i];
						}

					}
					else {
						do {

							uint32_t k = valuesBH[pos];
							uint32_t end = valuesBH[pos + 1];
							/*
							for (; k<end; ++k) {
								hashmapL.insertInc(colindicesBL[k], val*valuesBL[k]);
							}
							*/
							float32x4_t reg;
							for (; (end - k) >> 2; ++k) {
								reg = vld1q_f32(valuesBL + k);
								reg = vmulq_n_f32(reg, val);
								hashmapL.insertInc(colindicesBL[k], reg[0]);
								hashmapL.insertInc(colindicesBL[++k], reg[1]);
								hashmapL.insertInc(colindicesBL[++k], reg[2]);
								hashmapL.insertInc(colindicesBL[++k], reg[3]);
							}
							for (; k < end; ++k) {
								hashmapL.insertInc(colindicesBL[k], val * valuesBL[k]);
							}

						} while (hashmapH.getCollisions(pos, next, val));

						// read from hashmapL and write to C
						uint32x4_t nonzero_wLx4;
						float32x4_t valuecx4;
						while (hashmapL.getKeyValuex4(nonzero_wLx4, valuecx4)) {
							vst1q_u32(colindicesC + v_sizeC, vandq_u32(nonzero_wLx4, wH4));
							vst1q_f32(valuesC + v_sizeC, valuecx4);
							v_sizeC += 4;
						}

						uint16_t nonzero_wL;
						float32_t valuec;
						while (hashmapL.getKeyValue(nonzero_wL, valuec)) {
							colindicesC[v_sizeC] = nonzero_wH & nonzero_wL;
							valuesC[v_sizeC++] = valuec;
						}
						hashmapL.resetSize();
					}
				}
				hashmapH.resetSize();
				assert(v_sizeC == ptrC[v + 1]);
			}
		}
	}
}

template<typename CSROrdinal, typename Value, typename HOrdType, typename LOrdType>
void _2LN(Matrix<CSROrdinal, CSROrdinal, Value>& A,
	Matrix<CSROrdinal, CSROrdinal, Value>& B_org,
	TwoLevelMatrix<CSROrdinal, Value, HOrdType, LOrdType>& B,
	Matrix<CSROrdinal, CSROrdinal, Value>& C,
	UpperBounds<CSROrdinal, Value>& upb,
	CSROrdinal chunk_size, u32_f32_0_u32) {

	CSROrdinal nrowsA = A.nrows();
	CSROrdinal* ptrA = A.Ptr();
	CSROrdinal* colindicesA = A.ColIndices();
	Value* valuesA = A.Values();

	CSROrdinal nrowsB = B_org.nrows();
	CSROrdinal ncolsB = B_org.ncols();
	CSROrdinal* ptrB = B_org.Ptr();
	CSROrdinal* colindicesB = B_org.ColIndices();
	Value* valuesB = B_org.Values();

	CSROrdinal* ptrC = C.Ptr();
	CSROrdinal* colindicesC = C.ColIndices();
	Value* valuesC = C.Values();

	CSROrdinal widthH = ((long)1 << B.num_bits.higher);
	CSROrdinal widthL = ((long)1 << B.num_bits.lower);

	// 如果一个CACHE能装得下
	if (nrowsA < L2_CACHE_SIZE / sizeof(Value)) {
#pragma omp parallel
		{
			DenseHashMap<CSROrdinal, Value, Value, CSROrdinal> hashmap(ncolsB, upb.max_width, inf<Value>());
#pragma omp for schedule(dynamic, chunk_size)
			for (CSROrdinal v = 0; v < nrowsA; ++v) {
				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v + 1] - ptrA[v] == 1) {
					CSROrdinal u = colindicesA[ptrA[v]];
					Value valvu = valuesA[ptrA[v]];
					CSROrdinal pos_vw = ptrC[v];

					/*
					for (CSROrdinal pos_w=ptrB[u]; pos_w<ptrB[u+1]; pos_w++, pos_vw++) {
						colindicesC[pos_vw]=colindicesB[pos_w];
						valuesC[pos_vw]=valuesB[pos_w]*valvu;
					}
					*/
					float32x4_t reg;
					uint32x4_t v;
					CSROrdinal pos_w = ptrB[u];
					for (; (ptrB[u + 1] - pos_w) >> 2; pos_w += 4, pos_vw += 4) {
						reg = vld1q_f32(valuesB + pos_w);
						reg = vmulq_n_f32(reg, valvu);
						vst1q_f32(valuesC + pos_vw, reg);

						v = vld1q_u32(colindicesB + pos_w);
						vst1q_u32(colindicesC + pos_vw, v);
					}

					for (; pos_w < ptrB[u + 1]; pos_w++, pos_vw++) {
						colindicesC[pos_vw] = colindicesB[pos_w];
						valuesC[pos_vw] = valuesB[pos_w] * valvu;
					}
					continue;

				}
				else if (ptrA[v + 1] - ptrA[v] == 0)
					continue;
				// if degree of v in A is more than one
				/*
				for (CSROrdinal u_pos=ptrA[v]; u_pos<ptrA[v+1]; ++u_pos) {
					CSROrdinal u=colindicesA[u_pos];
					Value valvu=valuesA[u_pos];
					CSROrdinal start=ptrB[u];
					CSROrdinal end=ptrB[u+1];
					for (CSROrdinal w_pos=start; w_pos<end; ++w_pos) {
						CSROrdinal w=colindicesB[w_pos];
						Value valuw = valuesB[w_pos];
						hashmap.insertInc(w, valvu*valuw);
					}
				}*/

				for (CSROrdinal u_pos = ptrA[v]; u_pos < ptrA[v + 1]; ++u_pos) {
					CSROrdinal u = colindicesA[u_pos];
					Value valvu = valuesA[u_pos];
					float32x4_t reg;
					CSROrdinal pos_w = ptrB[u];
					for (; (ptrB[u + 1] - pos_w) >> 2; ++pos_w) {
						reg = vld1q_f32(valuesB + pos_w);
						reg = vmulq_n_f32(reg, valvu);
						hashmap.insertInc(colindicesB[pos_w], reg[0]);
						hashmap.insertInc(colindicesB[++pos_w], reg[1]);
						hashmap.insertInc(colindicesB[++pos_w], reg[2]);
						hashmap.insertInc(colindicesB[++pos_w], reg[3]);
					}

					for (; pos_w < ptrB[u + 1]; pos_w++) {
						hashmap.insertInc(colindicesB[pos_w], valvu * valuesB[pos_w]);
					}
				}

				CSROrdinal nonzerovw;
				Value valuevw;
				CSROrdinal posvw = ptrC[v];
				while (hashmap.getKeyValue(nonzerovw, valuevw)) {
					colindicesC[posvw] = nonzerovw;
					valuesC[posvw++] = valuevw;
				}
				hashmap.resetSize();
			}
		}
	}
	// 如果一个CACHE装不下
	else {
		CSROrdinal* ptrBH = B.H.Ptr();
		HOrdType* colindicesBH = B.H.ColIndices();
		CSROrdinal* valuesBH = B.H.Values();
		Value* valuesBL = B.values;
		LOrdType* colindicesBL = B.L;

#pragma omp parallel
		{
			SparseHashMap<CSROrdinal, Value, true, HOrdType> hashmapH(widthH, upb);
			DenseHashMap<CSROrdinal, Value, Value, LOrdType> hashmapL(widthL, widthL, inf<Value>());

#pragma omp for schedule(dynamic, chunk_size)
			for (CSROrdinal v = 0; v < nrowsA; ++v) {

				// if degree of v in A is less than one, then we can just add the degree
				if (ptrA[v + 1] - ptrA[v] == 1) {
					CSROrdinal u = colindicesA[ptrA[v]];

					// for each nonzero u-w in B
					Value val = valuesA[ptrA[v]];
					CSROrdinal j = ptrC[v];
					/*
					for (CSROrdinal w_pos=ptrBH[u]; w_pos<ptrBH[u+1]; ++w_pos) {
						CSROrdinal wH = (CSROrdinal)colindicesBH[w_pos];
						for (CSROrdinal i=valuesBH[w_pos]; i<valuesBH[w_pos+1]; ++i, ++j) {
							valuesC[j] = val * valuesBL[i];
							colindicesC[j] = (wH << B.num_bits.lower) & colindicesBL[i];
						}
					}
					*/
					for (CSROrdinal pos_w = ptrBH[u]; pos_w < ptrBH[u + 1]; ++pos_w) {
						CSROrdinal wH = (CSROrdinal)colindicesBH[pos_w] << B.num_bits.lower;
						uint32x4_t v;
						uint32x4_t wH4 = vdupq_n_u32(wH);
						float32x4_t reg;
						CSROrdinal i = valuesBH[pos_w], end = valuesBH[pos_w + 1];
						for (; (end - i) >> 2; i += 4, j += 4) {
							reg = vld1q_f32(valuesBL + i);
							reg = vmulq_n_f32(reg, val);
							vst1q_f32(valuesC + j, reg);

							v = vld1q_u32((uint32_t*)colindicesBL + i);

							v = vandq_u32(v, wH4);
							vst1q_u32(colindicesC + j, v);
						}

						for (; i < end; ++i, ++j) {
							colindicesC[j] = wH & colindicesBL[i];
							valuesC[j] = val * valuesBL[i];
						}
					}
					continue;
				}
				else if (ptrA[v + 1] - ptrA[v] == 0) {
					continue;
				}
				// if degree of v in A is more than one
				// gather all the position info
				CSROrdinal u_start = ptrA[v];
				CSROrdinal u_end = ptrA[v + 1];
				for (CSROrdinal u_pos = u_start; u_pos < u_end; ++u_pos) {
					CSROrdinal u = colindicesA[u_pos];
					Value val = valuesA[u_pos];
					const CSROrdinal wH_start = ptrBH[u];
					const CSROrdinal wH_end = ptrBH[u + 1];
					for (CSROrdinal wH_pos = wH_start; wH_pos < wH_end; ++wH_pos) {
						HOrdType wH = colindicesBH[wH_pos];
						hashmapH.insert(wH, wH_pos, val);
					}
				}

				CSROrdinal v_sizeC = ptrC[v];
				CSROrdinal keynum = hashmapH.getKeysNum();
				for (CSROrdinal k = 0; k < keynum; ++k) {
					HOrdType nonzero;
					Value      val;
					hashmapH.getKey(k, nonzero, val);
					CSROrdinal next;
					CSROrdinal pos;
					bool moreValue = hashmapH.getFirstValue(nonzero, pos, next);
					CSROrdinal nonzero_wH = (CSROrdinal)nonzero << B.num_bits.lower;
					if (!moreValue) {
						/*
						CSROrdinal start = valuesBH[pos];
						CSROrdinal end = valuesBH[pos+1];

						for (CSROrdinal w_pos=start; w_pos<end; ++w_pos, ++v_sizeC) {
							valuesC[v_sizeC] = val * valuesBL[w_pos];
							colindicesC[v_sizeC] = nonzero_wH & colindicesBL[w_pos];
						}
						*/
						uint32x4_t v;
						uint32x4_t wH4 = vdupq_n_u32(nonzero_wH);
						float32x4_t reg;
						CSROrdinal i = valuesBH[pos], end = valuesBH[pos + 1];
						for (; (end - i) >> 2; i += 4, v_sizeC += 4) {
							reg = vld1q_f32(valuesBL + i);
							reg = vmulq_n_f32(reg, val);
							vst1q_f32(valuesC + v_sizeC, reg);

							v = vld1q_u32((uint32_t*)colindicesBL + i);

							v = vandq_u32(v, wH4);
							vst1q_u32(colindicesC + v_sizeC, v);
						}

						for (; i < end; ++i, ++v_sizeC) {
							valuesC[v_sizeC] = val * valuesBL[i];
							colindicesC[v_sizeC] = nonzero_wH & colindicesBL[i];
						}

					}
					else {
						do {

							CSROrdinal k = valuesBH[pos];
							CSROrdinal end = valuesBH[pos + 1];
							/*
							for (; k<end; ++k) {
								hashmapL.insertInc(colindicesBL[k], val*valuesBL[k]);
							}
							*/
							float32x4_t reg;
							for (; (end - k) >> 2; ++k) {
								reg = vld1q_f32(valuesBL + k);
								reg = vmulq_n_f32(reg, val);
								hashmapL.insertInc(colindicesBL[k], reg[0]);
								hashmapL.insertInc(colindicesBL[++k], reg[1]);
								hashmapL.insertInc(colindicesBL[++k], reg[2]);
								hashmapL.insertInc(colindicesBL[++k], reg[3]);
							}
							for (; k < end; ++k) {
								hashmapL.insertInc(colindicesBL[k], val * valuesBL[k]);
							}

						} while (hashmapH.getCollisions(pos, next, val));

						// read from hashmapL and write to C
						LOrdType nonzero_wL;
						Value valuec;
						while (hashmapL.getKeyValue(nonzero_wL, valuec)) {
							colindicesC[v_sizeC] = nonzero_wH & nonzero_wL;
							valuesC[v_sizeC++] = valuec;
						}
						hashmapL.resetSize();
					}
				}
				hashmapH.resetSize();
				assert(v_sizeC == ptrC[v + 1]);
			}
		}
	}
}

template <typename CSROrdinal, typename Value, typename HOrdType, typename LOrdType>
void twoLNumeric(Matrix<CSROrdinal, CSROrdinal, Value>& A,
	Matrix<CSROrdinal, CSROrdinal, Value>& B_org,
	TwoLevelMatrix<CSROrdinal, Value, HOrdType, LOrdType>& B,
	Matrix<CSROrdinal, CSROrdinal, Value>& C,
	UpperBounds<CSROrdinal, Value>& upb,
	CSROrdinal chunk_size) {
	_2LN(A, B_org, B, C, upb, chunk_size, 
		typename TagDispatch<CSROrdinal, Value, HOrdType, LOrdType>::Tag{});
	return;
}
