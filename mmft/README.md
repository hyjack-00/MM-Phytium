
# Library Structure

```
├── src
│   ├── common
│   │   └── Type.h
│   ├── matrix
│   │   ├── DenseMat.cpp
│   │   ├── DenseMat.h
│   │   ├── Mat.h
│   │   ├── SparseMat.h
│   │   ├── SparseMat_COO.cpp
│   │   ├── SparseMat_CSR.cpp
│   │   └── SparseMat_DCSR.cpp
│   ├── gemm
│   │   ├── GEMM.cpp
│   │   ├── GEMM.h
│   │   └── kernels
│   │       ├── gemm_fp32.cpp
│   │       ├── gemm_fp64.cpp
│   │       └── gemm_int32.cpp
│   └── spgemm
│       ├── SpGEMM.h
│       ├── SpGEMM_CSR.cpp
│       ├── SpGEMM_DCSR.cpp
│       └── kernels
│           ├── spgemm_csr_fp32.cpp
│           └── spgemm_csr_fp64.cpp
├── test
│   ├── dense_test.cpp
│   └── sparse_test.cpp
├── README.md
└── updates.md
```

不单独隔离 include 了

# API

- GEMM.h
