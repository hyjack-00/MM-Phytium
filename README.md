# MM-Phytium
Matrix-Multiplication-Library-on-Phytium

# 硬件峰值

用乘 / 加操作计数：
```
fp32-kernel paral
perf: 73.154764 GFLOPS
fp32-kernel single
perf: 18.389867 GFLOPS

fp64-kernel paral
perf: 36.670165 GFLOPS
fp64-kernel single
perf: 9.192812 GFLOPS

int32-kernel paral
perf: 36.729423 GIPS
int32-kernel single
perf: 9.194918 GIPS
```

`ni=N, nj=M, nk=P` 的矩阵乘法，需要约 $ 2 \cdot N \cdot M \cdot P $ 条乘 / 加计算指令。

以 `int32` 下的 1024-1024-1024 矩阵大小为例，共需要 2G 条计算指令，极限用时 0.05 s 左右。

# 结果记录 

## 稠密矩阵乘 s32

### 不同 microkernels 的性能

- 时间单位 msec ，1024^3 的极限大约 54.45 msecs
- `i/4` 表示对 i 分块进行简单并行

| microkernel | 512^3  | 1024^3 | 2048^3 |
| ----------- | ------ | ------ | ------ |
| `4x4k4_ldB_fchC`          | 60.991    | 508.109   | 4838.53   |
| `4x4k4_ldA_fchC`          | 61.4193   | 512.649   | 4866.18   |
| `4x4k4_ldB_fchC_pkAB`     |           | 284.694   |           |
||||
| `4x8k8_ldB_fchC`          | 44.4709   | 371.4     |           |
| `4x8k8_ldB_fchC` i/4      |           | 128.873   |           |
| `4x8k8_ldA_fchC`          |           | 406.423   |           |
| `4x8k8_ldB_fchC_pkAB`     | 35.7291   | 280.976   | 2308.86   |
| `4x8k8_ldB_fchC_pkABC`    | 35.7907   | 284.464   | 2331.31   |
| `4x8k8_ldB_apdC_pkAB`     |           | 281.065   |           |
||||
| `8x4k8_ldB_fchC`          |           | 548.868   |           |
| `8x4k8_ldA_fchC`          |           | 544.788   |           |
||||
| `8x8k4_ldB_fchC`          |           | 452.039   |           |
| `8x8k4_ldA_fchC`          |           | 400.938   |           |

似乎 280 ms 已经是极限了

### 不同矩阵大小下的性能
best: `4x8k8_ldB_fchC_pkAB` ：

|           | 256^3    | 512^3    | 768^3    | 1024^3   | 1536^3   | 2048^3   | 4096^3   |
| --------  | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| time/ms   | 4.48872  | 35.7291  | 117.866  | 280.976  | 938.576  | 2308.86  | 20336    |
| GOPS      | 3.74     | 3.76     | 3.84     | 7.14     | 3.86     | 3.72     | 3.38     |
| effic     | 40.89%   |          |          | 77.65%   |          |          |          |

waiting for parallelization 


### 优化分块大小
暂时只测试了 512^3, 1024^3, 1536^3 的矩阵乘法

- `Ti, Tj` 的最优取值范围较广，对于 N^3 类型矩阵乘法基本在 N/2 附近的值都有较高性能
- `Tk` 应当取 256



# 想法

## 稠密矩阵乘

- [x] outer_kernel 的 ijk 顺序变成 ? 怎么样
    - `4x8k8_ldB_fchC_pkAB` 无区别
- [ ] micro_kernel 的 ijk 顺序变成 ikj 怎么样，packing 需要随之改变
- [x] 与 fetch 相对的 append：初始化 C 寄存器时直接全 0 ，最后再 取出相加存入
    - 没有提升
- [ ] 为什么 1024 大小如此特殊？
