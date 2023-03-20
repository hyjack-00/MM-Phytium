# MM-Phytium
Matrix-Multiplication-Library-on-Phytium

## 硬件峰值

用乘 / 加操作计数：
```
fp32-kernel paral
perf: 73.154764 GFlOPS
fp32-kernel single
perf: 18.389867 GFlOPS

fp64-kernel paral
perf: 36.670165 GFlOPS
fp64-kernel single
perf: 9.192812 GFlOPS

int32-kernel paral
perf: 36.729423 GIPS
int32-kernel single
perf: 9.194918 GIPS
```

以 `int32` 下的 1024-1024-1024 矩阵大小为例，共需要 2G 条乘 / 加操作指令，极限用时 0.05 s 左右。

## 结果记录 

## 稠密矩阵乘 s32

- 时间单位 msec ，1024^3 的极限大约 54.45 msecs
- `i/4` 表示对 i 分块进行简单并行

| microkernel | 512^3  | 1024^3 | 2048^3 |
| ----------- | ------ | ------ | ------ |
| `4x4k4_ldB_fchC`          | 60.991    | 508.109   | 4838.53   |
| `4x4k4_ldA_fchC`          | 61.4193   | 512.649   | 4866.18   |
| `4x4k4_ldB_fchC_pkAB`     | 35.1663   | 284.983   |           |
| `4x8k8_ldB_fchC`          | 44.4709   | 371.4     |           |
| `4x8k8_ldB_fchC` i/4      |           | 128.873   |           |
| `4x8k8_ldA_fchC`          |           | 406.423   |           |
| `4x8k8_ldB_fchC_pkAB`     |           | 279.976   |           |


## 想法

### 稠密矩阵乘

- outer_kernel 的 ijk 顺序变成 ? 怎么样
- micro_kernel 的 ijk 顺序变成 ikj 怎么样，packing 需要随之改变
- 与 fetch 相对的 append：

    初始化 C 寄存器时直接全 0 ，最后再 取出相加存入