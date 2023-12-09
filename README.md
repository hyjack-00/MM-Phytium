# MM-Phytium

Matrix-Multiplication-Library-on-Phytium

# 工具

## 编译运行脚本

如果按照 huangyj 的 cmake 项目结构，编译的命令如下，具体的编译选项见 `CMakeLists.txt`

```shell
cd build
cmake ..
make
cd ..
```

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

# 稠密矩阵乘 s32

## 不同 microkernels 的性能

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

## 不同矩阵大小下的性能
best: `4x8k8_ldB_fchC_pkAB` ：

|           | 256^3    | 512^3    | 768^3    | 1024^3   | 1536^3   | 2048^3   | 4096^3   |
| --------  | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| time/ms   | 4.48872  | 35.7291  | 117.866  | 280.976  | 938.576  | 2308.86  | 20336    |
| GOPS      | 3.74     | 3.76     | 3.84     | 7.14     | 3.86     | 3.72     | 3.38     |
| effic     | 40.89%   |          |          | 77.65%   |          |          |          |

waiting for parallelization 


## 优化分块大小
暂时只测试了 512^3, 1024^3, 1536^3 的矩阵乘法

- `Ti, Tj` 的最优取值范围较广，对于 N^3 类型矩阵乘法基本在 N/2 附近的值都有较高性能
- `Tk` 应当取 256



# 稀疏矩阵乘

## 性能分析

Perf 的结果

```
Samples: 11K of event 'cycles:ppp', Event count (approx.): 26682106052
Overhead  Command      Shared Object Symbol
-----------------------------------------------
28.76%  sparce_test  sparce_test          [.] std::__push_heap<__gnu_cxx::__normal_iterator<Cord<int>**, ...
22.50%  sparce_test  sparce_test          [.] std::__adjust_heap<__gnu_cxx::__normal_iterator<Cord<int>**, ...
21.44%  sparce_test  sparce_test          [.] dcgemm<int>
10.53%  sparce_test  libc-2.23.so         [.] malloc
-----------------------------------------------
1.34%  sparce_test  sparce_test           [.] dc_sparce_matrix<int>::dc_sparce_matrix
1.11%  sparce_test  [kernel.kallsyms]     [k] clear_page                             
0.98%  sparce_test  libc-2.23.so          [.] 0x000000000007021c                     
0.64%  sparce_test  libc-2.23.so          [.] 0x000000000007019c                     
0.53%  sparce_test  libc-2.23.so          [.] 0x0000000000070200                     
0.51%  sparce_test  sparce_test           [.] main                                   
0.42%  sparce_test  libc-2.23.so          [.] 0x0000000000070380                     
0.38%  sparce_test  libc-2.23.so          [.] 0x0000000000070974                     
0.35%  sparce_test  libstdc++.so.6.0.21   [.] operator new                           
0.32%  sparce_test  libstdc++.so.6.0.21   [.] _ZNSt6locale21_S_normalize_categoryEi@plt
0.30%  sparce_test  libc-2.23.so          [.] 0x0000000000070230
0.28%  sparce_test  libc-2.23.so          [.] 0x0000000000070b00
....
```

# 想法

## 稠密矩阵乘

- [x] outer_kernel 的 ijk 顺序变成 ? 怎么样
    - `4x8k8_ldB_fchC_pkAB` 无区别
- [ ] micro_kernel 的 ijk 顺序变成 ikj 怎么样，packing 需要随之改变
- [x] 与 fetch 相对的 append：初始化 C 寄存器时直接全 0 ，最后再 取出相加存入
    - 没有提升
- [ ] 为什么 1024 大小如此特殊？


# 结果

libshalom fp32

```
Size: i16 j16 k16
    total avg time: 0.00211111 msecs
    3.88042GFLOPS, 22.0478% peak
Size: i32 j32 k32
    total avg time: 0.00666667 msecs
    9.8304GFLOPS, 55.8545% peak
Size: i64 j64 k64
    total avg time: 0.0431111 msecs
    12.1613GFLOPS, 69.0984% peak
Size: i128 j128 k128
    total avg time: 0.260556 msecs
    16.0975GFLOPS, 91.4633% peak
Size: i256 j256 k256
    total avg time: 1.99433 msecs
    16.8249GFLOPS, 95.5959% peak
Size: i512 j512 k512
    total avg time: 15.7112 msecs
    17.0856GFLOPS, 97.0772% peak
Size: i1024 j1024 k1024
    total avg time: 131.822 msecs
    16.2908GFLOPS, 92.5613% peak
Size: i2048 j2048 k2048
    total avg time: 1094.78 msecs
    15.6925GFLOPS, 89.1622% peak
Size: i4096 j4096 k4096
    total avg time: 9532.35 msecs
    14.4182GFLOPS, 81.9214% peak
```
