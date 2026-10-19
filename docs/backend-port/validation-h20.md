# H20 远端验证记录

2026-10-19，在 `h20_219` 的 `/home/user/liuxianyun/code/nccl` 完成验证。
代码提交为 `492ae5b3be1865e12e999a7abb51be2476d15ea8`，分支为 `device_backend`。

## 环境

- GPU：NVIDIA H20，计算能力 9.0，GPU 间 NVLink 连接。
- 使用 `CUDA_VISIBLE_DEVICES=0,1,2,3`，验证 1、2、3、4 rank。
- CUDA Toolkit：13.0，nvcc V13.0.88；驱动：580.95.05。
- 构建：`ARCH=sm_90`，单 `.cu` / 单 kernel，`-rdc=false`，普通 Host 链接。

## 结果

9 类操作、36 个进程级自测，共通过 2016 个用例，无 FAIL 或 SKIP。
另有各操作的 2 rank / count=17 冒烟测试通过。

| 操作 | 1 rank | 2 rank | 3 rank | 4 rank | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AllReduce | 36 | 36 | 36 | 36 | 144 |
| Broadcast | 36 | 72 | 108 | 144 | 360 |
| Reduce | 36 | 72 | 108 | 144 | 360 |
| AllGather | 36 | 36 | 36 | 36 | 144 |
| ReduceScatter | 36 | 36 | 36 | 36 | 144 |
| SendRecv | 18 | 18 | 18 | 18 | 72 |
| AlltoAll | 18 | 18 | 18 | 18 | 72 |
| Gather | 36 | 72 | 108 | 144 | 360 |
| Scatter | 36 | 72 | 108 | 144 | 360 |

覆盖 count 为 `0, 1, 17, 1023, 1024, 1025, 4097, 65539, 1048576`，
每种重复两次；支持原地布局的操作同时测试原地/非原地，带 root 的操作遍历所有 root。
检查 CPU 参考结果、缓冲区哨兵以及 FIFO head/tail 和连接 step 的连续复用。

全部样例也通过 sm_90 构建、Host 链接和 inspect：每个对象只有一个具名 kernel，
没有 LL/LL128 实现头、通用函数分发表或 RDC 注册符号。

## 验证中修复的启动问题

首次 sm_90 运行返回 `cudaGetLastError(): invalid argument`。
此架构下 `NCCL_MAX_ARITY` 增大，原 NCCL shared context 加动态 scratch 后超过默认 48 KiB：

| Kernel | 静态 shared | 动态 scratch | 合计 |
| --- | ---: | ---: | ---: |
| AllReduce | 37952 B | 16448 B | 54400 B |
| SendRecv | 37872 B | 32896 B | 70768 B |

修复在 Host 初始化中查询 `cudaFuncGetAttributes`；当静态 shared 加 scratch
超过设备默认 block shared 限额时，通过具名 kernel 的
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, scratchBytes)`
申请所需额度。没有改变原 runRing、Primitives、FIFO 或归约流程。
上述完整矩阵在修复后执行。

## 复现与日志

在远端仓库根目录执行：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
make -C examples/collectives_simple all inspect-all -j2 ARCH=sm_90
make -C examples/allreduce_ring_simple build inspect ARCH=sm_90
make -C examples/collectives_simple test-matrix ARCH=sm_90 RANK_MATRIX='1 2 3 4'
for ranks in 1 2 3 4; do
  make -C examples/allreduce_ring_simple test ARCH=sm_90 RANKS="$ranks"
done
```

本次逐进程运行使用 120 秒超时，详细日志保存在远端：

```text
/home/user/liuxianyun/code/nccl/build-validation-h20/collectives-build.log
/home/user/liuxianyun/code/nccl/build-validation-h20/allreduce-build.log
/home/user/liuxianyun/code/nccl/build-validation-h20/<op>-ranks<1|2|3|4>.log
```

验证范围为单机、单进程、单 channel、float32/Sum 和普通 P2P FIFO。
没有验证 5～8 rank、跨节点传输、注册缓冲区、其他数据类型或其他归约操作。
