# 第一阶段：集合通信与 SendRecv 的局部裁剪

2026-10-19 完成。使用一个开关 `MINIMAL=1`，由两个样例 Makefile 统一传入
`NCCL_MINIMAL_RING_SIMPLE=1`。默认 `MINIMAL=0` 保留原路径。

支持 AllReduce、Broadcast、Reduce、AllGather、ReduceScatter，以及
SendRecv、由 SendRecv 组合的 AlltoAll/Gather/Scatter。
集合通信使用 float32/Sum/Ring/Simple；P2P 使用原 int8 helper 按字节搬运 float32
payload，不做精度转换和数值归约。继续采用单 `.cu`、具名 kernel、无函数表、无 RDC。

## 修改边界

| 文件 | 本阶段修改 |
| --- | --- |
| `prims_simple.h` | 局部排除 unpack 实现 include、setup/执行/进度/析构调用和 peer ballot；排除 NVLS multimem polling |
| `op128.h` | 最小模式不展开 multimem store 特化，保留现有 generic 模板入口 |
| `reduce_kernel.h` | 最小模式不展开 multimem load/reduce 特化 |
| `device.h` | 检查最小模式只能与独立样例模式一起使用 |
| 两个样例 Makefile | 统一开关、独立输出目录、配置变化触发重建、最小模式 inspect 检查 |

没有修改 runRing、原 slice/chunk/step 算法、FIFO 等待或发布步骤；没有复制 Simple 实现，
没有引入功能策略框架。核心源码修改为四个头文件的局部条件编译与约束检查。

构造阶段原来的 `barrierAny(NetDeviceUnpack)` 在最小模式中替换为 `barrier()`，
保留共享连接和数据指针初始化后的发布同步，不能直接删除这次同步。

模板实例检查限制归约为 float/Sum，允许 P2p=1 的单字节传输；
拒绝 multimem 输入/输出和 net offload。该开关针对样例手工建立的普通 FIFO 连接，
不支持用它驱动网络插件或 NVLS 连接。

## 仍然保留的内容

- `unpack_defs.h` 的纯数据结构与 shared 字段保留，因此相关布局没有重构；
  被排除的是包含设备实现的 `unpack.h`。
- half/BF16/FP8、其他算子、coop/atomic、PAT 及其他未实例化 helper 的定义仍可能被前端看到。
- 注册缓冲区、connFifo、DirectRead/Write 的已有分支暂未全面裁剪；样例将它们关闭。
- 仍保留设备断言 `__assertfail`、原 128/96 线程同步和必要的 volatile/system fence 语义。

这一步缩小生成代码与部分解析范围，尚未实现目标平台的 ISA 兼容后端，也没有把前端依赖
完全缩减到 float Sum。后续针对实际编译阻塞再局部处理。

## 构建与测试

在 NCCL 根目录执行：

```bash
make -C examples/allreduce_ring_simple build inspect MINIMAL=1 ARCH=sm_90
make -C examples/collectives_simple all inspect-all MINIMAL=1 ARCH=sm_90 -j2

make -C examples/allreduce_ring_simple test-matrix MINIMAL=1 ARCH=sm_90
make -C examples/collectives_simple test-matrix MINIMAL=1 ARCH=sm_90

# 默认模式回归：将上述 MINIMAL=1 改为 MINIMAL=0。
```

最小模式默认输出为 `build-allreduce-ring-simple-minimal/`、`build-collectives-simple-minimal/`，
默认模式仍使用原输出目录。显式复用 BUILD 时，配置文件会检测 MINIMAL、ARCH、NVCC/NVCCFLAGS
的变化并重建，避免拿错模式或架构的旧对象。

`inspect` 会额外检查实际依赖没有 `network/unpack/unpack.h`，预处理结果没有
ncclNetDeviceUnpack/IncrementHead/SaveHead 调用，生成 PTX 没有 unpack 实现或 `multimem.` 指令。

## 编译依赖对比

CUDA 13.0，以同一源码分别编译默认/最小模式；以下为 AllReduce 审计数据。
静态指令条数不代表动态执行次数或性能提升。

| 指标 | sm_80 默认 | sm_80 最小 | sm_90 默认 | sm_90 最小 |
| --- | ---: | ---: | ---: | ---: |
| PTX 助记符种类 | 135 | 108 | 136 | 108 |
| 静态 PTX 指令条数 | 8924 | 5611 | 8988 | 5603 |
| 宏展开后保留的 inline asm 块数 | 75 | 74 | 108 | 74 |
| 内建/构造/atomic 相关引用名称数 | 57 | 56 | 62 | 61 |

除 multimem 外，也消除了 unpack 引入的 b8/b16/b64 pack 读写、shared 元数据搬运、
ffs 展开、peer ballot 等生成代码。`__any_sync` 等名字还可能留在未实例化 helper 中，
最终 PTX 不再生成对应投票；不能把生成指令的减少等同于全部定义被移除。

完整四配置数据见 [summary.json](device-inventory/phase1/summary.json)，
逐条指令及模块作用见 [PTX 对照表](device-inventory/phase1/ptx-explained.md)。
各配置子目录保留 device-references、inline-asm、ptx-opcodes、special-registers TSV。

## 验证记录

- 本地全部 9 类操作通过最小模式 sm_80、sm_90 编译、普通链接和 inspect。
- 默认非 standalone 的 Broadcast/Reduce/AllGather/ReduceScatter/SendRecv 入口编译回归通过。
- double、Prod、multimem 的最小模式实例编译拒绝检查通过；非 standalone 使用该开关同样被拒绝。
- 同一个 BUILD 按 MINIMAL=1→0→1 切换，确认对象正确重建，unpack 依赖随模式切换。
- H20 / CUDA 13.0 / sm_90，使用 GPU 0～3，最小模式和默认模式各执行
  9 类操作 × 1/2/3/4 rank，共 2016 个用例，两模式总计 **4032 个用例全部通过**，没有 FAIL/SKIP。
  覆盖零长度、各个 root、支持的原地布局、尾部哨兵和连续 FIFO/conn.step 状态。

远端代码位于 `/home/user/liuxianyun/code/nccl`，本阶段以工作区补丁验证。
构建日志为 `build-validation-h20/phase1/build-mode{0,1}.log`，
测试日志为 `build-validation-h20/phase1/mode{0,1}-<op>-ranks<1|2|3|4>.log`。
