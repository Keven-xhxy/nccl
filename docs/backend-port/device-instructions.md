# 最小 AllReduce Ring/Simple 的设备指令清单

范围：`examples/allreduce_ring_simple/allreduce.cu`，float32 / Sum，
一个 block、128 个线程、单 channel、普通 P2P FIFO。
依据当前源码与 CUDA 13.0、`ARCH=sm_80` 生成的 PTX 9.0。
这里列的是 CUDA 内建操作和 PTX；SASS 是 NVIDIA 硬件指令，不能与 PTX 一一等同。
以下是静态分析，不是设备执行 trace。
功能正确性另有 [H20 远端验证记录](validation-h20.md)。

其他集合通信与 Send/Recv 的案例及指令差异见
[collectives_simple](../../examples/collectives_simple/README.md#device-指令差异)。
特别是 P2P 的字节尾部路径需要 b8 load/store，不能直接沿用本文对 float AllReduce 的裁剪范围。

## 1. 调用链与指令来源

```text
allreduce_ring_simple                         examples/.../allreduce.cu
  ├─ 单 rank：普通 float copy
  └─ 多 rank：初始化 shared context + __syncthreads
      └─ runRing<float, FuncSum<float>, ProtoSimple<2,2>>  all_reduce.h
          └─ Primitives 构造                              prims_simple.h
          └─ directSend / directRecvReduceDirectSend /
             directRecvReduceCopyDirectSend /
             directRecvCopyDirectSend / directRecv
              └─ genericOp
                  ├─ waitPeer → loadStepValue → ld_volatile_global
                  ├─ subBarrier
                  ├─ reduceCopy → reduceCopyPacks         common_kernel.h
                  │   ├─ 地址转换、pack load/store        op128.h
                  │   └─ Apply_Reduce<FuncSum<float>>      reduce_kernel.h
                  ├─ barrier
                  └─ postPeer → fence + step store
          └─ ~Primitives：保存 conn.step + barrier
```

`runRing` 主要提供分块、rank 索引和调用次序，核心 CUDA 特有操作集中在
`prims_simple.h`、`common.h`、`common_kernel.h`、`op128.h`。
`reduce_kernel.h` 的当前 float Sum 最终是普通浮点加法。

## 2. 当前多 rank 路径需要的操作

| 操作 | 源码入口 | 当前 PTX / CUDA 表达 | 适配要点 |
| --- | --- | --- | --- |
| 线程编号与线程数 | kernel、Primitives 构造 | `threadIdx.x` / `blockDim.x`，PTX 特殊寄存器 `%tid.x` / `%ntid.x` | 原实现以 32 线程为一个 warp 做分工与索引 |
| 共享状态 | kernel、Primitives | `__shared__`，`ld.shared.*` / `st.shared.*` | comm/channel、srcs/dsts、abort 状态在线程间共享 |
| 初始化同步 | kernel | `__syncthreads()` → `bar.sync 0` | 全 block 初始化完成后才使用共享状态 |
| 原语内同步 | `common.h::barrier_sync` | `barrier.sync.aligned name, count` | group=0 时全组 barrier 15、128 线程；worker barrier 14、96 线程 |
| block 投票同步 | `common.h::barrier_red_or`，`Primitives::barrierAny` | `setp.ne.s32` + `barrier.red.or.pred` + `selp.s32` | 构造时判断是否有网络 unpack；即使结果为 false，这次全组投票仍会执行 |
| warp 对齐判断 | `common_kernel.h::reduceCopy` | `__all_sync(~0u, aligned)` → `vote.sync.all.pred` | 全 warp 一致选择 16 字节 pack 路径或 4 字节路径 |
| 地址空间转换 | `op128.h::cvta_to_global` 等 | `__cvta_generic_to_global/shared`；`cvta.to.global.u64`、`cvta.to.shared.u64`、`cvta.global.u64`、`cvta.shared.u64` | 区分 generic / global / shared 地址；不是数据复制 |
| 轮询 FIFO 进度 | `loadStepValue` → `op128.h::ld_volatile_global` | `ld.volatile.global.u64` | 接收方等待 tail；发送方等待 head 对应的可用 credit，循环中必须能观察到对端更新 |
| 读取数据 | `reduceCopyPacks` → `op128.h` | `ld.volatile.global.v2.b64`、`ld.volatile.global.b32` | 对齐时每线程一次处理 16 字节，否则使用 float 大小的 pack；包括尾部处理 |
| 写入数据 | 同上 | `st.global.v2.b64`、`st.global.b32` | 写入 peer FIFO 或本地 output；16 字节版本要求对应对齐 |
| float 求和 | `reduce_kernel.h::Apply_Reduce<FuncSum<float>,1>` | C++ `a+b` → `add.f32` | pack 内逐 float 求和；不是 `atomicAdd` |
| 跨设备内存顺序 | `postPeer` → `op128.h::fence_acq_rel_sys` | `fence.acq_rel.sys` | 有发送数据时，先完成数据写入的必要顺序，再发布进度 |
| 发布进度 | `postPeer` → `op128.h::st_relaxed_sys_global` | `st.relaxed.sys.global.u64` | 64 位 head/tail 更新，system scope；结合前面的同步和 fence 理解 |
| abort 与持久状态 | `primitives.h::checkAbort`、析构 | 普通/volatile load、shared store、`conn->step` store | 长时间轮询时检查 abort，析构保存下一次调用需要的 step |
| 索引、分支与参数传递 | 各层、编译器生成 | `ld.param.*`、`mov`、`cvt`、`add/sub`、`mul`、`div`、`min/max`、位运算、`setp`、`selp`、`bra`、`ret` 等 | 需要 32/64 位整数、指针运算与谓词控制流；不是新增通信原语 |

128 个线程中，96 个 worker 执行复制/归约，另一个 warp 用于配合发送完成发布。
WaitRecv/WaitSend 角色分别落在 tid 0/1，PostRecv/PostSend 分别落在 tid 126/127。
128 与 96 两类 barrier 的参与者不同，不能直接把所有 barrier 替换成全 block 同步。

握手的核心关系是：

```text
发送方：等待 head credit → 写 peer FIFO → 组同步 → system fence → 发布 tail
接收方：轮询 tail → worker 同步 → 读 FIFO 并复制/归约 → 组同步 → 发布 head
```

源码明确要求数据读取继续使用 volatile，避免读取旧的 L1 数据。
迁移时不能仅把 `volatile` 去掉、或把 system fence 换成普通线程同步；
必须由目标后端提供对应的跨设备可见性、进度更新与先后顺序。
当前代码使用的是 volatile 轮询，并没有在这条普通 FIFO 路径使用 acquire load。

## 3. 已生成但样例条件不会进入的分支

单个 `.cu` 和单个 kernel 入口不意味着只生成了表中必需的操作。
conn/work 的配置在运行时读取，编译器未能消去全部条件分支。

| 分支 | 当前 PTX 中的额外操作 | 本样例为何不进入 |
| --- | --- | --- |
| 网络 unpack | `ld.relaxed.gpu.u64`；`ld.volatile.global.b8/b16/b64` 和对应 store；`ld/st.volatile.shared.v2.u64`；`bar.warp.sync`；`brev.b32`、`bfind.shiftamt.u32`；部分 `prmt.b32` | `conn->netDeviceHandle.netDeviceType` 没有设为 UNPACK；`AnyNetDeviceUnpack` 为 false |
| 构造中的 unpack peer mask | `vote.sync.ballot.b32` | 位于 `barrierAny(NetDeviceUnpack)` 为 true 的分支内 |
| 32 线程原语的同步分支 | `bar.warp.sync`、`vote.sync.any.pred` | 本样例 nthreads=128、nworkers=96；两者都不是 32 |
| 注册缓冲区和直接指针交换 | 额外指针/标志轮询及 load/store | work.regUsed/netRegUsed/direct 为 0，连接 flags 为 0；direct* 调用走 FIFO fallback |
| 设备断言 | `call.uni __assertfail` | 正常角色分配满足原构造函数断言；默认编译未定义 NDEBUG，因此对象仍含断言依赖 |

`prmt.b32` 也用于入口参数的字节重排，不能把所有 prmt 都归到网络分支。
当前对象确实保留 `__assertfail` 调用；没有函数表不代表完全没有任何 device call。

后续若目标编译器连这些未执行分支也不能接受，仍需要编译期开关裁掉网络
unpack、注册缓冲区等分支。当前独立样例宏只裁掉 LL/LL128、Tree 和通用分发等部分。

## 4. 当前不需要的指令族

当前 sm_80 PTX 没有 `multimem.*`、`cp.async*`、`mbarrier*`、`mma/wgmma*`、
`shfl*` 或 `atom*`。不需要为这个 float Sum FIFO 测例实现 half/bfloat16/FP8
归约指令，也不使用 LL/LL128 的专用流程。

`loadStepValue` 中确实有 `multimem.ld_reduce.acquire.sys.global.min.u64` 源码，
但受 `__CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010` 和 NVLS 标志约束，
本次 sm_80 构建不包含它。切换架构后要重新检查。

## 5. 建议的后端封装边界

最先集中封装 `op128.h` 中的地址转换、4/16 字节数据 load/store、64 位进度
load/store 和 system fence；再封装 `common.h` 中带 name/count 的同步和 OR 投票，
以及 `reduceCopy` 的 warp all 投票和线程索引入口。
这样可以保留 `runRing` 与 `genericOp` 的算法、slice、FIFO step 流程。

这是语义边界建议，不要求目标 ISA 使用同名指令。向量操作可在满足对齐、可见性和
正确性要求的前提下拆分；system scope 的同步语义和参与线程集合必须保留。
若目标 warp/subgroup 宽度不是 32，还需要调整原实现的 WARP_SIZE、角色划分、
全 mask 投票和 pack 工作分配，不能只替换 asm 字符串。

## 6. 从构建产物复查

在 NCCL 仓库根目录执行：

```bash
make -C examples/allreduce_ring_simple build inspect ARCH=sm_80
/usr/local/cuda/bin/cuobjdump --dump-ptx build-allreduce-ring-simple/allreduce.o > build-allreduce-ring-simple/allreduce.ptx
/usr/local/cuda/bin/cuobjdump --dump-sass build-allreduce-ring-simple/allreduce.o > build-allreduce-ring-simple/allreduce.sass
rg -n 'barrier\.|bar\.|vote\.|fence\.|ld\.volatile|st\.relaxed|add\.f32' build-allreduce-ring-simple/allreduce.ptx
```

Makefile 已启用 `-lineinfo`，PTX 的 `.file`、`.loc` 和 `inlined_at` 可追溯到具体头文件。
检查 PTX 应检查整个 kernel，包括运行时不进入的分支；评估执行路径则需结合上述条件。
工具链、优化选项和架构变化都可能改变具体 PTX/SASS 清单。
