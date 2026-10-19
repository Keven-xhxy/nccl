# 最小 Simple 集合通信与 Send/Recv 案例

可选 `MINIMAL=1` 会在全部案例中编译排除网络 unpack 实现和 NVLS/multimem，
默认 `MINIMAL=0` 保留原路径。两模式在 H20 上各通过 2016 个用例，详见
[第一阶段说明](../../docs/backend-port/phase1-minimal.md)。

```bash
make -C examples/collectives_simple all inspect-all MINIMAL=1 ARCH=sm_90 -j2
make -C examples/collectives_simple test-matrix MINIMAL=1 ARCH=sm_90
```

与 [AllReduce 样例](../allreduce_ring_simple/allreduce.cu) 使用同一标准：
一个 `.cu`、按操作单独编译、Host 直接调用具名 kernel、`-rdc=false`，
普通 `g++ + libcudart` 链接。没有 libnccl 依赖、device Python 生成器、
funcId、函数表或通用 work/batch 分发。

全部案例共用 `collectives.cu` 和 Makefile。每个对象只选择一个操作，
只包含对应的算法头和一个 kernel 入口；没有 device 函数指针调用。
数据固定为 float32，归约固定为 Sum。send/recv 按原 NCCL 实现使用 int8 字节传输，
不会对传输的 float 做数值转换。

## 案例与原实现的关系

| OP | Kernel 名 | 复用入口 | count 含义 |
| --- | --- | --- | --- |
| `broadcast` | `broadcast_ring_simple` | `broadcast.h::runRing` | 广播元素数 |
| `reduce` | `reduce_ring_simple` | `reduce.h::runRing` | 每个 rank 的输入元素数 |
| `allgather` | `allgather_ring_simple` | `all_gather.h::runRing` | 每个 rank 的输入元素数；输出为 ranks × count |
| `reducescatter` | `reducescatter_ring_simple` | `reduce_scatter.h::runRing` | 每个 rank 的输出元素数；输入为 ranks × count |
| `sendrecv` | `sendrecv_simple` | `sendrecv.h::runSend/runRecv` | 每个 rank 发往 next、从 prev 接收的元素数 |
| `alltoall` | `alltoall_simple` | 同上，Host 按轮次配对 | 发往每个 peer 的元素数；输入、输出均为 ranks × count |
| `gather` | `gather_simple` | 同上，Host 逐 peer 配对 root | 每个 rank 的输入元素数；root 输出为 ranks × count |
| `scatter` | `scatter_simple` | 同上，Host 逐 peer 配对 root | 每个 rank 的输出元素数；root 输入为 ranks × count |

四个 `runRing` 本体保持原样，在独立样例模式下排除后续 RunWorkColl 特化。
原 sendrecv 批次类中的 runSend/runRecv 提取为文件内模板函数，原批次路径也调用
这两个函数；样例直接调用它们，跳过批次调度。两者保留原 Primitives 的 P2p=1、
连接索引 1、`ProtoSimple<1,1>` 和 do/while 零长度握手。

AlltoAll/Gather/Scatter 与此版本的 `src/collectives.cc` 一样由 send/recv 组合，
样例只提供显式的最小轮次调度，不复用完整 Host enqueue。
AlltoAll 第 d 轮：rank r 发送到 `(r+d)%n`，从 `(r-d+n)%n` 接收。
Gather/Scatter 第 p 轮：只连接 peer p 与 root。自传输由同一 kernel 做本地复制。
内部 `AllGatherV` 批次算法没有加入这些等长公共 API 案例。

## 构建、检查与运行

从 NCCL 仓库根目录执行：

```bash
# 只编译一个操作，修改 OP 即可切换。
make -C examples/collectives_simple build inspect OP=allgather ARCH=sm_80

# 编译并检查所有八个独立产物。
make -C examples/collectives_simple all inspect-all -j2 ARCH=sm_80

make -C examples/collectives_simple run OP=reduce RANKS=3 ROOT=2 COUNT=4097
make -C examples/collectives_simple run OP=sendrecv RANKS=2 COUNT=17
make -C examples/collectives_simple run OP=allgather RANKS=3 COUNT=1025 ARGS=--in-place

make -C examples/collectives_simple test OP=scatter RANKS=3
make -C examples/collectives_simple test-all RANKS=2
make -C examples/collectives_simple test-matrix RANK_MATRIX='1 2 3 4'
```

输出在 `build-collectives-simple/`；例如可直接运行
`build-collectives-simple/sendrecv --ranks 2 --self-test`。
支持 `CUDA_HOME/NVCC/CXX/NVCCFLAGS/BUILD`，编译选项需保留 `--extended-lambda`。
更换 ARCH 或 NVCCFLAGS 后使用新 BUILD 目录或 clean 后重建。

`inspect` 对选定操作验证：原算法/Simple 头确实参与编译；没有 LL/LL128 实现头、
函数表、RunWorkColl/RunWorkBatch 分发、RDC 注册符号；对象恰好只有一个具名 kernel。
`inspect-all` 对八个产物分别检查，不需要 GPU。

`run/test` 默认使用 120 秒超时。需要 1～8 个 CUDA GPU；Ring 和 sendrecv 需要
每条 r→next 边支持 P2P；AlltoAll/Gather/Scatter 的最小环境统一检查并启用全 peer P2P。
无 GPU 或不满足拓扑要求时，程序返回 77（SKIP）。Make 会将非零状态报告为失败，
日志中的 `SKIP` 表示环境不满足，不能视为通过通信测试。

## 测试覆盖

每个 `--self-test` 使用 9 种 count：
`0, 1, 17, 1023, 1024, 1025, 4097, 65539, 1048576`，每种重复两轮。

- Broadcast/Reduce/Gather/Scatter 遍历所有 root，并覆盖原地/非原地。
- AllGather/ReduceScatter 覆盖原地/非原地。
- SendRecv/AlltoAll 使用独立输入、输出；本案例不支持它们的原地模式。
- 使用包含正、负、零的整数 float 输入，与 CPU 参考逐元素比较。
- 同时检查输入和输出；两端各有 16 个 float 哨兵，未写入区域也必须保留原值。
- 所有调用复用原连接，不清空 FIFO 或 conn.step；检查每条有向边的 head/tail
  及每个 peer 的 recv/send conn.step，包含本轮没有参与的边。
- 零长度集合通信不传数据；零长度远端 send/recv 仍执行原 do/while 握手并推进一次 step。

原地布局按操作分别设置：AllGather 的 send 位于 output 的本 rank 段；
ReduceScatter 的 recv 位于 input 的本 rank 段；Gather/Scatter 只在 root 做对应布局；
Broadcast/Reduce 的 send 与 recv 使用同一地址。
单 rank 使用相同具名 kernel 进行本地复制。

## Device 指令差异

共同基础见 [AllReduce 指令清单](../../docs/backend-port/device-instructions.md)。
新增操作仍沿 `Primitives → genericOp → waitPeer/reduceCopy/postPeer` 运行。

| 案例 | 相比 AllReduce 的主要变化 |
| --- | --- |
| Broadcast、AllGather | 使用复制，无数值归约 |
| Reduce、ReduceScatter | 仍是 `add.f32`，每个元素只沿归约阶段传播 |
| Broadcast、Reduce | `ProtoSimple<1,1>`，每 chunk 推进一个 step |
| AllGather、ReduceScatter | `ProtoSimple<2,2>`，每个 primitive 推进四个 step |
| SendRecv 及其组合 | `ProtoSimple<1,1>`，byte pack；需要支持 `ld.volatile.global.b8` / `st.global.b8` 的非对齐或尾部路径 |

P2P kernel 使用 256 线程：send 子组 128 线程，包含 96 个 worker，使用 barrier 15/14；
recv 子组 128 线程，使用 barrier 13。两个子组独立运行，不能用全 block barrier
替代原语内的分组 barrier。只有入口初始化使用全 block 同步。
Host 会查询 kernel 的静态 shared 用量；静态 shared 与动态 scratch 合计超过设备默认
block 限额时，按 kernel 名申请更高的动态共享内存额度，以支持 sm_90 等架构。

默认模式与已有 AllReduce 一样，网络 unpack、注册缓冲区分支未启用，但部分分支仍会生成 PTX。
`MINIMAL=1` 会排除 unpack 实现与 multimem；注册缓冲区等其他分支暂时保留。
编译器适配范围应检查完整对象；不能把运行时关闭等同于已从编译中裁掉。

## 本机验证状态

CUDA 13.0 / sm_80 下，八个独立程序均通过编译、普通 Host 链接及静态 inspect。
未定义独立样例宏时，四个原集合通信入口和 sendrecv 批次入口也通过编译回归。
本机没有可用 CUDA GPU，通信运行返回 SKIP。
远端 H20 / CUDA 13.0 / sm_90 已完成 1、2、3、4 rank 的全部案例验证；
连同 AllReduce 共通过 2016 个用例，详见 [H20 验证记录](../../docs/backend-port/validation-h20.md)。
