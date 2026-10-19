# 最小 AllReduce Ring/Simple：复用原 runRing 和 primitives

设备调用链的 CUDA 内建操作、PTX 指令及适配边界见
[设备指令清单](device-instructions.md)。

其他集合通信与 Send/Recv 的同标准最小案例见
[collectives_simple](../../examples/collectives_simple/README.md)，包括 Broadcast、Reduce、
AllGather、ReduceScatter、SendRecv，以及由 Send/Recv 组合的 AlltoAll、Gather、Scatter。

这个样例只有一个 `.cu`，Host 按 kernel 名直接启动，设备端实际复用
`src/device/all_reduce.h` 的 `runRing` 以及 `src/device/prims_simple.h`，
不再实现自己的 Ring 循环、send/recv、握手或归约原语。

## 调用链

```text
Host 手工建立连接，并填写 ncclDevWorkColl（buffer / count / chunk）
    |
    v
allreduce_ring_simple<<<1, 128, scratchBytes, stream>>>(...)
    |
    v
runRing<float, FuncSum<float>, ProtoSimple<2, 2>>()
    |
    v
Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, ProtoSimple<2, 2>, 0>
    |
    +-- directSend
    +-- directRecvReduceDirectSend
    +-- directRecvReduceCopyDirectSend
    +-- directRecvCopyDirectSend
    +-- directRecv
    |
    v
原 waitPeer / reduceCopy / postPeer / ~Primitives
```

直接包含并调用 `src/device/all_reduce.h::runRing`，该函数本体没有修改。
固定 float32 + Sum、单进程、1～8 个 GPU、一个 channel、一个线程块。
初始 chunk 大小为 1024 个 float；末轮重新分配 chunk 和对齐由原 `runRing` 完成。
Host 不做拓扑选择。一个 channel 对应 `channelLo=channelHi=0`，所有元素填入
`work.cbd.countLo`；`chunkGrainsLo` 按 Simple 的 512 字节粒度填写为 8。
该描述不包含函数编号，也不经过任务队列或分发表。
单 rank 在相同入口中直接复制。

## 原 Simple 流程如何复用

每条 ring 边分配原 `NCCL_STEPS=8` 的 Simple FIFO；每个 step 为 1024 个 float。
接收方的 `recv[0]` 和发送方的 `send[0]` 指向同一条边的数据及 head/tail。
`conn.stepSize` 和 `comm.buffSizes[NCCL_PROTO_SIMPLE]` 按实际缓冲区初始化。

连接 flags 为 0，没有注册用户缓冲区；上述 direct* 方法进入原有 FIFO fallback。
收发角色划分、额外 post warp、barrier、等待 credit/data、slice/step 更新、
向量复制和归约、内存 fence 都使用原 NCCL 实现。
没有修改 `prims_simple.h`、`common_kernel.h`、`reduce_kernel.h` 或 `op128.h`。

Host 在首次启动前完成所有 GPU 的连接初始化和输入拷贝，再启动所有 GPU，
最后等待结果。FIFO 与连接状态只初始化一次，后续调用复用它们；
原析构函数负责把 step 写回 `ncclConnInfo`。

## 必要的头文件支持

样例在包含头文件前定义 `NCCL_PRIMITIVES_STANDALONE`，在四个头文件中添加了局部开关：

| 文件 | 独立样例模式下的处理 |
| --- | --- |
| `src/device/all_reduce.h` | 保留原 runRing 模板，排除 Tree 辅助函数及通用 RunWorkColl 特化 |
| `src/device/common.h` | 在本编译单元定义 shared 状态；排除函数表和通用批次/kernel 分发 |
| `src/device/primitives.h` | 保留原 ProtoSimple/Fan/Primitives，排除 LL/LL128 定义及实现头 |
| `src/include/device.h` | 排除 Host kernel/func 映射表声明和 funcId 查表辅助函数 |

普通 NCCL 构建不定义该宏，保留原路径。没有新增 NCCL 构建 profile，
不编译完整 libnccl，不调用 device Python 生成器，不产生通用 kernel 或函数表。
Make 只用 sed 从 `nccl.h.in` 生成带版本号的公共头文件，所有 CUDA 实现仍来自一个 `.cu`。

复用原 primitives 会引入其必要的 NCCL 头文件依赖。真实依赖写在构建目录的
`allreduce.d`，预处理结果写在 `allreduce.ii`（执行 inspect 后生成）。
后续指令适配应沿 `all_reduce.h::runRing -> prims_simple.h -> common_kernel.h/reduce_kernel.h/op128.h`
及 `common.h` 的同步操作展开，直接验证原 NCCL 数据路径。
网络 unpack 的辅助头是原 primitives 的静态依赖；样例连接不启用该功能。

## 构建和运行

从 NCCL 仓库根目录执行，按实际 GPU 调整 `ARCH`：

```bash
make -C examples/allreduce_ring_simple build inspect ARCH=sm_80
make -C examples/allreduce_ring_simple run RANKS=2 COUNT=4097
make -C examples/allreduce_ring_simple test RANKS=2
```

输出默认位于 `build-allreduce-ring-simple/`。编译是 `nvcc -c -rdc=false`，
随后使用 `g++ + libcudart` 进行普通 Host 链接，没有 device link 或 libnccl 依赖。
可以通过 `CUDA_HOME`、`NVCC`、`CXX`、`NVCCFLAGS` 和 `BUILD` 调整工具链和输出位置。
`NVCCFLAGS` 需保留原复制/归约辅助函数需要的 `--extended-lambda`。

多 GPU 要求每条 ring 边支持 CUDA P2P；没有加入 RDMA/Socket 传输。
运行目标默认有 120 秒超时；原 primitives 仍保留 abortFlag 检查，样例把它初始化为 0。

## 测试

`--self-test` 对指定 rank 数执行 36 项用例：
9 种长度（0、1、17、1023、1024、1025、4097、65539、1048576），
每种分别测试原地/非原地，并重复两次。

- 每个 rank 的结果与 CPU 逐元素求和比较，输入覆盖正数、负数和零。
- 输出尾部有 16 个 float 的哨兵检查越界写入。
- 连续调用不清零 FIFO/连接 step，验证原析构函数保存的状态能够复用。
- 每次检查 FIFO head/tail 和本 rank 的 send/recv conn.step 是否达到预期值。

令本次 tile 数为 `ceil(count / (nranks * 1024))`，多 rank 每条连接的 step 增量为：

```text
本次 tile 数 * 2 * (nranks - 1) * ALLREDUCE_CHUNKSTEPS
```

有条件时分别执行 `RANKS=1/2/3/4`。
`inspect` 无需 GPU：验证实际包含原 all_reduce/prims_simple、保留 runRing 且排除 Tree 和 RunWorkColl，
没有 LL/LL128 实现头，
预处理源码没有原函数表和通用 kernel 分发，对象没有 RDC 注册符号且只有一个 kernel 入口。

本机通过 CUDA 13.0 / sm_80 的编译、普通 Host 链接和 inspect 检查。
PTX 行信息用于确认实际使用了原 all_reduce/prims_simple 等源码。
未定义独立样例宏时的原头文件编译回归也已通过。
本机无可用 GPU，1/2 rank 的通信用例返回 77（SKIP）。
随后在远端 H20 上完成 sm_90 的 1、2、3、4 rank 验证，连同其他操作共通过 2016 个用例，
并修复了超过默认共享内存限额的启动问题，详见 [H20 验证记录](validation-h20.md)。
