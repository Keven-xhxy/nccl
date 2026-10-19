# 最终 PTX 全部助记符、模块与作用

由 inventory-device.py 从完整 PTX 生成；次数是代码中的静态指令条数，不是执行次数。
模块由 .file/.loc/inlined_at 映射；完整行号见各架构 ptx-opcodes.tsv。

| 指令 | sm_80 | sm_90 | 来源模块 | 作用 |
| --- | ---: | ---: | --- | --- |
| `add.f32` | 159 | 159 | `src/device/reduce_kernel.h` | FuncSum<float> 的逐元素求和 |
| `add.s32` | 313 | 313 | `CUDA_SDK/sm_20_intrinsics.hpp`, `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h`, `src/device/primitives.h`, `src/device/prims_simple.h`, `src/include/bitops.h`, `src/include/device.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `add.s64` | 2270 | 2270 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h`, `src/device/prims_simple.h`, `src/include/bitops.h`, `src/include/device.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `and.b16` | 28 | 28 | `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `and.b32` | 309 | 321 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h`, `src/device/primitives.h`, `src/device/prims_simple.h`, `src/include/bitops.h`, `src/include/device.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `and.b64` | 65 | 65 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h`, `src/include/bitops.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `and.pred` | 13 | 13 | `src/device/prims_simple.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `bar.sync` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu` | kernel 初始化后的全 block 同步 |
| `bar.warp.sync` | 33 | 33 | `CUDA_SDK/sm_30_intrinsics.hpp`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 32 线程原语备用路径及 unpack warp 同步 |
| `barrier.red.or.pred` | 1 | 1 | `src/device/common.h`, `src/device/prims_simple.h` | 原语构造时的 block OR 投票并同步 |
| `barrier.sync.aligned` | 21 | 21 | `src/device/common.h`, `src/device/prims_simple.h` | 按 barrier ID 和参与线程数同步 worker/原语全组 |
| `bfind.shiftamt.u32` | 4 | 4 | `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 实现 unpack 的 __ffs，定位第一个 peer 位 |
| `bra` | 596 | 608 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/primitives.h`, `src/device/prims_simple.h`, `src/include/bitops.h`, `src/include/device.h` | 条件/一致性分支、轮询及循环控制 |
| `bra.uni` | 200 | 224 | `CUDA_SDK/sm_30_intrinsics.hpp`, `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h`, `src/device/prims_simple.h`, `src/include/bitops.h`, `src/include/device.h` | 条件/一致性分支、轮询及循环控制 |
| `brev.b32` | 4 | 4 | `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 实现 unpack 的 __ffs，定位第一个 peer 位 |
| `call.uni` | 1 | 1 | `src/device/all_reduce.h`, `src/device/prims_simple.h` | 调用设备断言 __assertfail；正常测例不进入 |
| `cvt.s64.s32` | 91 | 91 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h`, `src/include/device.h` | 整数位宽与有符号性转换，用于索引、地址及元数据 |
| `cvt.u16.u32` | 12 | 12 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/network/unpack/unpack.h` | 整数位宽与有符号性转换，用于索引、地址及元数据 |
| `cvt.u16.u64` | 4 | 4 | `src/device/network/unpack/unpack.h` | 整数位宽与有符号性转换，用于索引、地址及元数据 |
| `cvt.u32.u16` | 16 | 16 | `examples/allreduce_ring_simple/allreduce.cu` | 整数位宽与有符号性转换，用于索引、地址及元数据 |
| `cvt.u32.u64` | 17 | 17 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h`, `src/include/bitops.h` | 整数位宽与有符号性转换，用于索引、地址及元数据 |
| `cvt.u64.u32` | 65 | 65 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h`, `src/include/bitops.h` | 整数位宽与有符号性转换，用于索引、地址及元数据 |
| `cvta.global.u64` | 3 | 3 | `src/device/all_reduce.h`, `src/device/prims_simple.h` | generic 与 global/shared 地址空间转换 |
| `cvta.shared.u64` | 2 | 2 | `src/device/all_reduce.h`, `src/device/prims_simple.h` | generic 与 global/shared 地址空间转换 |
| `cvta.to.global.u64` | 105 | 105 | `CUDA_SDK/sm_20_intrinsics.hpp`, `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/op128.h`, `src/device/prims_simple.h` | generic 与 global/shared 地址空间转换 |
| `cvta.to.shared.u64` | 8 | 8 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | generic 与 global/shared 地址空间转换 |
| `div.s32` | 4 | 4 | `src/device/network/unpack/unpack.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `div.s64` | 1 | 1 | `src/include/bitops.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `div.u32` | 1 | 1 | `src/include/bitops.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `fence.acq_rel.sys` | 8 | 8 | `src/device/op128.h`, `src/device/prims_simple.h` | 发送数据后、发布进度前的 system scope 内存顺序 |
| `ld.global.f32` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.param.u32` | 2 | 2 | `examples/allreduce_ring_simple/allreduce.cu` | 读取 kernel 参数 |
| `ld.param.u64` | 20 | 20 | `examples/allreduce_ring_simple/allreduce.cu` | 读取 kernel 参数 |
| `ld.param.u8` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu` | 读取 kernel 参数 |
| `ld.param.v2.u32` | 4 | 4 | `examples/allreduce_ring_simple/allreduce.cu` | 读取 kernel 参数 |
| `ld.param.v4.u32` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu` | 读取 kernel 参数 |
| `ld.param.v4.u8` | 6 | 6 | `examples/allreduce_ring_simple/allreduce.cu` | 读取 kernel 参数 |
| `ld.relaxed.gpu.u64` | 4 | 4 | `src/device/network/unpack/unpack.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.shared.s32` | 2 | 2 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h` | 读取共享上下文、src/dst 指针或 unpack 元数据 |
| `ld.shared.u32` | 20 | 20 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/prims_simple.h`, `src/include/device.h` | 读取共享上下文、src/dst 指针或 unpack 元数据 |
| `ld.shared.u64` | 186 | 170 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/primitives.h`, `src/device/prims_simple.h` | 读取共享上下文、src/dst 指针或 unpack 元数据 |
| `ld.shared.u8` | 2 | 2 | `src/device/prims_simple.h` | 读取共享上下文、src/dst 指针或 unpack 元数据 |
| `ld.shared.v2.u32` | 1 | 1 | `src/device/all_reduce.h`, `src/device/prims_simple.h` | 读取共享上下文、src/dst 指针或 unpack 元数据 |
| `ld.shared.v2.u64` | 6 | 14 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/include/device.h` | 读取共享上下文、src/dst 指针或 unpack 元数据 |
| `ld.u32` | 20 | 20 | `src/device/prims_simple.h` | 通过 generic 指针读取连接、进度、abort 或字节数据 |
| `ld.u64` | 23 | 23 | `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 通过 generic 指针读取连接、进度、abort 或字节数据 |
| `ld.u8` | 1 | 1 | `src/device/prims_simple.h` | 通过 generic 指针读取连接、进度、abort 或字节数据 |
| `ld.volatile.global.b16` | 224 | 224 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.volatile.global.b32` | 299 | 299 | `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.volatile.global.b64` | 56 | 56 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.volatile.global.b8` | 64 | 64 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.volatile.global.u32` | 10 | 10 | `src/device/common_kernel.h`, `src/device/prims_simple.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.volatile.global.u64` | 12 | 12 | `src/device/op128.h`, `src/device/prims_simple.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.volatile.global.v2.b64` | 127 | 127 | `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.volatile.global.v2.u64` | 4 | 4 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义 |
| `ld.volatile.shared.v2.u64` | 4 | 4 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 读取共享上下文、src/dst 指针或 unpack 元数据 |
| `ld.volatile.u32` | 15 | 15 | `src/device/primitives.h`, `src/device/prims_simple.h` | 通过 generic 指针读取连接、进度、abort 或字节数据 |
| `ld.volatile.u64` | 5 | 5 | `src/device/all_reduce.h`, `src/device/prims_simple.h` | 通过 generic 指针读取连接、进度、abort 或字节数据 |
| `ld.volatile.u8` | 4 | 4 | `src/device/network/unpack/unpack.h` | 通过 generic 指针读取连接、进度、abort 或字节数据 |
| `max.s32` | 5 | 5 | `src/device/prims_simple.h` | chunk/slice/尾部或 unpack 长度边界 |
| `min.s32` | 9 | 9 | `src/device/prims_simple.h` | chunk/slice/尾部或 unpack 长度边界 |
| `min.s64` | 5 | 5 | `CUDA_SDK/math_functions.hpp`, `src/device/all_reduce.h` | chunk/slice/尾部或 unpack 长度边界 |
| `min.u64` | 4 | 4 | `src/device/network/unpack/unpack.h` | chunk/slice/尾部或 unpack 长度边界 |
| `mov.b32` | 477 | 477 | `src/device/common_kernel.h`, `src/device/op128.h`, `src/device/reduce_kernel.h` | 寄存器赋值、bit pack 拆装或读取线程特殊寄存器 |
| `mov.b64` | 167 | 167 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h`, `src/device/reduce_kernel.h` | 寄存器赋值、bit pack 拆装或读取线程特殊寄存器 |
| `mov.u16` | 8 | 8 | `src/device/common_kernel.h`, `src/device/prims_simple.h` | 寄存器赋值、bit pack 拆装或读取线程特殊寄存器 |
| `mov.u32` | 262 | 262 | `CUDA_SDK/sm_30_intrinsics.hpp`, `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/primitives.h`, `src/device/prims_simple.h` | 寄存器赋值、bit pack 拆装或读取线程特殊寄存器 |
| `mov.u64` | 118 | 118 | `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h`, `src/device/prims_simple.h`, `src/include/device.h` | 寄存器赋值、bit pack 拆装或读取线程特殊寄存器 |
| `mul.lo.s32` | 17 | 17 | `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `mul.lo.s64` | 36 | 36 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/prims_simple.h`, `src/include/device.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `mul.wide.s32` | 23 | 23 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `multimem.ld_reduce.acquire.sys.global.min.u64` | 0 | 12 | `src/device/prims_simple.h` | sm_90 的 NVLS 最小进度轮询分支；本测例标志关闭但代码仍生成 |
| `neg.s16` | 4 | 4 | `src/device/network/unpack/unpack.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `not.b32` | 2 | 2 | `src/device/all_reduce.h`, `src/include/device.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `not.pred` | 9 | 9 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/prims_simple.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `or.b16` | 4 | 4 | `src/device/network/unpack/unpack.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `or.b32` | 27 | 27 | `src/device/all_reduce.h`, `src/device/primitives.h`, `src/device/prims_simple.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `or.b64` | 5 | 5 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/prims_simple.h`, `src/include/bitops.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `or.pred` | 21 | 21 | `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `prmt.b32` | 32 | 32 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/network/unpack/unpack.h` | kernel 结构体参数及 unpack 字节重排 |
| `ret` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu` | 退出 kernel |
| `selp.b32` | 52 | 52 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/prims_simple.h` | 按谓词选择值或转换投票结果 |
| `selp.s32` | 1 | 1 | `src/device/common.h`, `src/device/prims_simple.h` | 按谓词选择值或转换投票结果 |
| `selp.u16` | 15 | 15 | `src/device/common_kernel.h` | 按谓词选择值或转换投票结果 |
| `selp.u32` | 11 | 11 | `src/device/all_reduce.h`, `src/device/common.h`, `src/device/common_kernel.h`, `src/device/prims_simple.h` | 按谓词选择值或转换投票结果 |
| `setp.eq.s16` | 17 | 17 | `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.eq.s32` | 205 | 217 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/primitives.h`, `src/device/prims_simple.h`, `src/include/device.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.eq.s64` | 93 | 93 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h`, `src/include/bitops.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.ge.s32` | 11 | 11 | `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.ge.s64` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.ge.u32` | 4 | 4 | `src/device/network/unpack/unpack.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.ge.u64` | 30 | 30 | `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.gt.s32` | 78 | 78 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.gt.s64` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.gt.u64` | 8 | 8 | `src/device/network/unpack/unpack.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.le.s32` | 1 | 1 | `src/device/all_reduce.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.le.u64` | 9 | 9 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/network/unpack/unpack.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.lt.s32` | 80 | 80 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/primitives.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.lt.s64` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.lt.u32` | 10 | 10 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.lt.u64` | 37 | 37 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/network/unpack/unpack.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.ne.s16` | 10 | 10 | `CUDA_SDK/sm_30_intrinsics.hpp`, `src/device/common_kernel.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.ne.s32` | 62 | 62 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common.h`, `src/device/network/unpack/unpack.h`, `src/device/primitives.h`, `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `setp.ne.s64` | 2 | 2 | `src/device/prims_simple.h` | 产生循环、分支、角色、对齐和投票条件的谓词 |
| `shl.b32` | 99 | 99 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `shl.b64` | 50 | 50 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `shr.s32` | 110 | 110 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `shr.u32` | 76 | 76 | `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h`, `src/include/device.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `shr.u64` | 18 | 18 | `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h`, `src/include/device.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
| `st.global.b16` | 224 | 224 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 写输出/FIFO；relaxed.sys.u64 用于发布进度 |
| `st.global.b32` | 282 | 282 | `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 写输出/FIFO；relaxed.sys.u64 用于发布进度 |
| `st.global.b64` | 56 | 56 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 写输出/FIFO；relaxed.sys.u64 用于发布进度 |
| `st.global.b8` | 64 | 64 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 写输出/FIFO；relaxed.sys.u64 用于发布进度 |
| `st.global.f32` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu` | 写输出/FIFO；relaxed.sys.u64 用于发布进度 |
| `st.global.v2.b64` | 118 | 118 | `src/device/common_kernel.h`, `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 写输出/FIFO；relaxed.sys.u64 用于发布进度 |
| `st.param.b32` | 1 | 1 | `src/device/all_reduce.h`, `src/device/prims_simple.h` | 填写设备断言函数的调用参数 |
| `st.param.b64` | 4 | 4 | `src/device/all_reduce.h`, `src/device/prims_simple.h` | 填写设备断言函数的调用参数 |
| `st.relaxed.sys.global.u64` | 10 | 10 | `src/device/op128.h`, `src/device/prims_simple.h` | 写输出/FIFO；relaxed.sys.u64 用于发布进度 |
| `st.shared.u32` | 18 | 18 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/primitives.h`, `src/device/prims_simple.h` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.shared.u64` | 88 | 88 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.shared.u8` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.shared.v2.u32` | 1 | 1 | `examples/allreduce_ring_simple/allreduce.cu` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.shared.v2.u64` | 7 | 7 | `examples/allreduce_ring_simple/allreduce.cu` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.shared.v4.u16` | 2 | 2 | `examples/allreduce_ring_simple/allreduce.cu` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.shared.v4.u32` | 3 | 3 | `examples/allreduce_ring_simple/allreduce.cu` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.shared.v4.u8` | 2 | 2 | `examples/allreduce_ring_simple/allreduce.cu` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.u64` | 12 | 12 | `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h`, `src/device/prims_simple.h` | 通过 generic 指针保存连接 step、指针交换或 unpack 字节 |
| `st.volatile.shared.v2.u64` | 4 | 4 | `src/device/network/unpack/unpack.h`, `src/device/op128.h` | 发布共享上下文、src/dst 指针或 unpack 元数据 |
| `st.volatile.u64` | 4 | 4 | `src/device/prims_simple.h` | 通过 generic 指针保存连接 step、指针交换或 unpack 字节 |
| `st.volatile.u8` | 4 | 4 | `src/device/network/unpack/unpack.h` | 通过 generic 指针保存连接 step、指针交换或 unpack 字节 |
| `sub.s32` | 182 | 182 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/common_kernel.h`, `src/device/prims_simple.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `sub.s64` | 17 | 17 | `examples/allreduce_ring_simple/allreduce.cu`, `src/device/all_reduce.h`, `src/device/network/unpack/unpack.h` | chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算 |
| `vote.sync.all.pred` | 8 | 8 | `CUDA_SDK/sm_30_intrinsics.hpp`, `src/device/common_kernel.h` | reduceCopy 的 warp 全体对齐判断 |
| `vote.sync.any.pred` | 1 | 1 | `CUDA_SDK/sm_30_intrinsics.hpp`, `src/device/prims_simple.h` | 原语的单 warp any 投票备用分支 |
| `vote.sync.ballot.b32` | 1 | 1 | `CUDA_SDK/sm_30_intrinsics.hpp`, `src/device/prims_simple.h` | 建立 unpack peer 位掩码 |
| `xor.b64` | 2 | 2 | `src/device/prims_simple.h`, `src/include/bitops.h` | 标志/谓词、对齐、取模、数据 pack 与地址计算 |
