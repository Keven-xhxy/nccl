# AllReduce device 端完整 CUDA 编译依赖

下述根目录完整索引为裁剪前的默认模式审计快照，来源行号对应当时源码。
第一阶段已增加面向全部样例的可选 `MINIMAL=1`，
将最终 PTX 收敛到两架构各 108 种助记符，见 [局部裁剪说明与验证](phase1-minimal.md)
及 [默认/最小模式完整 PTX 对照](device-inventory/phase1/ptx-explained.md)。

范围是当前 `examples/allreduce_ring_simple/allreduce.cu`，定义
`NCCL_PRIMITIVES_STANDALONE`，float32 / Sum / Ring / Simple，CUDA 13.0、
`-std=c++17 -O2 -lineinfo --extended-lambda -rdc=false`。
本次重新编译了 **sm_80 与 sm_90**，审计 device 预处理文件 `allreduce.cpp1.ii`
和最终 `allreduce.ptx`，不包含 Host Runtime/Driver API。

这里的“完整”分为两个范围，两者都不能省略：

- **代码生成范围**：该 kernel 的全部 PTX 助记符，包括运行时关闭的分支和编译器生成的指令。
- **前端编译范围**：NCCL 头文件经预处理后仍保留的 CUDA 内建引用、类型、inline asm 和设备模板定义，
  包括最终没有实例化或已被优化掉的部分。CUDA SDK 的内部实现不作为 NCCL 自己调用的指令重复展开。

完整性只针对上述两个编译配置；更换 CUDA 版本、架构特性、宏或优化参数需要重新生成清单。
这是编译产物审计，不以“某个正常测例没执行”作为删除指令的依据，也不是动态执行次数统计。

## 1. 审计结果与完整索引

| 项目 | sm_80 | sm_90 |
| --- | ---: | ---: |
| 最终 PTX 的不同助记符 | 135 | 136 |
| 最终 PTX 的静态指令条数 | 8924 | 8988 |
| NCCL 设备源码中的 inline asm 块，宏展开后计数 | 75 | 108 |
| 内建函数/构造引用及 cuda::atomic_ref/相关枚举名称 | 57 | 62 |
| kernel 入口 | 1 | 1 |
| 设备外部函数 | `__assertfail` | `__assertfail` |

**最终 136 种 PTX 指令逐项的模块、作用和两架构条数见
[全部 PTX 指令解释表](device-inventory/ptx-explained.md)**。该表没有用通配符省略助记符。
以下文件保留完整来源行号及原始 asm 模板：

| 范围 | sm_80 | sm_90 |
| --- | --- | --- |
| 全部生成指令、静态条数、来源行号 | [ptx-opcodes.tsv](device-inventory/sm_80/ptx-opcodes.tsv) | [ptx-opcodes.tsv](device-inventory/sm_90/ptx-opcodes.tsv) |
| 前端保留的设备内建/构造/原子引用 | [device-references.tsv](device-inventory/sm_80/device-references.tsv) | [device-references.tsv](device-inventory/sm_90/device-references.tsv) |
| 全部 inline asm 块，含宏展开结果 | [inline-asm.tsv](device-inventory/sm_80/inline-asm.tsv) | [inline-asm.tsv](device-inventory/sm_90/inline-asm.tsv) |
| 最终使用的特殊寄存器 | [special-registers.tsv](device-inventory/sm_80/special-registers.tsv) | [special-registers.tsv](device-inventory/sm_90/special-registers.tsv) |

[summary.json](device-inventory/summary.json) 记录 PTX 版本、入口、外部函数、数量和输入文件 SHA256。
57/62 是引用名称数，包含 CUDA 类型构造及编译器 builtin，不代表 57/62 条硬件指令。
75/108 是 asm 块数，一个块可以含多条指令，不能与 PTX 指令数相加。

## 2. 模块与职责

```text
allreduce.cu：具名 kernel、线程索引、shared 上下文、初始化同步
  all_reduce.h::runRing：分块、Ring rank 顺序、归约/广播两个阶段
    primitives.h：ProtoSimple、Fan、abort 检查
    prims_simple.h：角色分工、FIFO 等待、组同步、数据操作、进度发布
      common.h：带 ID/线程数的 barrier、block OR 投票、共享状态
      common_kernel.h：reduceCopy、warp 对齐投票、向量/尾部处理
        op128.h：pack load/store、地址转换、volatile/relaxed/acquire/release、fence
        reduce_kernel.h：类型转换、归约算子、pre/post-op、multimem 特化
      network/unpack/unpack.h：网络设备 unpack，仍被包含且生成备用代码
  间接公共头：device.h → nccl_device/core.h → coop.h / utility.h
              collectives.h、bitops.h
```

状态约定：**主路径**是普通 FIFO 流程会用到的操作；**备用生成**表示最终 PTX 中存在、
但本例配置不进入；**定义保留**表示前端仍看到定义，本 kernel 不生成对应指令。
一个接口可能被不同模块用于多种状态。

## 3. CUDA 语法、类型与特殊寄存器

| 编译能力 | 模块 | 作用与状态 |
| --- | --- | --- |
| `__global__`、`__device__`、`__host__ __device__`、`__forceinline__` | kernel 及所有设备头 | 入口、设备函数、共用模板、内联属性；不是 PTX 指令 |
| `__shared__`、`extern __shared__`、`alignas` | kernel、common.h | 静态上下文与动态 per-warp scratch，要求共享地址空间和对齐支持 |
| `threadIdx.x`、`blockDim.x` | kernel、Primitives、coop.h | 线程角色、步长、组大小；本 kernel 最终特殊寄存器只有 `%tid.x`、`%ntid.x` |
| `%laneid`、`%lanemask_lt` 的 inline `mov.u32` | utility.h | coop 的 lane 编号和低位 lane mask；定义保留，最终 PTX 没有读取这两个寄存器 |
| CUDA 向量类型与布局，例如 `ulong2`、`ulonglong2`、`float2`、`float4` | common.h、op128.h、reduce_kernel.h、utility.h | shared scratch、BytePack、向量 load/store 与类型转换；需要 ABI/对齐布局 |
| `half`/`__half`、`half2`/`__half2` | reduce_kernel.h、CUDA fp16 头 | FP16 cast、Sum/Prod/MinMax、pre/post-op 的定义保留 |
| `__nv_bfloat16`、`__nv_bfloat162` | reduce_kernel.h、CUDA bf16 头 | BF16 同类特化的定义保留 |
| `__nv_fp8_e4m3/e5m2` 及 x2/x4 类型 | reduce_kernel.h、CUDA fp8 头 | FP8 cast 与特化；sm_80 保留部分 cast，sm_90 额外保留 FP8 归约/pre/post-op |
| 设备 lambda、模板/特化、constexpr、union/位重解释、循环展开 pragma | all_reduce.h、common_kernel.h、reduce_kernel.h、op128.h | 编译分发、pack 展开和位重解释；原代码要求 `--extended-lambda` |
| `asm volatile`、输入输出约束 `r/h/l`、`=r/=h/=l/+r`、`memory` clobber | op128.h、common.h、reduce_kernel.h 等 | 内联 PTX、16/32/64 位寄存器、读改写和编译器内存屏障；全部原文见 asm TSV |

`coop.h` 还保留 `ncclCoopAny` 的设备函数指针/vtable 定义；本 kernel 未实例化它，
最终没有该分发表。若目标前端连这种定义都不能解析，仍须在后续裁剪中排除它。

## 4. 全部设备内建、构造与原子引用：模块和作用

以下各组名称均来自设备预处理结果；精确出现位置见 device-references.tsv。

| 接口 | 使用模块 | 作用与本例状态 |
| --- | --- | --- |
| `__syncthreads` | allreduce.cu、coop.h | block 同步；kernel 初始化是主路径，coop 封装定义保留 |
| `__syncwarp` | prims_simple.h、unpack.h、coop.h | warp 同步；32 线程备用分支与 unpack 均生成，coop 定义保留 |
| `__all_sync` | common_kernel.h | 所有 lane 对齐时选择 16 B pack，主路径；生成 `vote.sync.all.pred` |
| `__any_sync` | prims_simple.h | 单 warp OR 投票，备用生成；128/96 线程配置使用 block barrier OR |
| `__ballot_sync` | prims_simple.h | 生成 unpack peer 位掩码，备用生成 |
| `__barrier_sync_count` | coop.h | 指定 barrier ID/参与线程数的 warp group 同步，定义保留 |
| `__activemask` | coop.h | 取得当前活动 lane mask，定义保留 |
| `__shfl_sync` | coop.h | coop 广播/从 root lane 取值，定义保留，最终无 shfl 指令 |
| `__fns` | coop.h | 在 mask 中寻找第 n 个置位 lane，定义保留 |
| `__ffs`、`__ffsll` | unpack.h、bitops.h、collectives.h、coop.h | 查找最低置位位；unpack 的 __ffs 生成 brev+bfind，其他 helper/PAT/coop 定义保留 |
| `__popc`、`__popcll` | bitops.h、collectives.h、coop.h | 置位数、PAT 调度和 lane 排序，定义保留，最终无 popc 指令 |
| `__clz`、`__clzll` | bitops.h | 前导零计数、log2、紧凑整数编码，定义保留 |
| `__brev`、`__brevll` | bitops.h | 位序反转 helper，定义保留；最终 brev.b32 是 unpack __ffs 的编译展开，不能混同来源 |
| `__funnelshift_r` | op128.h | 合并相邻字处理非对齐读/复制，定义保留，最终无 shf 指令 |
| `__byte_perm` | reduce_kernel.h | uint8 pack Sum/Prod 的字节重排，定义保留；最终 prmt 另来自 kernel 参数与 unpack |
| `__umulhi`、`__umul64hi` | bitops.h、utility.h、reduce_kernel.h | 乘积高半部，辅助快速除法/整数 Avg，定义保留 |
| `__ldg` | utility.h | 只读缓存加载、opaqueZero、loadConst；定义保留 |
| `__cvta_generic_to_global` | op128.h | 转到 global 地址，FIFO/数据访问主路径 |
| `__cvta_generic_to_shared` | op128.h、common.h | shared 地址转换 helper；定义保留，另有 unpack inline cvta 备用生成 |
| `__trap` | common_kernel.h、reduce_kernel.h | 非法 pack/不支持的 multimem 模板的失败分支；本例不实例化，最终无 trap 指令 |
| `__assert_fail` → `__assertfail` | prims_simple.h、设备断言支持 | 检查角色数不超过线程数；生成 `call.uni` 和外部设备符号，正常数据不触发 |
| `__builtin_bswap16`、`__builtin_bswap32`、`__builtin_bswap64` | bitops.h | CPU/GPU 共用位运算中的字节反转，定义保留；它们是编译器 builtin，不是 CUDA Runtime API |
| `__builtin_unreachable` | utility.h | atomic scope switch 的不可达分支，定义保留；不是可执行 ISA |
| `__half2float`、`__float2half_rn` | reduce_kernel.h | FP16 与 float 标量 cast，定义保留 |
| `__half22float2`、`__float22half2_rn` | reduce_kernel.h | 两个 FP16 与 float2 cast，定义保留 |
| `__bfloat1622float2`、`__float22bfloat162_rn` | reduce_kernel.h | BF16x2 与 float2 cast，定义保留 |
| `__hadd`、`__hadd2` | reduce_kernel.h | FP16/BF16 scalar/x2 Sum；sm_90 还用于 FP8 转为 half 后归约，定义保留 |
| `__hmul`、`__hmul2` | reduce_kernel.h | FP16/BF16 Prod、PreMulSum、浮点 Avg，sm_90 还含 FP8，定义保留 |
| `__hmin`、`__hmin2`、`__hmax`、`__hmax2` | reduce_kernel.h | FP16/BF16 MinMax；sm_90 还含 FP8，定义保留 |
| `__nv_cvt_fp8_to_halfraw` | reduce_kernel.h | FP8 redOpArg/pre/post-op 参数转 half，sm_90 定义保留 |
| `__half`、`__half2` 构造表达式 | reduce_kernel.h | sm_90 FP8 运算的 half 中间表示，定义保留 |
| `__nv_fp8_e4m3`、`__nv_fp8_e5m2` 构造表达式 | reduce_kernel.h | sm_90 FP8 运算结果/缩放转换，定义保留 |
| `__nv_fp8x2_e4m3`、`__nv_fp8x2_e5m2`、`__nv_fp8x4_e4m3`、`__nv_fp8x4_e5m2` 构造表达式 | reduce_kernel.h | FP8 向量 cast；x2 还用于 sm_90 归约与 pre/post-op，定义保留 |
| `cuda::atomic_ref<...>.load/store` | utility.h、collectives.h | abort helper、通用 atomicLoad/Store、PAT 调度发布 flags，定义保留；原 Ring abort 使用 volatile load |
| `cuda::thread_scope_thread/block/device/system` | utility.h，PAT 使用 block | 原子操作的线程/CTA/GPU/system 可见范围，定义保留 |
| `cuda::memory_order_relaxed/acquire/release/acq_rel` | utility.h，PAT 发布使用 release | 原子访问顺序及其 acquire/release 分解，定义保留 |

表中使用斜线合并的 thread_scope/memory_order 是完整列举对应枚举，不是指令通配符。
FP8 类型构造属于 C++ 类型接口，不能把它们算成独立 PTX 指令。
CUDA SDK 的 `cuda_fp16.h/.hpp`、`cuda_bf16.h/.hpp`、`cuda_fp8.h/.hpp`、
`<cuda/atomic>` 仍是解析依赖；不能因为最终仅有 `add.f32` 就认为这些头已被裁掉。

## 5. 全部 inline PTX 类别：模块和作用

原文中每一条 asm，包括宏展开的每个尺寸/类型特化，都在 inline-asm.tsv。
本节用集合表示共有变体，准确拼写和完整块以 TSV 为准。

| 指令/完整变体集合 | 模块 | 作用与本例状态 |
| --- | --- | --- |
| `barrier.sync.aligned name`；`barrier.sync.aligned name,count` | common.h | 全组/worker 组同步，主路径生成；额外封装也保留定义 |
| `setp.ne.s32` + `barrier.red.or.pred` + `selp.s32` | common.h | bool→谓词、block OR 投票、谓词→bool，主路径生成 |
| `mov.u32` 读取 `%laneid` / `%lanemask_lt` | utility.h | lane helper，定义保留 |
| `ld.v2.u64`、`st.shared.v2.u64` | common.h::copyToShmem16 | 16 B 全局/generic→shared 搬运，定义保留；入口结构体赋值没有调用此 helper |
| `ld.volatile.global.v2.u64`、`st.volatile.global.v2.u64` | op128.h::load128/store128 | 128 位显式访问；load 在 unpack 中生成，store helper 定义保留 |
| `cvta.to.shared.u64`、`ld.volatile.shared.v2.u64`、`st.volatile.shared.v2.u64` | op128.h，供 unpack 调用 | unpack 元数据的 shared 访问，备用生成 |
| `ld.shared.b32`、`ld.shared.b64` | op128.h::loadShmemMisaligned128 | 非对齐 shared pack 加载，定义保留 |
| `cvta.shared.u64`、`cvta.global.u64` | op128.h::cvta_from_shared/global | 空间地址→generic 的封装定义保留；最终同名 cvta 也由编译器为共享结构体和断言生成 |
| `ld.{global,shared}.{b8,b16,b32,b64,v2.b64}` | op128.h 的尺寸宏 | 普通 scalar/16 B pack 加载，定义保留；不等于 volatile load |
| `ld.volatile.{global,shared}.{b8,b16,b32,b64,v2.b64}` | 同上 | 显式 volatile pack 读；global 数据/网络备用分支生成，shared 宏特化定义保留 |
| `st.{global,shared}.{b8,b16,b32,b64,v2.b64}` | 同上 | pack 写出；global 在 reduceCopy/unpack 中生成，shared 宏特化定义保留 |
| `ld.relaxed.gpu.global.{b8,b16,b32,b64,v2.b64}`；对应 `st` | op128.h 的 relaxed 宏 | GPU scope pack 访问，定义保留 |
| `ld.volatile.global.u32` | common_kernel.h::loadInt | connFifo offset 读取，备用生成 |
| `ld.volatile.global.u64` | op128.h | head/tail 轮询，主路径生成 |
| `ld.relaxed.sys.global.u64`、`ld.relaxed.gpu.global.u64`、`ld.acquire.sys.global.u64` | op128.h | 64 位进度读取的不同范围/顺序封装，定义保留 |
| `st.volatile.global.u64`、`st.relaxed.sys.global.u64`、`st.release.sys.global.u64` | op128.h | 64 位进度写出；只有 relaxed.sys 封装在本 kernel 生成 |
| `fence.acq_rel.sys`、`fence.acq_rel.gpu` | op128.h | system/GPU 范围内存顺序；sys 主路径生成，gpu 定义保留 |
| `ld.relaxed.gpu.u64` | unpack.h::load64gpu | 网络页元数据数量读取，备用生成；注意它没有显式 `.global` 后缀 |
| `mad.lo.u32`、`mul.hi.u32`、`mad.hi.u32` | reduce_kernel.h 的 uint8 Prod 特化 | packed byte 乘积重排，定义保留 |
| `multimem.ld_reduce.acquire.sys.global.min.u64` | prims_simple.h::loadStepValue | NVLS 最小进度轮询，sm_90 实际生成，sm_80 被预处理排除；flags=0 时不执行 |
| `multimem.st.global.b32`、`.b64`、`.v4.f32` | op128.h | NVLS 多目标写；sm_90 定义保留，MultimemDsts=0 不实例化；1/2 B 特化退化为普通 st.global.b8/b16 |
| 下述 multimem 归约变体 | reduce_kernel.h::Apply_LoadMultimem | sm_90 定义保留，MultimemSrcs=0 不实例化 |

sm_90 保留的 multimem 归约变体完整集合如下（每项前缀为 `multimem.ld_reduce.relaxed.sys.global.`）：

- `add.u32`、`add.s32`、`add.u64`、`add.f32`、`add.v4.f32`、`add.f64`。
- `min.u32`、`max.u32`、`min.s32`、`max.s32`、`min.u64`、`max.u64`、`min.s64`、`max.s64`。
- `add.acc::f32.f16x2`、`add.acc::f32.v4.f16x2`、`add.acc::f32.bf16x2`、`add.acc::f32.v4.bf16x2`。
- `min.f16x2`、`max.f16x2`、`min.v4.f16x2`、`max.v4.f16x2`。
- `min.bf16x2`、`max.bf16x2`、`min.v4.bf16x2`、`max.v4.bf16x2`。

原代码的 int64 Sum 特化也使用 `add.u64`，因此 asm 块有重复指令拼写；没有漏列 `add.s64`，
而是该处源码本来就没有使用这个拼写。FP8 multimem 的 sm_100 等架构特化不在 sm_80/sm_90 范围。

`membar.sys`、`membar.gl` 及旧架构 fence fallback 在这两个配置均被预处理排除；
`common.h` 中 `#if 0` 的非 aligned barrier 也不在保留定义清单中。

## 6. 编译器另外生成的指令

只搜索 `asm` 或 `__xxx` 会漏掉以下类别。完整 136 种助记符的每一项均在解释表中：

| 指令类别 | 来源模块 | 用途 |
| --- | --- | --- |
| `ld.param`、`st.param` 的标量/向量变体 | kernel 入口、设备断言调用 | kernel 参数载入与 __assertfail 调用参数 |
| 普通 generic/shared/global 的 `ld/st` | kernel、Primitives、reduceCopy、unpack | 共享上下文、连接结构体、指针交换备用分支、单 rank copy、元数据 |
| `add.f32` | reduce_kernel.h | 真正的 float Sum，无需原子加 |
| `add/sub/mul/div/neg` 的整数变体 | runRing、Primitives、reduceCopy、unpack、bitops | 分块、步数、偏移、范围和 pack 分配 |
| `and/or/xor/not/shl/shr` 的整数/谓词变体 | 同上 | 角色 flags、对齐、取模、数据位拆装、控制谓词 |
| `min/max`、`setp`、`selp` | 同上 | 长度裁剪、循环条件、空 slice、选择指针/数值 |
| `cvt`、`cvta`、`mov` | 各模块和编译器 | 位宽/符号、地址空间、位重解释、特殊寄存器读取 |
| `prmt.b32` | kernel 结构体赋值、unpack | 参数和 byte pack 重排；不是 uint8 归约在本例实例化了 |
| `brev.b32`、`bfind.shiftamt.u32` | unpack 的 __ffs 展开 | 查找最低置位 peer |
| `bra`、`bra.uni`、`ret`、`call.uni` | 所有控制流与设备断言 | 轮询、循环、返回、断言调用；没有函数表间接分发 |

sm_90 比 sm_80 多出的唯一**助记符种类**是 NVLS min polling 的 multimem load；
同种指令的条数也会因架构改变，不能说两个架构除了这 12 条指令以外二进制完全相同。
两个最终 PTX 都没有 `atom`、`shfl`、`shf`、`popc`、`trap`、half/BF16/FP8 数值指令、
`cp.async`、`mbarrier` 或 `mma/wgmma`。这不等于对应内建定义已从头文件裁掉。

## 7. 对兼容层工作的含义

- **必须生成的部分**：同步/投票、标量/向量内存访问、地址转换、整数控制流、float 加法、
  system fence，以及未裁掉的网络 unpack 和 sm_90 NVLS polling 备用代码。
- **仍须通过前端的部分**：半精度/FP8 类型和特化、coop/atomic/helper 定义、设备断言与扩展 asm 语法。
  最终没有某条指令，不代表前端可以完全不认识相关名字和类型。
- 定义保留的 asm 是否需要后端实现，取决于目标编译器何时验证/降低未实例化函数。
  可以通过编译期开关缩小范围；本次仅审计，没有进一步修改设备代码。

最适合优先收口的文件仍是 `op128.h`、`common.h`、`common_kernel.h`、`prims_simple.h`。
若目标是让前端也只看到 float Ring/Simple，下一步还应裁剪 `reduce_kernel.h` 的其他类型/算子、
网络 unpack、NVLS polling，以及间接引入的 coop/atomic 设备工具定义。

## 8. 如何复查完整性

在 NCCL 根目录执行，使用独立输出目录，避免已有对象掩盖不同编译选项：

```bash
for arch in 80 90; do
  build="$PWD/build-device-inventory-sm$arch"
  mkdir -p "$build/keep"
  make -C examples/allreduce_ring_simple build \
    BUILD="$build" ARCH="sm_$arch" \
    NVCCFLAGS="-std=c++17 -O2 -lineinfo --extended-lambda --keep --keep-dir $build/keep"
done
python3 docs/backend-port/inventory-device.py \
  --build sm_80=build-device-inventory-sm80 \
  --build sm_90=build-device-inventory-sm90 \
  --output docs/backend-port/device-inventory
```

审计脚本只读取编译产物，不生成/修改 kernel，也不调用 NCCL 的 device/generate.py。
它读取真正的 device 预处理文件 cpp1.ii，展开后的 asm 不会留下未展开的尺寸宏；
保留每个 asm 块和每个设备内建引用的来源，逐条读取最终 PTX，并用 .file/.loc/inlined_at 关联模块。
会处理一行多条 inline 指令、带谓词指令、无操作数 ret、设备 call 和 NVLS 指令。

这里有意排除 NCCL 文件中的纯 Host refcount builtin，且不把 CUDA SDK 自己定义的整套数学/原子
实现误算为 NCCL 调用。SDK 路径中确实生成到 kernel 的指令仍包含在最终 PTX 完整表里。
源文件行号为 0 表示编译器没有更精确的行号，不是源码第 0 行。
