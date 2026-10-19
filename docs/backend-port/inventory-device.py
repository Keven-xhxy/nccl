#!/usr/bin/env python3
"""Audit existing NVCC --keep outputs; does not generate or modify device code."""
import argparse
import ast
import bisect
import collections
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/allreduce_ring_simple"
# PTX statements can be predicated, have no operands, or share an inline-asm line.
OP = re.compile(r"(?:^|[;{}])\s*(?:@!?%[\w$]+\s+)?([a-z][\w.:]*)(?=\s|;)")
ASM = re.compile(r'\b(?:asm|__asm__)\s*(?:volatile\s*)?\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)')
STRING = re.compile(r'"(?:\\.|[^"\\])*"')


def relative(name):
    path = Path(name)
    if not path.is_absolute():
        path = EXAMPLE / path
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return "CUDA_SDK/" + path.name


def preprocessing(path):
    chunks, origins, offsets = [], [], []
    current, line, offset = "", 0, 0
    example_lines = (EXAMPLE / "allreduce.cu").read_text().splitlines()
    cutoff = next(i for i, s in enumerate(example_lines, 1) if s.startswith("#define CUDA("))
    for text in path.read_text().splitlines():
        marker = re.match(r'#\s+(\d+)\s+"([^"]+)"', text)
        if marker:
            line, current = int(marker[1]), relative(marker[2])
            continue
        if current.startswith(("src/", "examples/")) and not (
                current.startswith("examples/") and line >= cutoff):
            chunks.append(text)
            origins.append(f"{current}:{line}")
            offsets.append(offset)
            offset += len(text) + 1
        line += 1
    symbols = collections.defaultdict(set)
    for text, origin in zip(chunks, origins):
        for name in re.findall(r"\b(__\w+)\s*\(", text):
            if name in ("__attribute__", "__align__"):
                continue
            # These two occurrences are Host RingAlgorithm refcount methods.
            if name in ("__atomic_add_fetch", "__atomic_sub_fetch") and origin.startswith("src/include/collectives.h:"):
                continue
            symbols[name].add(origin)
        for name in re.findall(r"cuda::(?:atomic_ref|thread_scope_\w+|memory_order_\w+)", text):
            symbols[name].add(origin)
    blocks = []
    joined = "\n".join(chunks)
    for match in ASM.finditer(joined):
        template = "".join(ast.literal_eval(s) for s in STRING.findall(match[1]))
        origin = origins[bisect.bisect_right(offsets, match.start()) - 1]
        blocks.append({"template": template, "source": origin})
    assert len(blocks) == len(re.findall(r"\b(?:asm|__asm__)\s*(?:volatile\s*)?\(", joined)), \
        "Unrecognized inline asm block"
    return dict(sorted(symbols.items())), blocks


def instructions(path):
    text_lines = path.read_text().splitlines()
    # NVCC can place .file definitions after the kernel body.
    files = {int(m[1]): relative(m[2]) for text in text_lines
             if (m := re.match(r'\s*\.file\s+(\d+)\s+"([^"]+)"', text))}
    location = []
    counts = collections.Counter()
    sources = collections.defaultdict(set)
    registers = collections.defaultdict(set)
    entries, externs = [], []
    version = target = ""
    for text in text_lines:
        marker = re.match(r'\s*\.file\s+(\d+)\s+"([^"]+)"', text)
        if marker:
            files[int(marker[1])] = relative(marker[2])
        marker = re.match(r"\s*\.loc\s+(\d+)\s+(\d+)", text)
        if marker:
            location = [tuple(map(int, marker.groups()))] + [
                (int(a), int(b)) for a, b in re.findall(r"inlined_at\s+(\d+)\s+(\d+)", text)]
        if text.strip().startswith(".version "):
            version = text.split()[1]
        if text.strip().startswith(".target "):
            target = text.split()[1]
        entries += re.findall(r"\.entry\s+([\w$]+)", text)
        externs += re.findall(r"\.extern\s+\.func\s+([\w$]+)", text)
        clean = text.split("//")[0]
        refs = {f"{files.get(f, str(f))}:{n}" for f, n in location}
        for match in OP.finditer(clean):
            opcode = match[1]
            counts[opcode] += 1
            sources[opcode].update(refs)
        for reg in re.findall(r"%(?:(?:tid|ntid|ctaid|nctaid)\.[xyz]|laneid|lanemask_\w+|warpid|smid|clock64|globaltimer)\b", clean):
            registers[reg].update(refs)
    assert len(entries) == 1, entries
    assert all(sources[k] for k in counts), "Missing PTX source information"
    assert all(not re.match(r"^\d+:", ref) for refs in sources.values() for ref in refs), "Unresolved .file ID"
    return counts, sources, registers, {"ptx_version": version, "target": target,
                                       "entries": entries, "device_externs": externs}


def tsv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def purpose(op):
    if op.startswith("ld."):
        if ".param." in op:
            return "读取 kernel 参数"
        if ".shared." in op:
            return "读取共享上下文、src/dst 指针或 unpack 元数据"
        if ".global." in op or op == "ld.relaxed.gpu.u64":
            return "读取输入/FIFO/进度或 unpack 数据；volatile/relaxed 保留各自访问语义"
        return "通过 generic 指针读取连接、进度、abort 或字节数据"
    if op.startswith("st."):
        if ".param." in op:
            return "填写设备断言函数的调用参数"
        if ".shared." in op:
            return "发布共享上下文、src/dst 指针或 unpack 元数据"
        if ".global." in op:
            return "写输出/FIFO；relaxed.sys.u64 用于发布进度"
        return "通过 generic 指针保存连接 step、指针交换或 unpack 字节"
    if op.startswith("cvta."):
        return "generic 与 global/shared 地址空间转换"
    if op.startswith("cvt."):
        return "整数位宽与有符号性转换，用于索引、地址及元数据"
    if op == "add.f32":
        return "FuncSum<float> 的逐元素求和"
    if op.startswith(("add.", "sub.", "mul.", "div.", "neg.")):
        return "chunk/slice/FIFO、指针偏移和 unpack 索引的整数计算"
    if op.startswith(("and.", "or.", "xor.", "not.", "shl.", "shr.")):
        return "标志/谓词、对齐、取模、数据 pack 与地址计算"
    if op.startswith(("min.", "max.")):
        return "chunk/slice/尾部或 unpack 长度边界"
    if op.startswith("setp."):
        return "产生循环、分支、角色、对齐和投票条件的谓词"
    if op.startswith("selp."):
        return "按谓词选择值或转换投票结果"
    if op.startswith("mov."):
        return "寄存器赋值、bit pack 拆装或读取线程特殊寄存器"
    if op in ("bra", "bra.uni"):
        return "条件/一致性分支、轮询及循环控制"
    if op == "ret":
        return "退出 kernel"
    if op == "call.uni":
        return "调用设备断言 __assertfail；正常测例不进入"
    if op == "prmt.b32":
        return "kernel 结构体参数及 unpack 字节重排"
    if op in ("brev.b32", "bfind.shiftamt.u32"):
        return "实现 unpack 的 __ffs，定位第一个 peer 位"
    if op == "bar.sync":
        return "kernel 初始化后的全 block 同步"
    if op == "bar.warp.sync":
        return "32 线程原语备用路径及 unpack warp 同步"
    if op == "barrier.sync.aligned":
        return "按 barrier ID 和参与线程数同步 worker/原语全组"
    if op == "barrier.red.or.pred":
        return "原语构造时的 block OR 投票并同步"
    if op == "vote.sync.all.pred":
        return "reduceCopy 的 warp 全体对齐判断"
    if op == "vote.sync.any.pred":
        return "原语的单 warp any 投票备用分支"
    if op == "vote.sync.ballot.b32":
        return "建立 unpack peer 位掩码"
    if op == "fence.acq_rel.sys":
        return "发送数据后、发布进度前的 system scope 内存顺序"
    if op == "multimem.ld_reduce.acquire.sys.global.min.u64":
        return "sm_90 的 NVLS 最小进度轮询分支；本测例标志关闭但代码仍生成"
    raise ValueError("Unexplained opcode: " + op)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, all_counts, all_sources = {}, {}, collections.defaultdict(set)
    for spec in args.build:
        name, folder = spec.split("=", 1)
        keep = Path(folder) / "keep"
        preprocessed, ptx = keep / "allreduce.cpp1.ii", keep / "allreduce.ptx"
        symbols, blocks = preprocessing(preprocessed)
        counts, sources, registers, metadata = instructions(ptx)
        all_counts[name] = counts
        for op, refs in sources.items():
            all_sources[op].update(refs)
        out = args.output / name
        out.mkdir(parents=True, exist_ok=True)
        tsv(out / "device-references.tsv", ["symbol", "sources"],
            [(k, "; ".join(sorted(v))) for k, v in sorted(symbols.items())])
        tsv(out / "inline-asm.tsv", ["source", "template"],
            [(b["source"], b["template"]) for b in blocks])
        tsv(out / "ptx-opcodes.tsv", ["opcode", "static_instruction_count", "source_locations"],
            [(k, counts[k], "; ".join(sorted(sources[k]))) for k in sorted(counts)])
        tsv(out / "special-registers.tsv", ["register", "sources"],
            [(k, "; ".join(sorted(v))) for k, v in sorted(registers.items())])
        summary[name] = {**metadata, "device_reference_names": len(symbols),
                         "inline_asm_blocks": len(blocks), "ptx_opcode_names": len(counts),
                         "static_ptx_instructions": sum(counts.values()),
                         "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (preprocessed, ptx)}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    names = sorted(all_counts)
    explanation = ["# 最终 PTX 全部助记符、模块与作用", "",
                   "由 inventory-device.py 从完整 PTX 生成；次数是代码中的静态指令条数，不是执行次数。",
                   "模块由 .file/.loc/inlined_at 映射；完整行号见各架构 ptx-opcodes.tsv。", "",
                   "| 指令 | " + " | ".join(names) + " | 来源模块 | 作用 |",
                   "| --- | " + " | ".join("---:" for _ in names) + " | --- | --- |"]
    for op in sorted(all_sources):
        modules = sorted({ref.rsplit(":", 1)[0] for ref in all_sources[op]})
        explanation.append("| `" + op + "` | " + " | ".join(str(all_counts[n][op]) for n in names)
                           + " | " + ", ".join("`" + m + "`" for m in modules)
                           + " | " + purpose(op) + " |")
    (args.output / "ptx-explained.md").write_text("\n".join(explanation) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
