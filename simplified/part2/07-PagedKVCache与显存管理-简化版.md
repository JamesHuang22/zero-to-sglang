# 07 · Paged KV Cache 与显存管理（简化版）

> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第7章_PagedKVCache与显存管理.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

PagedAttention 把 KV Cache 从"按最大长度预分配连续显存"改成"按固定大小 Block 分页 + 块表映射"，像 OS 虚拟内存一样解耦逻辑连续与物理连续，把显存利用率从 20%–40% 拉高，且数学结果与标准 Attention 完全等价。

## 核心概念（只留能算账的公式）

**为什么要分页——连续分配的三种浪费**：
1. **内部碎片（最严重）**：按最大长度（如 2048）预分配，实际只生成约 200 token，约 90% 预分配显存闲置；而且整个空间在请求生命周期内被独占。
2. **外部碎片**：长短不一的请求交替进出，留下大小不等的空洞；总空闲 3.60 GiB 但最大连续块仅 1024 MiB 时，一个需要 1.50 GiB 的请求就 OOM——有空间却用不了。
3. **动态扩展难**：序列每步增长，预留浪费、按需分配又要反复拷贝整块 KV；也无法高效支持并行采样 / beam search 的共享前缀。

**分页抽象（三个层级）**：
- **Block（物理块）**：分配最小单元，常见默认 16 token（vLLM 默认 block_size=16，可配 32/64）。
- **Slot（槽位）**：块内单 token 的存储位置。
- **Block Table（块表）**：第 i 个逻辑块 → 物理块号的映射数组，CPU 端供调度、GPU 端供 kernel 查表。

映射链路：`Token →（逻辑块号, 块内偏移）→ Block Table →（物理块号, Slot）→ 显存地址`。

**能算账的结论**：
- 内部碎片 ≤ 1 个 Block（只有最后一个块可能填不满）。
- 外部碎片 = 0（所有 Block 等大等价，任何空闲块可用）。
- 动态扩展 = 追加新 Block + 改块表，零拷贝。
- 序列所需块数 = ⌈L / b⌉（L 是 KV 可见长度，b 是 block size）。

**PagedAttention 怎么保证数学等价——在线 Softmax**：
分块读 K/V 不能直接相加（各块 Softmax 基准不同）。维护三个运行统计量：当前最大值 m、指数和 ℓ、加权输出 O；每读一块算出局部 (m_j, ℓ_j, O_j)，再按新基准合并：
`m' = max(m, m_j)`，`ℓ' = ℓ·e^(m−m') + ℓ_j·e^(m_j−m')`，`O' = O·e^(m−m') + O_j·e^(m_j−m')`。
最终 `o = O/ℓ`，与一次性 Softmax 等价（浮点舍入顺序不同可有微小差异，不是近似）。**PagedAttention 只改 K/V 的寻址方式，不改数学**。

## 行业 Best Practice

- **Block size 是核心调优旋钮**：小 → 块表查表开销大；大 → 内部碎片大；16 是常见默认，可按负载评估 32/64。
- **非连续访存靠两级优化**：memory coalescing（相邻线程读相邻地址）+ shared memory 暂存（把全局内存块搬进共享内存再算），否则间接寻址会拖慢 decode。
- **现代替换**：PagedAttention（vLLM）/ RadixAttention（SGLang）是生产标配，别再按最大长度预分配连续 KV；分页 + 前缀缓存（第 8 章）组合才完整。
- **failure mode**：block size 选太大 → 长尾内部碎片回到老路；忽略 coalescing → decode 带宽利用率下降；把"分页省显存"当成"显存不再紧张"——KV Cache 总量随并发×序列长度线性增长这个事实没变。

## 面试考点

1. **Q：为什么按最大长度预分配连续 KV Cache 会浪费这么多？**
   A：生产负载实际生成长度远小于最大值（约 200 vs 2048，约 90% 闲置），且整个空间独占整个请求生命周期。加上长短请求交替造成的空洞（外部碎片），总利用率只有 20%–40%。
   追问：外部碎片为什么"有空间却 OOM"？→ 连续分配要求一整块连续空间，最大连续块 < 需求就失败，总空闲再大也没用。
   陷阱答案：只提"预分配浪费"，漏掉外部碎片和动态扩展两个问题。

2. **Q：Block Table 是什么？和 OS 页表什么关系？**
   A：Block Table 是每请求的"逻辑块号→物理块号"映射数组，CPU 一份供调度、GPU 一份供 kernel 实时查表。它直接类比 OS 虚拟内存的页表：把连续逻辑地址映射到非连续物理页。
   追问：块表被写错会怎样？→ 读到错误位置的 K/V，输出错乱且难以定位。
   陷阱答案：说"Block Table 存 KV 数据"——它只存映射，KV 张量在 KV Cache Pool。

3. **Q：PagedAttention 分块计算为什么结果和一次性 Attention 一样？**
   A：注意力输出是各 Value 的加权和，权重由全局 Softmax 决定；分块只改读取顺序、不改 K/V 数值。用在线 Softmax 维护 (m, ℓ, O) 三个统计量，读一块合并一块，最终 o=O/ℓ 与全局 Softmax 等价。
   追问：为什么各块结果不能直接相加？→ 各块用各自局部最大值做基准，指数项尺度不同，直接加等于混了两套基准。
   陷阱答案：把在线 Softmax 说成"近似"——它是精确等价，浮点差异只来自舍入顺序。

> 外部检索：[PagedAttention 论文（vLLM）](https://arxiv.org/abs/2309.06180)、[Online Softmax（Milakov & Gimelshein）](https://arxiv.org/abs/1805.02867)、[vLLM cache.py（block_size 默认 16）](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)。

## 自测（3 题）

1. 不看笔记，说出连续 KV Cache 分配的三种浪费，以及分页如何各自治愈。
2. 写出"序列所需块数"的公式，并解释 block size 变大时查表开销和内部碎片怎么变。
3. 写出在线 Softmax 合并公式，解释为什么块间结果不能直接相加。

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第7章_PagedKVCache与显存管理.md
