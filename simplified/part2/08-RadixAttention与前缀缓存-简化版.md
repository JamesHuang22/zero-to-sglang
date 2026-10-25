# 08 · RadixAttention 与前缀缓存（简化版）

> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第8章_RadixAttention与前缀缓存.md
> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第8章_RadixAttention与前缀缓存_代码.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

RadixAttention 用一棵路径压缩的 Radix Tree 把 KV Cache 按 token 前缀跨请求共享：公共前缀（system prompt、few-shot 示例、多轮对话历史）只 prefill 一次，后续请求直接复用，省下重复 prefill 的算力和显存。

## 核心概念（只留能算账的公式）

**为什么只能复用前缀、不能复用任意片段？**
因果注意力下，位置 i 的隐藏状态 `h_i = Attention(q_i, K_{0:i}, V_{0:i})` 只依赖左侧上下文；第一层之后每个位置的 K/V 都融合了从起点到此处的信息，所以只要前面任何一个 token 不同，后面的 K/V 全变。加 RoPE 后 Q/K 还显式依赖绝对位置，更锁死"只有完整一致的前缀能直接复用"。例：A=["天气","很","好"] 与 B=["心情","很","好"]，后缀 ["很","好"] 位置相同但左侧上下文不同，不能复用。

**Radix Tree vs 普通 Trie**：普通 Trie 每条边一个 token，LLM prompt 可达数千 token，节点/指针开销爆炸；Radix Tree 把无分叉的连续 token 段压缩进一条边，只在真正分裂处建节点。

**节点存什么**：`_key`（一段 token id 序列）+ `_value`（对应物理槽位索引）+ `ref_count`（引用计数）+ `timestamp`（LRU 时间戳）。关键：**Radix Tree 只存"token id → 物理槽位"的索引，不存 KV 浮点数据**，真正的 K/V 张量在 KV Cache Pool（`MHAKVCache._kv_buffer`）。

**缓存生命周期 = 六步**：
`match → schedule → lock → allocate → insert → unlock → evict/free`
- **match**：最长前缀匹配，返回 handle（cached_len + 节点），得到可复用范围；必要时 split_at 分裂节点（只重组元数据，KV 不搬移）。
- **lock**：从节点沿根逐级 `ref_count += 1`，把候选变成受保护引用（匹配到 ≠ 安全持有）。
- **allocate**：只为未命中后缀 `[cached_len, device_len)` 分配新 page，命中前缀复用已有物理页。
- **insert**：prefill 完成后把新后缀的 (token_ids, page_indices) 登记回树，发现竞争态重复页立即 free 防泄漏。
- **unlock**：`ref_count -= 1`；降到 0 变可淘汰，但不会立即释放。
- **evict / free**：空间不足时 evict 只摘 `ref_count==0` 的叶子；free 归还无法入树的非共享段。

**能算账**：命中长度 cached_len 越大，本次 prefill 要算的 `extend_len = input_len - cached_len` 越小，TTFT 和 prefill 算力都省；前缀命中率是 RadixAttention 的核心收益来源。

## 行业 Best Practice

- **前缀感知调度**：SGLang 提供 lpm（最长前缀匹配）、dfs-weight、shortest-prefill-first 等策略，把共享长前缀的请求放一起跑，提高命中率；mini-sglang 是简化 FCFS。
- **淘汰策略**：LRU（命中路径刷新 timestamp）+ 只从 ref_count==0 的叶子摘，保证被活动请求锁住的前缀不被误删。
- **trade-off**：缓存保留越多前缀 → 命中率越高但占用显存越大；evict 太激进 → 命中率掉、prefill 重复计算变多。
- **failure mode**：命中不等于安全（没 lock 就可能被 evict）；evict 摘内部节点会连带破坏所有下游路径；"请求结束就删 KV"是错的——共享前缀 unlock 后保留，等空间压力才 evict。

## 面试考点

1. **Q：前缀缓存为什么只缓存前缀，不能缓存任意公共子串？**
   A：因果注意力让每个位置的 KV 依赖全部左侧上下文，任何一个前缀 token 不同都会让后续 KV 改变，所以只有从起点开始完全一致的前缀能直接复用。后缀再相同，左侧上下文不同也白搭。
   追问：RoPE 会让这更严吗？→ 会，Q/K 显式依赖绝对位置，同 token 在不同位置也不能直接复用。
   陷阱答案：拿"字符串公共子串"的直觉套，以为中间任意相同片段都能缓存。

2. **Q：为什么用 Radix Tree 而不是普通 Trie？**
   A：普通 Trie 每条边一个 token，LLM 长 prompt（系统提示 + 多轮对话）会造出海量节点和指针；Radix Tree 把无分叉的连续 token 段压缩成一条边，只在分裂处建节点，节点少、遍历短。
   追问：split 什么时候发生？→ 匹配停在某节点中间且 match_len < node.length 时，就地拆成公共前缀节点 + 后缀节点。
   陷阱答案：说 Radix Tree 比 Trie "更准确"——两者前缀匹配都准，差别在空间和时间开销。

3. **Q：Radix Cache 里 match、lock、evict、free 各干什么？**
   A：match 找最长前缀并返回 handle；lock 沿根给 ref_count+1 保护前缀不被淘汰；evict 在空间不足时摘 ref_count==0 的叶子并归还物理页；free 直接归还无法入树的非共享段。命中只是起点，不 lock 就可能被 evict。
   追问：为什么 evict 只能摘叶子？→ 摘内部节点会破坏所有以它为前缀的下游路径，命中率崩塌。
   陷阱答案：把 lock 当分配动作（它只保护不分配）、把 evict 和 free 混为一谈。

> 外部检索：[SGLang 官方博客：RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)、[SGLang schedule_policy.py（lpm 等策略）](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)、[SGLang RadixAttention 论文](https://arxiv.org/html/2312.07104v2)。

## 自测（3 题）

1. 不看笔记，解释为什么 A=["天气","很","好"] 和 B=["心情","很","好"] 的后缀 ["很","好"] 不能复用。
2. 画出 match → lock → allocate → insert → unlock → evict 的完整生命周期，说明每步改的是"映射"还是"物理 KV"。
3. 为什么 evict 只能从 ref_count==0 的叶子开始？free 和 evict 的本质区别是什么？

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第8章_RadixAttention与前缀缓存.md
https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第8章_RadixAttention与前缀缓存_代码.md
