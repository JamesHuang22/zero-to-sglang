# 04 · KV Cache 优化（简化版）

> 来源：
> - https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第4章_KVCache优化.md
> - https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第4章_KVCache优化_代码.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

KV Cache 把每一层 self-attention 里历史 token 的 K、V 投影结果存下来，decode 每步只算新 token 的 K/V 并追加，把重复投影从 O(n²) 降到 O(n)——但完整的 decode 累计注意力仍是 O(n²)。

## 核心概念（只留能算账的公式）

- **无 Cache 的浪费**：模型 forward 无状态，每步都把整段历史重算。历史 token 的 K/V 投影累计 `1+2+…+n = O(n²)`；完整注意力矩阵 `1²+2²+…+n² = O(n³)`。
- **缓存对象（易错）**：每层 self-attention 的 **K 和 V**，不缓存 Q（只服务当前步）、不缓存 attention 矩阵（临时算）、不缓存 logits、不缓存 hidden state；**每一层各一份**。用 RoPE 时 Key 在旋转之后缓存，Query 当前步旋转后使用，Value 不旋转。
- **复杂度收益**（设 prompt 长度 p、生成 n 个 token）：
  - 单步 K/V 投影：`O(T) → O(1)`；n 步累计：`O(n²) → O(n)`。
  - 单步注意力：`O(T²) → O(T)`（当前 query 只查历史 KV）；n 步累计：`O(n³) → O(n²)`。
  - 所以"降到 O(n)"**只指被缓存消除的那部分重复 K/V 投影**，不是整个推理线性化。
- **KV Cache 显存（关键公式）**：
  `KV Cache 字节 = 2 × 层数 L × KV头数 H_kv × 头维 d_head × 序列长度 S × 每元素字节数`
  （2 = K 和 V；GQA/MLA 通过减小 H_kv 直接省显存；S 或 batch 翻倍，显存翻倍。）
- **读写方式**：prefill 一次算完并写入 prompt 全部 K/V；decode 每步算 1 个新 token 的 K/V 追加，再用当前 Q 查历史 KV。
- **代码篇三件事**：K/V 张量放哪（物理池）、每个 token 写到哪（逻辑→物理映射）、哪些位置可复用（前缀缓存）。

## 行业 Best Practice

- **decode 是 memory-bound，压 KV 尺寸是降 TPOT 的核心**：GQA/MLA 减 K/V 头、**KV Cache 量化（FP8/INT8）**、PagedAttention 消除碎片（第 7 章）、前缀缓存复用相同前缀（第 8 章 RadixAttention）。
- **逻辑/物理分离是分页缓存的基础**：`page_table[请求行, 逻辑位置] = 物理 token slot`，`batch.out_loc` 只含本轮新增 token 的写入地址；前缀缓存记录"某段 token 已对应哪些物理位置"，让多个请求共享同一份 K/V。
- **Failure mode**：以为"O(n²)→O(n)"是整体线性——累计注意力仍 O(n²)；KV Cache 未命中或被驱逐导致重复计算、单 token 成本抬升；写 cache 与读 cache 顺序反了 → 新 token 不参与自己这步的注意力。

## 面试考点

1. **Q："KV Cache 把复杂度从 O(n²) 降到 O(n)"具体指什么？**
   A：指生成 n 个 token 时，历史 token 的 K/V 投影从每步重算（累计 O(n²)）变成每步只算新 token（累计 O(n)）。但 decode 每步 query 仍要查全部历史 K，单步 O(T)、累计 O(n²)。
   追问：那还有没有必要优化注意力那部分？→ 有，滑窗/稀疏/线性注意力、FlashAttention 等再降。
   陷阱答案：说"整个推理变成线性复杂度"。

2. **Q：为什么只缓存 K/V，不缓存 Q、attention 矩阵或 logits？**
   A：Q 只被"当前 token 自己"用，后续 token 用不到；attention 权重是当前步临时产物，每步都变；logits 只在最后一层用于当前步采样。历史 K/V 会被每个后续 token 反复查询，缓存收益最大。
   追问：每层都要 cache 吗？→ 是，每层 self-attention 各一份，不共享。
   陷阱答案：说"只缓存最后一层"或"缓存 hidden state 就够"。

3. **Q：KV Cache 显存怎么算？怎么省？**
   A：`2 × L × H_kv × d_head × S × bytes`。省法：GQA/MLA 减小 H_kv、KV 量化减 bytes、PagedAttention 减碎片、前缀缓存复用、控制并发数（batch）。
   追问：一个 70B 模型长上下文高并发，显存大头通常是谁？→ KV Cache（可占 GPU 显存 70%+），不是权重。
   陷阱答案：只想着量化权重，忽略 KV Cache 这块更大的优化空间。

4. **Q：prefill 和 decode 对 KV Cache 的读写有何不同？**
   A：prefill 一次性并行写 prompt 全部 K/V（compute-bound，O(p²) 注意力）；decode 每步只写 1 个新 token、读全部历史 KV（memory-bound）。
   追问：为什么 decode 读历史 KV 是瓶颈？→ 每步都要从显存把整段 KV 搬进计算单元，受带宽限制。
   陷阱答案：说 prefill/decode 读写方式一样。

5. **Q：物理 KV 池、page_table、out_loc 分别是什么？**
   A：物理池（`MHAKVCache._kv_buffer`）只存数值，不知道属于谁；page_table 记录每个请求"逻辑位置→物理 slot"；out_loc 是当前 batch 本轮新增 token 的写入地址列表。三者配合才实现"写对地方、读对历史"。
   追问：前缀缓存复用什么？→ 复用物理位置映射（同一份 K/V 给多个请求），不是复制数值。
   陷阱答案：说 page_table 存的是 token id 或 K/V 数值。

> 面试点外部检索：本 topic 参考 [SGLang：KV Cache 量化](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache)、[SGLang：Attention 后端](https://docs.sglang.io/docs/advanced_features/attention_backend)、[vLLM 架构总览（PagedAttention）](https://docs.vllm.ai/en/latest/design/arch_overview.html)。

### 代码在干嘛（不抄代码）

- `kvcache/mha_pool.py` `MHAKVCache`：一次性分配 `[K/V, layer, page, offset, kv_head, head_dim]` 张量；`k_cache(layer_id)/v_cache(layer_id)` 按层取 view（不复制）；`store_kv(k,v,out_loc,layer_id)` 写入新 K/V。
- `scheduler/cache.py`：`CacheManager.free_slots` 记空闲页起始 slot；`allocate_paged` 只为 `[cached_len, device_len)` 覆盖的页补齐 page_table；`cache_req` 把有效前缀登记复用、归还不需要的位置。
- `Req`（`core.py`）：`cached_len`（已有 K/V）/`device_len`（本轮处理到哪）/`extend_len = device_len - cached_len`（本轮要算的 K/V 数）。
- `attention/fa.py`：`store_kv` **先写**，再 `_fa_sgl_impl` 读历史+本轮新写入的 KV。
- **改错会出什么问题**：store_kv 与 attention 顺序反了 → 新 token 查不到自己；page_table 写错请求行 → 读到别人数据；前缀命中不 lock → 被其他请求驱逐后指向错误位置。

## 自测（3 题）

1. 不看笔记，写出 KV Cache 显存公式，并说出 GQA 和 KV 量化分别动哪个因子。
2. 解释"O(n²)→O(n)"到底指哪部分计算，为什么 decode 累计注意力仍是 O(n²)。
3. 说出 prefill/decode 对 KV Cache 的读写差异，以及它们各自是 compute-bound 还是 memory-bound。

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第4章_KVCache优化.md
https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第4章_KVCache优化_代码.md
