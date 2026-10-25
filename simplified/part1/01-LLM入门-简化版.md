# 01 · LLM 入门（简化版）

> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part1/第1章_LLM入门.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

LLM 是一个大规模自回归语言模型，本质是"给定上文、预测下一个 token"；推理系统要解决的问题，就是把这个 token 级的自回归过程在给定硬件上做到**成本可控、延迟可接受**。

## 核心概念（只留能算账的公式）

- **Token**：模型实际消费的最小单位（分词后），中文约 1 token ≈ 1.6~1.7 个汉字。成本、上下文长度、吞吐全部按 token 计。
- **参数**：可学习权重，1 参数 = 1 个浮点数。**参数量 → 权重显存**：
  `权重显存 ≈ 参数量 × 每参数字节数`（FP16/BF16 = 2B，FP8 = 1B，INT4 = 0.5B）。
  例：70B 模型 FP16 ≈ 140GB，单张 80GB 卡放不下 → 必须量化或张量并行。
- **自回归生成**：每次只生成 1 个 token，再把它拼回输入预测下一个。代价是**显存随序列长度线性增长**（KV Cache），且无法完全并行。
- **KV Cache 显存**（直觉，第 4 / 7 章展开）：`KV Cache ≈ 2 × 层数 L × 头维 d × 序列长度 n × 字节数`。
  n 变大一倍、batch 变大一倍，KV Cache 就大一倍。原文指出：高并发下 GPU 显存 **70%+ 被 KV Cache 占用**，限制单卡并发上限与上下文长度；缓存被驱逐导致的重复计算会把单 token 成本抬高 2–3 倍（[英伟达测算，DoNews](https://www.donews.com/news/detail/4/6512672.html#1)）。
- **注意力计算量**：prefill 阶段要为全部 n 个 token 两两算注意力分数，矩阵是 `[seqLen × seqLen]`，复杂度 O(n²)，所以长输入是 **compute-bound**；decode 阶段每步只算当前 token 对历史 KV，算力小、但要把整段 KV 从显存读一遍，所以是 **memory-bound**。
- **TTFT / TPOT**：TTFT（首 token 延迟）≈ prefill 时长；TPOT（单 token 延迟）≈ decode 每步时长。原文给的交互约束：人眼阅读 ≤ 150ms/token，生产部署常以此定 SLO。

## 行业 Best Practice

- **现代架构默认（替代原始 Transformer 组件）**：Decoder-only；位置编码 **RoPE**（替代正余弦绝对编码，支持长上下文外推）；注意力 **GQA / MLA**（替代 MHA，压缩 KV，省显存）；归一化 **RMSNorm + Pre-Norm**（替代 LayerNorm + Post-Norm，训练更稳）；激活 **SwiGLU**（替代 ReLU）；FFN 常做 **MoE**。
- **推理侧**：KV Cache + continuous batching + PagedAttention / RadixAttention 是生产标配（本课程 Part II / III 展开）。
- **Trade-off**：上下文窗口越大 → KV Cache 显存越大、单卡并发越低；量化越低 → 显存越小但精度/质量风险越高。**上下文工程**（只把当前任务需要的资料放进窗口）既是成本控制也是质量手段——全塞进去会耗尽窗口、让模型"降智"。
- **Failure mode**：超上下文窗口 → 遗忘/幻觉；把资料无差别塞进 prompt → 成本暴涨 + 注意力分散；拿绝对位置编码/无 KV Cache 的老架构去套现代推理系统 → 成本模型全错。

## 面试考点

1. **Q：为什么 prefill 是 compute-bound、decode 是 memory-bound？**
   A：prefill 对整段输入做一次并行 attention，O(n²) 算力密集，受算力（FLOPS）限制；decode 每步只算 1 个新 token，算力很小，但要遍历整段 KV Cache，受显存带宽限制。所以两者优化手段不同：prefill 靠并行/算力，decode 靠压 KV 尺寸。
   追问：那"长上下文 + 高并发"场景瓶颈在哪？→ 通常卡在 KV Cache 显存和带宽。
   陷阱答案：说"decode 也靠堆算力"——decode 的算力利用率很低，堆算力几乎不降 TPOT。

2. **Q：一个 70B 模型 FP16 怎么放不进单卡？怎么解决？**
   A：70B × 2B = 140GB，超过 80GB 显存。解法：量化（INT8/INT4）、张量并行（多卡分权重）、或上更大显存卡。每降低一档精度，权重显存近似减半，但要评估精度损失。
   追问：量化后还能省什么？→ KV Cache 同样可以量化，是更大的显存优化空间。
   陷阱答案：只说"加显存"——成本不收敛；生产首选是量化 + 并行。

3. **Q：为什么现代模型用 GQA/MLA 替代 MHA？**
   A：MHA 的 K、V 头数与 Q 头数相同，KV Cache 显存大；GQA 让多个 Q 头共享一组 K/V，MLA 进一步低秩压缩 KV，显著减少 decode 阶段的显存与带宽。代价是表达能力略降，通常换来吞吐提升。
   追问：这与 MQA 什么关系？→ MQA 是全部头共享一组 KV（最省但质量降），GQA 是折中。
   陷阱答案：把注意力头数当 KV Cache 大小的唯一因素——还要看 K/V 头数（共享组数）。

4. **Q：TTFT 和 TPOT 分别由什么决定？**
   A：TTFT 由 prefill 时长决定（输入越长越慢，受算力限制）；TPOT 由 decode 单步决定（受显存带宽 + KV 规模限制）。交互产品更敏感 TTFT，流式生成体验更敏感 TPOT。
   追问：长 prompt 但用户只想看第一句，先优化谁？→ 先优化 TTFT（如前缀缓存 / prefill 加速）。
   陷阱答案：混为一谈，认为两者都靠"加大算力"解决。

> 本文件是范例；面试点的外部检索方式：每个 topic 至少 2 个来源（官方文档 / 技术博客），链接附在考点后。本章基础可查 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[DoNews 英伟达 KV Cache 测算](https://www.donews.com/news/detail/4/6512672.html#1)。

## 自测（3 题）

1. 不看笔记，写出 70B FP16 的权重显存，并说出两种把它放进 80GB 卡的方法。
2. 解释为什么"上下文越长，单卡并发越低"，并写出 KV Cache 随序列长度如何变化。
3. 说出 prefill 与 decode 各自的瓶颈类型，以及各自对应的优化方向。

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part1/第1章_LLM入门.md
