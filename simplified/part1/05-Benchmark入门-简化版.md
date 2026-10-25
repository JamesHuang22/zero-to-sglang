# 05 · Benchmark 入门（简化版）

> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part1/第5章_Benchmark入门.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

benchmark 的价值在于**可比性**：推理负载 benchmark 用 TTFT / TPOT / ITL / 吞吐 / Goodput 量化"跑得快不快、稳不稳"，且要用**分位数（P99）而非均值**评估延迟。

## 核心概念（只留能算账的公式）

- **两类 benchmark 别混**：模型能力 benchmark（MMLU/GSM8K/HumanEval，测"答得好不好"）vs 推理负载 benchmark（TTFT/TPOT/吞吐，测"跑得快不快"）。本章只讲后者。
- **延迟指标**：**TTFT**（首 token 延迟，含排队+Prefill+首个 Decode）；**TPOT**（除首 token 外平均每 token 时间，主要由 Decode 决定）；**ITL**（相邻 token 间隔，可逐次观测抖动，TPOT 是它的均值）；**E2E**（端到端总时长）。
- **算账公式**：`E2E = TTFT + (输出 token 数 − 1) × TPOT`。例：TTFT=250ms、TPOT=25ms、生成 200 token → `250 + 25×199 = 5225ms ≈ 5.2s`。**Prefill 只占 250ms，其余全是 Decode**——长输出优化 TPOT，短输出长输入（文档问答）优化 TTFT。
- **吞吐指标**：**Throughput**（token/s 或 request/s，总体效率）；**Goodput**（满足延迟 SLA 前提下的吞吐，超限请求不计入）。并发越高吞吐越大但延迟越差，所以近年倾向用 Goodput 而非裸吞吐。
- **百分位数**：P50（中位数）、P99（尾延迟）。原文例子：TTFT 中位数 202ms、均值 246ms、**P99 862ms**（中位数 4 倍）——均值被长尾拉高，服务稳定性要守住的是 P99。
- **参考线**：行业常以 **100ms/token（=10 token/s）** 作为生成速率下限，低于此阅读体验明显下降。

## 行业 Best Practice

- **解读顺序**：先定 SLA（TTFT/TPOT 上限）→ 看分位数（P50/P99 一起看，不只均值）→ 逐项核对条件（模型/精度/硬件/并发/请求集/输出长度，任一项不同就没可比性）。
- **设计一次可复现的测试**：固定并发（`--max-concurrency`）或固定速率（`--request-rate`）；固定硬件/模型/精度（FP16 vs INT8 吞吐可差 2 倍以上）；**充分预热**（让 CUDA Graph、KV 分配器进稳态）；固定输出长度分布（长度不同吞吐不可比）。
- **SGLang 四个工具（分层）**：`bench_serving`（走 HTTP+调度器，最接近线上，输出 TTFT/TPOT/ITL/吞吐+分位数）、`bench_one_batch_server`（单 batch 端到端含 HTTP）、`bench_offline_throughput`（去网络开销测最大吞吐）、`bench_one_batch`（直接调 kernel，只做 kernel 分析）。多数情况用 `bench_serving`；内置请求集 sharegpt / random / generated-shared-prefix / agentic-trace。
- **高频误区**：只报均值不看尾延迟；未预热拿冷启动当稳态；把 `bench_one_batch` 结果当线上性能（它绕过调度器/HTTP，数值显著偏高）；跨模型比 token/s 忽略分词器差异（同一段中文 A 切 1000 token、B 切 1500 token，直接比无意义）；单机结果外推多卡（多卡有 NVLink/RDMA 通信开销）。

## 面试考点

1. **Q：TTFT 和 TPOT 分别优化什么场景？**
   A：TTFT 由 Prefill 决定，优化短输出长输入（文档问答、搜索摘要）的"首屏"体验；TPOT 由 Decode 决定，优化长输出（写作、代码生成）的生成速率。两者瓶颈不同，不能混。
   追问：聊天场景为什么 TTFT 敏感？→ 用户第一眼等首 token，TTFT 高就"发出后长时间无响应"。
   陷阱答案：认为两者都靠"加算力"解决——TTFT 是 compute-bound，TPOT 是 memory-bound，优化手段不同。

2. **Q：为什么看 P99 不看均值？**
   A：均值被长尾拉高或掩盖尾部，1% 请求的 P99 可能比中位数慢 4 倍，这些慢请求就是用户投诉来源。在线服务要同时看 P50/P95/P99。
   追问：P99 超限说明什么？→ 尾部请求遇到排队、长上下文或缓存未命中，需要定位是调度还是负载尖峰。
   陷阱答案："均值达标就算稳定"——均值达标但 P99 爆炸，服务照样被长尾用户骂。

3. **Q：Goodput 和 Throughput 的区别？**
   A：Throughput 是单位时间产出，不看质量；Goodput 是满足延迟 SLA 的那部分吞吐。并发拉高能提 Throughput 但延迟全面超限，这部分吞吐没有业务价值。
   追问：怎么定 SLA？→ 按业务可接受的 TTFT/TPOT 上限，超限请求不计入 Goodput。
   陷阱答案：只报并发 64 的吞吐比 16 高——P99 TTFT 可能已从 300ms 恶化到 3s。

4. **Q：怎么设计一次可复现、可对比的 benchmark？**
   A：固定并发或速率、硬件、模型、精度、输出长度分布，先预热再统计，报告 TTFT/TPOT/ITL/吞吐和 P50/P99。对比两个版本时请求集必须相同。
   追问：为什么必须预热？→ 冷启动含 CUDA 初始化、模型加载、KV 分配，会把一次性开销混进结果。
   陷阱答案：只发几个请求就统计，或用不同长度负载比吞吐——结果无意义。

5. **Q：`bench_one_batch` 的结果能当线上性能吗？**
   A：不能。它绕过调度器和 HTTP 层直接调 kernel，数值显著偏高，只用于 kernel 级分析；线上还含排队、调度、网络开销。
   追问：那想测真实线上怎么办？→ 用 `bench_serving`，走 HTTP + 调度器。
   陷阱答案：拿 kernel 级 20ms 说"线上也 20ms"——漏掉了调度和排队，是常见翻车点。

> 外部来源：[SGLang Benchmark & Profiling（四个工具与指标，已直接核对）](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)、[SGLang Bench Serving Guide](https://docs.sglang.io/docs/developer_guide/bench_serving)、[Artificial Analysis（性能 vs 成本 Pareto 前沿）](https://artificialanalysis.ai/)。

## 自测（3 题）

1. 不看笔记，写出 TTFT、TPOT、ITL、E2E 各自定义，以及 E2E 的公式。
2. 说出一段文本"均值 246ms、P50 202ms、P99 862ms"意味着什么，为什么服务要盯 P99。
3. 列出 SGLang 四个 benchmark 工具及其"走不走 HTTP/调度器"的区别，说明线上评估该用哪个。

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part1/第5章_Benchmark入门.md
