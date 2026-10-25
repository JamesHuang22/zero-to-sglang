# 01 · mini-sglang：推理引擎长什么样（简化版）

> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第1章_mini-sglang：推理引擎长什么样.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

mini-sglang 是一个教学版 LLM 推理引擎，回答的核心问题是：**模型负责前向计算，推理引擎负责组织这些计算**——请求谁接收、每轮算谁、KV Cache 放哪、token 怎么返回，都由引擎承担。

## 核心概念（只留能算账的公式）

- **推理引擎四类职责**：① 输入输出管理（tokenizer/detokenizer，文本↔token ids）；② 请求编排（记录每个 Req 的状态、停止条件、清理）；③ 计算调度（决定每轮把哪些请求放进 batch、算 prefill 还是 decode）；④ 状态与内存管理（KV Cache 的分配/复用/释放）。
- **逻辑分层四层**（不是进程边界）：接口层 → 编排层 → 执行层 → 状态与内存层。官方实现把它们拆到多个 worker 进程，职责划分相近但运行方式不同。
- **请求状态机**：`WAITING → PREFILL → DECODE → FINISHED`（异常则 `ABORTED`）。真实系统还会细分排队/暂停/抢占。
- **四个核心对象**（在 `core.py`）：
  - `SamplingParams`：温度、top_k、top_p、ignore_eos、max_tokens。
  - `Req`：单个请求，存 input_ids、当前序列长度、**已缓存长度**、采样参数、缓存句柄。
  - `Batch`：一次模型执行的一组请求 + 本轮 input_ids/位置/输出位置，用 `phase` 区分 prefill/decode。
  - `Context`：一次前向共享的运行时依赖（KV Cache 池、Attention 后端）。
- **两个关键区分**：
  - `cached_len`（已有 K/V 的 token 数）vs `device_len`（本轮处理到哪），差值 `extend_len = device_len - cached_len` 就是"这一轮真正要算 K/V 的 token 数"，是 KV Cache 章节的核心。
  - **控制路径**（传"这一轮算谁"的状态，scheduler/engine 处理）vs **数据路径**（传"拿什么数据算"：input_ids、位置、logits、KV Cache）。

## 行业 Best Practice

- **进程拆分**：mini-sglang 主进程跑 API Server，另起 tokenizer / detokenizer / scheduler（单卡时即 TP0，内部函数调用 engine）子进程；SGLang 单 tokenizer 模式下 HTTP Server 与 TokenizerManager 同进程，scheduler、detokenizer 各是子进程。
- **Trade-off**：拆进程换来"网络 IO / CPU 分词 / GPU 计算"重叠推进（规避短板原理），排查边界清晰（日志按 uid 看卡在哪站）；代价是跨进程通信（ZMQ）、状态同步、故障恢复（一个进程退出不会自动恢复）。
- **按瓶颈扩容，不拍脑袋加卡**：长文本请求多、tokenizer 前排队，加 tokenizer worker 才有用；如果 GPU 已满，加 tokenizer 只会让队列积压更久。先观测时间花在哪一段。
- **Failure mode**：把逻辑分层当进程边界；各环节串行导致一个慢环节拖累全局；进程崩了没做协调退出/清理。

## 面试考点

1. **Q：能跑 `model(input_ids)` 的自回归循环，为什么还不能叫推理引擎？**
   A：那只是"模型调用"。引擎要额外解决：多请求排队与 batch 选择、请求状态记录与停止条件、KV Cache 放哪/何时释放、token 如何流式返回。模型无状态（给什么算什么），状态和调度都在引擎侧。
   追问：单请求循环有什么具体缺陷？→ 每步重算整段历史（引出 KV Cache），且无法并发服务多用户。
   陷阱答案：说"包一层 HTTP 就是引擎"——HTTP 只是接口层，缺调度、缓存、编排。

2. **Q：`Req`、`Batch`、`Context` 各存什么？关系？**
   A：Req=单个请求的进度和采样参数；Batch=一轮执行的一组 Req + 本轮输入组织；Context=共享运行时（KV 池、attention 后端）。一个 Context 对应一个当前 Batch，Batch 里挂多个 Req。
   追问：为什么 `cached_len` 和 `device_len` 要分开？→ 决定这轮新增 token 的 K/V 要写多少、写哪（extend_len）。
   陷阱答案：把三个对象当进程——它们是数据结构，不是进程/线程。

3. **Q：为什么拆多进程？代价是什么？**
   A：让 tokenizer 处理新请求的同时 scheduler 推进已有请求，叠加不同耗时工作；多卡时每卡一个 worker 管本卡执行。代价：函数调用变消息传递（通信开销）、状态同步、故障恢复变难。
   追问：多卡 worker 为什么必须调度一致？→ 四卡服务同一批请求，各自选批会导致结果不一致。
   陷阱答案：说"多进程一定比单进程异步快"——单进程用线程/异步 IO/异步 GPU 也能并发，只是管理边界不同。

4. **Q：控制路径和数据路径为什么要分开？**
   A：控制路径决定"算谁"（请求状态流转），数据路径决定"拿什么算"（张量流动）。分开后能独立分析调度决策和模型计算，定位 bug 时知道卡在决策还是数据。
   追问：新机制（如 KV Cache）主要动哪条路径？→ 数据路径（缓存写入/读取），控制路径基本不变。
   陷阱答案：认为两者是两条独立网络——它们是同一系统里的两类信息流。

> 面试点外部检索：本 topic 参考 [mini-sglang 架构文档](https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md)、[SGLang 快速上手](https://docs.sglang.io/docs/get-started/quickstart)、[vLLM 架构总览](https://docs.vllm.ai/en/latest/design/arch_overview.html)。

## 自测（3 题）

1. 不看笔记，写出推理引擎四类职责，并各举一个模块例子。
2. 画出请求状态机，说明 `cached_len` 与 `device_len` 的差在 KV Cache 里代表什么。
3. 说出多进程设计的一个收益和一个代价，并解释为什么 GPU 已满时加 tokenizer worker 没用。

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第1章_mini-sglang：推理引擎长什么样.md
