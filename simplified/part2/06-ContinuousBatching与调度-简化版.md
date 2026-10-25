# 06 · Continuous Batching 与调度（简化版）

> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第6章_ContinuousBatching与调度_代码.md
> 说明：本章概念篇（第6章_ContinuousBatching与调度.md）上游仍撰写中，本文仅据代码走读提炼，只写代码里可验证的机制，不补充代码之外的结论。
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

Continuous batching 让调度器每轮动态重组 batch：请求一完成就立刻释放槽位/显存，让新请求马上复用，而不是等整个静态 batch 全部结束——这是推理服务从"按 batch 服务"到"按 token 流式服务"的关键一跳。

## 核心概念（只留能算账的公式）

先讲直觉：静态 batching 里 batch 大小固定，一个短请求也得等同批长请求一起结束；continuous batching 每轮重新组批，任何一步空出位置就让新请求进来。

**一轮调度 = 四步**（`normal_loop`）：
`receive（收消息）→ schedule（组批）→ forward（前向）→ process（写回/回收）`。
有活干时收消息不阻塞本轮计算，没活干才阻塞等新消息。

**prefill 优先**：
```
batch = prefill_manager.schedule_next_batch(budget) or decode_manager.schedule_next_batch()
```
能组出 prefill batch 就先 prefill，否则才推进 decode。注意"等待队列非空 ≠ 一定能组出 prefill batch"，资源不足时会转去 decode。

**三种资源约束要分开算**（`PrefillAdder`）：

| 约束 | 检查 | 含义 |
| --- | --- | --- |
| 请求槽位 | table_manager.available_size | 请求表还有没有行可分配 |
| token 预算 | token_budget | 本轮还能算多少个 prefill token |
| KV 容量 | estimated_len + reserved_size > cache_manager.available_size | 为新请求扩展 + 运行中请求预留的 token 空间够不够 |

其中：
- `estimated_len = extend_len + output_len`，`extend_len = input_len - cached_len`（cached_len 是前缀命中长度，第 8 章展开）。
- `reserved_size` 初始来自 `decode_manager.inflight_tokens`（运行中 decode 已占 token 数），再累加本轮已接入请求。
- `token_budget` 是 token 数量的估算，不是显存字节数。

**chunked prefill**：长 prompt 不必一次算完，`chunk_size = min(token_budget, remain_len)`；有剩余就生成 ChunkedReq，中间片段不提交输出 token、也不进 decode 集合，放回队头下一轮继续，避免长请求一轮 prefill 把峰值显存打爆。

**结束与回收**（`_process_last_data`）：`finished =（不再有长度余量）or（命中 EOS 且未 ignore_eos）`；结束时 `decode_manager.remove_req` + `_free_req_resources`（归还请求表行 + 处理缓存）。取消走同一回收入口。注意：结束 ≠ 立即删全部物理 KV，可复用前缀可能继续留在缓存里（第 7/8 章）。

## 行业 Best Practice

- **prefill 优先是常见策略**，但持续优先 prefill 会推迟已有请求的 decode（饿死 decode），公平性要结合真实负载评估。
- **chunked prefill 是现代推理引擎标配能力**（vLLM / SGLang 均支持，默认是否开启需核对当前版本）：把长 prefill 切成多轮，控制单轮峰值显存、改善 TTFT 抖动，代价是引入多轮调度开销。
- **资源约束分层**：槽位、token 预算、KV 容量是三个独立维度，生产排查 OOM/拒绝服务时要分清是哪个先耗尽。
- **公平性 trade-off**：prefill 优先降低 TTFT，但可能抬高已有请求的 TPOT；FCFS 公平但吞吐可能下降。
- **failure mode**：队首一个超大请求按到达顺序接入失败就 `break`，会挡住它后面更小的请求——mini-sglang 是 FCFS + break，真实 SGLang 有前缀感知等更复杂策略。

## 面试考点

1. **Q：continuous batching 比静态 batching 好在哪？**
   A：静态 batching 里 batch 内所有请求等最慢的那个一起结束，短请求被长请求拖死；continuous batching 每步重新组批，完成的请求立刻释放资源、新请求立即补位。本质是把并发单位从"整个 batch"细化到"单个请求/token"。
   追问：代价是什么？→ 调度器和 kernel 要支持变长、变 batch 的输入，工程复杂度更高。
   陷阱答案：说"continuous batching 让单 token 更快"——它优化的是吞吐和资源利用率，不降低单 token 的计算延迟。

2. **Q：为什么 prefill 通常优先于 decode？为什么不能只按到达顺序？**
   A：prefill 是 compute-bound 且直接决定 TTFT，尽早完成能让请求尽快进入 decode 流式输出；decode 是 memory-bound，多攒几个一起跑更能吃满带宽。若纯 FCFS，可能让大量小请求被一个长 prefill 挡住。
   追问：prefill 一直优先有什么问题？→ 已有请求的 decode 被饿死，TPOT 恶化；需要 chunked prefill + 公平策略平衡。
   陷阱答案：说"prefill 优先级是硬性要求"——它是调度策略选择，可配置，不同负载下最优策略不同。

3. **Q：一个请求能不能进 batch，要看哪几道闸？**
   A：至少三道：请求槽位（表行）、token 预算（本轮 prefill 的算力/带宽配额）、KV 容量（为新请求扩展 + 运行中请求预留）。三者任一不满足都进不来。槽位够但 KV 容量不够，仍会被拒绝。
   追问：token_budget 和显存字节什么关系？→ 它是 token 数估算，不是字节数；实际显存要按每 token 的 KV 字节数换算。
   陷阱答案：把三者混为一谈，只查"显存够不够"。

> 外部检索：[Anyscale：Continuous Batching 详解](https://www.anyscale.com/blog/continuous-batching-llm-inference)、[vLLM 官方文档](https://docs.vllm.ai/en/latest/)。注：本章概念篇上游未完成，机制结论均来自代码走读原文，chunked prefill 的默认行为请核对 vLLM/SGLang 当前版本。

## 自测（3 题）

1. 不看笔记，写出 `normal_loop` 一轮的四个步骤，并说 prefill 与 decode 的调度优先级关系。
2. 请求进 batch 要过哪三道资源闸？写出 KV 容量判断的不等式和每个符号含义。
3. chunked prefill 解决什么问题？中间片段为什么不能提交输出 token？

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第6章_ContinuousBatching与调度_代码.md
