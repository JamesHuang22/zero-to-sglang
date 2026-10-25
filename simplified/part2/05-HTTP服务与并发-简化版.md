# 05 · HTTP 服务与并发（简化版）

> 来源：
> - https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第5章_HTTP服务与并发.md
> - https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第5章_HTTP服务与并发_代码.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

把生成程序包成 HTTP 服务，解决三件事：多用户请求隔离、流式返回、取消与资源清理——但要记住 **HTTP 并发 ≠ GPU 利用率**（那要等下一章 Continuous Batching）。

## 核心概念（只留能算账的公式）

- **请求结构**：方法（POST）/路径（`/v1/chat/completions`）/头（Host、Content-Type）/体（`model`、`messages`、`max_tokens`、`stream`）。`max_tokens` 是生成 token 上限（非字符数、非保证生成这么多）；`messages` 是多轮对话列表，服务端不会自动"记住"上次对话。
- **三个容易混淆的词**：
  - 并发 = 同时管理多项未完成任务（A 等待时仍收 B）；
  - 并行 = 多项工作同一时刻执行（多核/多进程）；
  - 批处理 = 多条请求进同一次模型执行（scheduler 组 batch）。
  同时收 10 条 HTTP 请求 ≠ GPU 正在一起算它们。
- **请求隔离三件套**：`uid`（进程内递增，跨组件不变，重启后重置）+ `ack_map[uid]`（存尚未消费的结果列表）+ `event_map[uid]`（asyncio.Event，通知"有结果了"）。list 负责存数据、Event 负责唤醒，二者不能互相替代。
- **流式 vs 非流式**：流式用 SSE 逐段返回（`delta.content` 增量 + `[DONE]` 结束标记），改善 TTFT/体感；非流式收集完再一次返回。**流式不减少前向次数、不提升吞吐。**
- **协作式取消**：断开 → `AbortMsg` → tokenizer 转 `AbortBackendMsg` → scheduler 移除并释放。不是瞬间停止：消息传递有延迟，已提交 GPU 的算力不会立刻消失。
- **背压**：生成快于客户端读取 → `ack_map` 里结果积压；需缓冲上限/超时/取消，把消费速度反馈给上游。

## 行业 Best Practice

- **`async def` 不自动消除阻塞**：关键是把耗时活交给独立后端 + 用 `await` 让出事件循环。把同步生成函数放进 async 路由、中间没有 await，仍然占住线程，新连接一直得不到处理。
- **服务层与计算层解耦**：前台等待后端时继续收新请求；HTTP 层管"请求怎么进来、结果怎么出去"，后端管"模型怎么算"。
- **生命周期清理要用 `try/finally`**：当前实现里生成器收到 `finished` 后提前 `break`，`wait_for_ack` 尾部的 `del ack_map/event_map` 不会执行 → 状态残留（参考实现已知缺口）。非流式断连、detokenizer 取消清理、部分错误回复也是缺口。
- **Failure mode**：漏取消消息 → GPU 为断开的用户继续生成；慢客户端无背压 → 内存积压；把 OpenAI 风格接口当完整兼容（`model` 不切模型、`n`/`stop` 未接入、token 用量是占位）。

## 面试考点

1. **Q：`async def` 路由里放同步生成调用，为什么还是阻塞？**
   A：异步靠 `await` 让出事件循环；同步调用执行期间没有让出点，会持续占住线程，事件循环无法调度其他协程。正确做法是把耗时计算交给后端进程，HTTP 协程只 `await` 后端消息。
   追问：怎么判断某段代码会不会阻塞事件循环？→ 看它内部有没有 `await`/IO 让出，而不是看有没有 `async` 关键字。
   陷阱答案：说"加了 async 就是异步并发"。

2. **Q：并发、并行、批处理有什么区别？HTTP 并发为什么不提高 GPU 利用率？**
   A：并发是"同时管理多任务"（时间片/让出），并行是"同时刻真执行"，批处理是"多条请求进同一次前向"。API Server 能同时收 10 条请求，但后端仍可逐条算，GPU 利用率不变；要提高利用率要靠 scheduling（Continuous Batching）。
   追问：离线不做 HTTP 能做批处理吗？→ 能，直接把一组输入打包进 batch，批处理与 HTTP 无关。
   陷阱答案：说"接收的并发越多 GPU 越满"。

3. **Q：流式输出改善什么、不改善什么？**
   A：改善 TTFT 和用户体感（更早看到首段），不减少前向次数、不提升 TPOT/吞吐。SSE 是服务端单向数据流，`[DONE]` 是接口约定不是 SSE 标准。
   追问：SSE 事件和 token 一一对应吗？→ 不一定，解码和网络缓冲都会改变客户端一次读到的量。
   陷阱答案：说"流式让模型生成更快"。

4. **Q：客户端断开后 GPU 为什么可能继续生成？取消为什么不是瞬间？**
   A：断开只关网络连接，后端要靠取消消息沿链路传到 scheduler 才停止；这是协作式取消，各组件在处理到消息时主动停，已提交 GPU 的任务不会瞬间消失。
   追问：迟到的结果怎么处理？→ `listen` 发现 uid 已不在 `ack_map` 就丢弃。
   陷阱答案：说"断开连接立即停止推理"。

5. **Q：uid + ack_map + event_map 如何保证结果不串线？**
   A：uid 随消息跨组件不变；共享 `listen` 协程收到 `UserReply` 后按 uid 存进对应 `ack_map` 并 `event_map[uid].set()` 唤醒对应请求的消费者；消费者只取自己 uid 的列表，A/B 交错返回也各归各。
   追问：为什么 Event 不能替代 list？→ Event 只通知"有结果"不携带内容；list 能存数据但不能主动唤醒，缺一不可。
   陷阱答案：说"每个请求直接各抢一条消息"——实际是共享通道 + 按 uid 分发。

> 面试点外部检索：本 topic 参考 [FastAPI：StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)、[MDN：Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)、[vLLM 架构总览](https://docs.vllm.ai/en/latest/design/arch_overview.html)。

### 代码在干嘛（不抄代码）

- `OpenAICompletionRequest`（Pydantic）：声明接口字段和默认值（此"请求模型"是数据校验类，不是推理大模型）；FastAPI 解析 JSON，非法输入返回错误。
- `v1_completions`：`new_user()` 发 uid → 文本/采样参数装进 `TokenizeMsg` → 异步 ZMQ 发给 tokenizer，流式返回 `StreamingResponse`。
- `new_user`：分配 uid，建 `ack_map[uid]=[]` 和 `event_map[uid]=asyncio.Event()`。
- `listen`：生产者，循环收 `UserReply` 按 uid append + `set()` 唤醒；`wait_for_ack`：消费者，`wait → clear → 取出列表换空 → yield`。
- `stream_chat_completions`：把 `UserReply` 编成 SSE（首块带 role、增量进 `delta.content`、结尾 `finish_reason` + `[DONE]`）；`stream_with_cancellation`：检测 `request.is_disconnected()` 后 `abort_user`（清前台状态 + 发 `AbortMsg`）。
- **改错会出什么问题**：只有 `async def` 没 await → 阻塞事件循环；用 Event 替代 list → 拿不到结果内容；尾部清理放 `yield` 之后、调用方提前 break → 状态残留；取消消息不转发 → 后端继续生成。

## 自测（3 题）

1. 说出并发/并行/批处理的区别，并解释为什么 HTTP 并发不自动提高 GPU 利用率。
2. 写出 `wait_for_ack` 的 wait→clear→取列表→yield 顺序，说明 Event 为什么不能替代 ack_map。
3. 客户端断开后，取消通知依次经过哪些站？为什么"取消"不等于"立即停止"？

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第5章_HTTP服务与并发.md
https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第5章_HTTP服务与并发_代码.md
