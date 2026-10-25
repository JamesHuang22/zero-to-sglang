# 《从零手搓 SGLang》面试导向学习笔记（simplified/）

Datawhale × RadixArk《从零手搓 SGLang》教程的面试导向简化笔记。原作 CC BY-NC-SA 4.0，本目录为非商业学习笔记，不替代原文。
原文仓库：https://github.com/datawhalechina/zero-to-sglang

## 进度表

| Part | 章节 / 主题 | 简化文件 | 对应 agent | 状态 |
|------|------------|---------|-----------|------|
| part0 | 编码伦理与开源精神 | — | Agent 1 | 待其他 agent 完成 |
| part0 | 部署你的第一个 SGLang 服务 | — | Agent 1 | 待其他 agent 完成 |
| part1 | LLM 入门 | `part1/01-LLM入门-简化版.md` | Agent 1 | ✅ 已完成（范例） |
| part1 | 推理入门 | — | Agent 1 | 待其他 agent 完成 |
| part1 | GPU 入门 | — | Agent 1 | 待其他 agent 完成 |
| part1 | 推理的核心数据结构入门 | — | Agent 1 | 待其他 agent 完成 |
| part1 | Benchmark 入门 | — | Agent 1 | 待其他 agent 完成 |
| part2 | mini-sglang：推理引擎长什么样 | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | 一个请求的旅程（概念 + 代码） | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | 前向与生成（概念 + 代码） | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | KV Cache 优化（概念 + 代码） | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | HTTP 服务与并发（概念 + 代码） | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | Continuous Batching 与调度（概念 + 代码） | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | Paged KV Cache 与显存管理 | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | RadixAttention 与前缀缓存（概念 + 代码） | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | 多进程与张量并行（概念 + 代码） | — | Agent 2 / 3 | 待其他 agent 完成 |
| part2 | 投机解码 | — | Agent 2 / 3 | 待其他 agent 完成 |
| part3 | 注意力后端 | — | Agent 4 | 待其他 agent 完成 |
| part3 | 量化与低精度推理 | — | Agent 4 | 待其他 agent 完成 |
| part3 | 分层缓存 | — | Agent 4 | 待其他 agent 完成 |
| part3 | 横向扩展 | — | Agent 4 | 待其他 agent 完成 |
| part3 | Prefill/Decode 分离 | — | Agent 4 | 待其他 agent 完成 |
| part4 | 用 Cookbook 部署 SGLang | — | Agent 5 | ⛔ 上游占位待完成 |
| part4 | Profiling 与 Trace 分析 | — | Agent 5 | ⛔ 上游占位待完成 |
| part4 | SGLang PR 工作流 | — | Agent 5 | ⛔ 上游占位待完成 |

> 说明：
> - 「✅ 已完成」仅标注本 agent 亲眼确认存在的文件；其余由对应 agent 负责，不在本表虚构进度。
> - 「⛔ 上游占位待完成」：原文文件内容为「本章内容撰写中」，按约束跳过，不编内容补齐。
> - 对应 agent 依据 worktree 分支推断（part0+part1 一名、part2 拆为 part2a/part2b 两段、part3 一名、part4 为 Agent 5）。除 Agent 5 外，其余编号为推断，以实际分工为准；part2a/part2b 的章节切分边界以对应 agent 产出为准。
