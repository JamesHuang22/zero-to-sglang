# 《从零手搓 SGLang》面试导向学习笔记（simplified/）

Datawhale × RadixArk《从零手搓 SGLang》教程的面试导向简化笔记。原作 CC BY-NC-SA 4.0，本目录为非商业学习笔记，不替代原文。
原文仓库：https://github.com/datawhalechina/zero-to-sglang

## 进度表

| Part | 章节 / 主题 | 简化文件 | 状态 |
|------|------------|---------|------|
| part0 | 编码伦理与开源精神 | `part0/01-编码伦理与开源精神-简化版.md` | ✅ 已完成 |
| part0 | 部署你的第一个 SGLang 服务 | `part0/02-部署你的第一个SGLang服务-简化版.md` | ✅ 已完成 |
| part1 | LLM 入门 | `part1/01-LLM入门-简化版.md` | ✅ 已完成（范例） |
| part1 | 推理入门 | `part1/02-推理入门-简化版.md` | ✅ 已完成 |
| part1 | GPU 入门 | `part1/03-GPU入门-简化版.md` | ✅ 已完成 |
| part1 | 推理的核心数据结构入门（KV Cache） | `part1/04-推理的核心数据结构-简化版.md` | ✅ 已完成 |
| part1 | Benchmark 入门 | `part1/05-Benchmark入门-简化版.md` | ✅ 已完成 |
| part2 | mini-sglang：推理引擎长什么样 | `part2/01-mini-sglang推理引擎长什么样-简化版.md` | ✅ 已完成 |
| part2 | 一个请求的旅程（概念 + 代码） | `part2/02-一个请求的旅程-简化版.md` | ✅ 已完成 |
| part2 | 前向与生成（概念 + 代码） | `part2/03-前向与生成-简化版.md` | ✅ 已完成 |
| part2 | KV Cache 优化（概念 + 代码） | `part2/04-KVCache优化-简化版.md` | ✅ 已完成 |
| part2 | HTTP 服务与并发（概念 + 代码） | `part2/05-HTTP服务与并发-简化版.md` | ✅ 已完成 |
| part2 | Continuous Batching 与调度（概念占位，据代码走读提炼） | `part2/06-ContinuousBatching与调度-简化版.md` | ✅ 已完成（代码走读） |
| part2 | Paged KV Cache 与显存管理 | `part2/07-PagedKVCache与显存管理-简化版.md` | ✅ 已完成 |
| part2 | RadixAttention 与前缀缓存（概念 + 代码） | `part2/08-RadixAttention与前缀缓存-简化版.md` | ✅ 已完成 |
| part2 | 多进程与张量并行（概念 + 代码） | `part2/09-多进程与张量并行-简化版.md` | ✅ 已完成 |
| part2 | 投机解码 | — | ⛔ 上游占位待完成 |
| part3 | 注意力后端 | `part3/01-注意力后端-简化版.md` | ✅ 已完成 |
| part3 | 量化与低精度推理 | — | ⛔ 上游占位待完成 |
| part3 | 分层缓存 | `part3/03-分层缓存-简化版.md` | ✅ 已完成 |
| part3 | 横向扩展 | — | ⛔ 上游占位待完成 |
| part3 | Prefill/Decode 分离 | — | ⛔ 上游占位待完成 |
| part4 | 用 Cookbook 部署 SGLang | — | ⛔ 上游占位待完成 |
| part4 | Profiling 与 Trace 分析 | — | ⛔ 上游占位待完成 |
| part4 | SGLang PR 工作流 | — | ⛔ 上游占位待完成 |

> 说明：
> - 「✅ 已完成」：简化文件已产出并合入本目录。
> - 「⛔ 上游占位待完成」：原文文件内容为「本章内容撰写中」，按约束跳过，不编内容补齐，等上游完成后可再补。
> - 分工：Agent 1 = part0 + part1（02–05）；Agent 2 = part2 01–05；Agent 3 = part2 06–09（第 10 章占位跳过）；Agent 4 = part3（仅 01、03，其余占位跳过）；Agent 5 = part4（全部占位，仅维护本进度表）。
