# 02 · 部署你的第一个 SGLang 服务（简化版）

> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part0/Part0-部署你的第一个SGLang服务.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

在本地 NVIDIA GPU 上把 Qwen3-0.6B（权重约 1.5GB，4GB 显存可跑）用 SGLang 跑成 OpenAI 兼容的 HTTP 服务并发第一个请求，掌握一条最小可用的部署链路。

## 核心概念（只留能算账的公式）

- **模型规模账**：Qwen3-0.6B ≈ 1.5GB 权重（BF16 下 0.6B × 2B ≈ 1.2GB，加 embedding/head 等开销后约 1.5GB），是教学模型；真实 70B 模型 BF16 约 140GB，单卡放不下（第 1 章公式）。
- **环境门槛**：NVIDIA GPU 显存 ≥ 4GB，Linux 或 WSL2（不支持 Windows 原生 / macOS），驱动 `nvidia-smi` 右上角 CUDA Version ≥ 12.x。
- **安装按 CUDA 版本分叉**：SGLang 默认按 CUDA 13 构建（`uv pip install --prerelease=allow sglang`）；若驱动是 12.x，需补三条 `--force-reinstall` 把 torch / sglang-kernel / sgl-deep-gemm 换成 CUDA 12 构建，否则报 "CUDA driver version is insufficient / no kernel image is available"。
- **启动一条命令**：`python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 0.0.0.0 --port 30000`，看到 "The server is fired up and ready to roll!" 即成功。
- **接口是 OpenAI 兼容**：`/v1/chat/completions`，`openai.Client(base_url=..., api_key="None")` 直接可用——这是推理引擎的"通用方言"。

## 行业 Best Practice

- **显存控制的三个旋钮**：OOM 时降 `--mem-fraction-static 0.6`（静态显存占比）+ `--context-length 8192`（上下文窗口，直接影响 KV Cache 预留）。这是推理服务调参的两大杠杆：显存占比和上下文长度，两者都直接决定能塞多少并发。
- **老卡回退**：RTX 20 系 / T4 等老架构不兼容默认 attention backend，加 `--attention-backend triton`。这是"现代架构默认值"的典型 failure mode——默认值面向新卡，旧硬件要显式回退。
- **Qwen3 思考过程不是 bug**：默认输出 `<think>...</think>`，可请求里 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关掉，或启动加 `--reasoning-parser qwen3` 把思考内容分离到 `reasoning_content` 字段。
- **国内下载加速**：`export HF_ENDPOINT=https://hf-mirror.com` 或 `export SGLANG_USE_MODELSCOPE=true`；`export` 只对当前终端生效，启动 server 的终端必须设过，这是"下载卡住"的头号原因。
- **排查习惯**：报错时发"完整命令 + 完整报错（文本非截图）+ nvidia-smi 输出 + SGLang 版本号"，定位速度翻倍——对应第 01 章的"说人话"。

## 面试考点

1. **Q：从零部署一个 LLM 推理服务的完整步骤？**
   A：确认硬件（NVIDIA、显存、CUDA 版本）→ 建 Python 环境 → 按 CUDA 版本装 SGLang → 启动 `launch_server` 指定 model-path/host/port → 用 OpenAI 兼容接口发请求验证。核心是"环境 → 服务 → 请求"三段。
   追问：多卡时怎么变？→ 加 `--tp-size N` 做张量并行（SGLang server arguments 已确认）。
   陷阱答案：只记"装个包跑起来"，忽略了 CUDA 版本分叉——这是最常见的部署报错来源。

2. **Q：线上 OOM 了怎么排查？**
   A：先降 `--mem-fraction-static` 和 `--context-length`，再考虑模型量化或换大卡。显存被权重 + KV Cache + 激活瓜分，前两个是主要项。
   追问：为什么优先调 context-length？→ 它直接决定 KV Cache 预留，长上下文是显存主导项（第 4 章）。
   陷阱答案：只会"加显存"——成本不收敛，生产先调参、量化、并行。

3. **Q：为什么推理服务要做成 OpenAI 兼容接口？**
   A：客户端生态（openai SDK、LangChain 等）都按这个协议写，兼容后换引擎零改动；SGLang/vLLM 都支持，这是推理服务的"事实标准"。
   追问：兼容接口有什么代价？→ 需要把引擎内部参数映射到 OpenAI 字段，部分引擎特有功能（如 `enable_thinking`）要放 `extra_body`。
   陷阱答案：认为兼容接口和性能无关——接口层多一层适配，但对部署可移植性是净收益。

4. **Q：老卡启动报 "no kernel image is available" 是什么意思？**
   A：装的 kernel 是按新 CUDA/架构编译的，当前 GPU 架构或驱动 CUDA 版本不匹配。解法是换对应 CUDA 版本的 kernel 构建，或老架构回退 `--attention-backend triton`。
   追问：这和驱动版本什么关系？→ 驱动 CUDA 12.x 却装了默认 CUDA 13 依赖，就会触发，补装 CUDA 12 的 force-reinstall 命令。
   陷阱答案：以为重装驱动就行——问题常出在依赖的 CUDA 构建版本，不是驱动本身。

> 外部来源：[SGLang 安装文档](https://docs.sglang.io/get_started/install.html)、[SGLang Qwen3 Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3)、[SGLang Server Arguments（--mem-fraction-static / --attention-backend / --context-length 已核对）](https://docs.sglang.io/docs/advanced_features/server_arguments)。

## 自测（3 题）

1. 不看笔记，写出启动 SGLang 服务的一条命令，并说出 `--mem-fraction-static` 和 `--context-length` 分别控制什么。
2. 说出 CUDA 12.x 驱动下安装 SGLang 与默认安装的区别，以及装错会报什么错。
3. 用 OpenAI Python SDK 发一个请求需要哪几行关键代码？Qwen3 输出 `<think>` 怎么关？

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part0/Part0-部署你的第一个SGLang服务.md
