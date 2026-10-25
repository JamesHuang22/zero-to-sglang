# 03 · GPU 入门（简化版）

> 来源：https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part1/第3章_GPU入门.md
> 原作 Datawhale × RadixArk，CC BY-NC-SA 4.0，非商业学习笔记，不替代原文。

## 一句话核心

GPU 用"海量简单计算单元 + 极少控制逻辑"换**吞吐**，CPU 用"复杂控制逻辑 + 大缓存"换**低延迟**；LLM 推理的 Prefill 吃 Tensor Core 算力（GEMM），Decode 吃 HBM 带宽（GEMV）。

## 核心概念（只留能算账的公式）

- **CPU vs GPU 设计哲学**：CPU 优化单任务延迟（乱序执行、分支预测、大缓存，几十个核）；GPU 优化吞吐（几千个简单 ALU，靠 Warp 切换隐藏访存延迟）。没有优劣，只是分工——复杂控制流给 CPU，海量同质乘加给 GPU。
- **A100 拓扑（能数数）**：`GPC → TPC → SM → CUDA/Tensor Core`。A100 PCIe 80GB：7 GPC / 54 TPC / **108 SM**；每 SM 64 FP32 CUDA Core → 共 **6912** 个；每 SM 4 个 Tensor Core → 共 **432** 个。功耗 300W，HBM2e 带宽约 2TB/s。
- **Tensor Core 精度账**（A100）：FP16/BF16 **312 TFLOPS**、INT8 624 TOPS、TF32 156、FP64 19.5；2:4 结构化稀疏翻倍（FP16 → 624）。推理主力是 FP16/BF16 和 INT8。
- **执行模型**：`Block → SM`（一个 block 只能落在一个 SM），Block 内线程按 **Warp = 32 线程** 调度（最小调度单元），Thread 是最细粒度执行单元。SIMT：一个 Warp 共享一条指令、各自操作不同数据；分支会导致 **Warp Divergence**（同 Warp 内走不同分支就串行执行，性能损失近半）。
- **内存层次**（离 SM 越近越快越小）：寄存器（256KB/SM，每线程最多 255 个）→ L1/共享内存（合计 192KB/SM，共享最多 164KB）→ L2（40MB，全 GPU 共享）→ HBM（80GB，~2TB/s）。全局内存延迟约 290 周期，L1 约 33 周期，差一个数量级。
- **现代替换**：A100 是 2020 架构，但分层模型不变；新卡是"各层数量和带宽更大"——H100 SXM 约 **989 TFLOPS**（BF16）、HBM3 带宽约 **3.35TB/s**（NVIDIA 官方页面已核对），B200 继续提升。面试谈"SM/Tensor Core/HBM 三层"在新卡上照样成立。

## 行业 Best Practice

- **为什么推理离不开 GPU**：① Transformer 本质是 GEMM/GEMV 海量乘加；② 70B 权重 140GB，每 token 都要搬运，HBM 2TB/s 带宽远超 CPU 主存；③ 吞吐换延迟恰好匹配"多请求并行"；④ CUDA/cuDNN/FlashAttention 生态成熟。
- **两阶段在 GPU 上的形态**：Prefill = 批量矩阵乘 GEMM，compute-bound，满载 Tensor Core，优化靠提高并行度；Decode = 矩阵-向量乘 GEMV，memory-bound，瓶颈是搬权重 + KV Cache，优化靠 KV Cache 复用、量化、continuous batching。
- **写 kernel 的实用直觉**：数据尽量放进共享内存复用（Tiling），避免频繁跨 block 访问全局内存；避免同 Warp 内分支发散；寄存器用量越少，SM 能驻留的 Warp 越多，越能隐藏延迟——这三条直接对应 FlashAttention 等高性能 kernel 的设计。

## 面试考点

1. **Q：CPU 和 GPU 的本质区别？**
   A：CPU 大面积用于控制逻辑和缓存，优化单任务低延迟；GPU 大部分晶体管是 ALU，靠海量线程和 Warp 切换优化吞吐。所以 GPU 适合大量同质的简单乘加，不适合重分支逻辑。
   追问：GPU 单任务延迟其实更高，为什么整体还快？→ 用吞吐掩盖延迟：等待访存时立刻切换其他 Warp，让算力和带宽持续满载。
   陷阱答案：说"GPU 更快所以更好"——延迟其实更差，快在吞吐，场景错了反而慢。

2. **Q：为什么 LLM 推理一定要 GPU？**
   A：推理是海量矩阵乘加 + 极高显存带宽需求 + 多请求并行，三点正好是 GPU 的长项；CPU 几十个重逻辑核 + 几百 GB/s 主存带宽扛不住 140GB 权重逐 token 搬运。
   追问：Decode 阶段 GPU 算力其实空闲，为什么还是它？→ 因为瓶颈在带宽不在算力，2TB/s 的 HBM 是关键，CPU 主存带宽差一个数量级。
   陷阱答案：只说"GPU 算力强"——Decode 恰恰是算力弱、带宽强的阶段，答案要落到带宽。

3. **Q：GPU 内存为什么分这么多层？**
   A：物理上离核心越近越快但越贵越小，所以分级：寄存器/共享内存（片上，快、小）→ L2 → HBM（片外，慢、大）。利用时间局部性（L2 缓存复用权重）和空间局部性（共享内存做 Tiling）。
   追问：这对 kernel 性能意味着什么？→ 频繁访问全局内存会让 SM 空转等数据，利用率暴跌，所以要做数据分块复用。
   陷阱答案：把各层容量背错，或不知道延迟差一个数量级（290 vs 33 周期）。

4. **Q：什么是 Warp Divergence，怎么避免？**
   A：Warp 内 32 线程共享一条指令，一旦 if/else 走不同分支，两条路径就要串行执行（掩码屏蔽），性能损失近半。避免方法是让同 Warp 线程走相同控制流，或按分支把数据重排到不同 Warp。
   追问：在 LLM 场景常见吗？→ attention/softmax 里按 mask 分支容易触发，kernel 作者会特意设计避免。
   陷阱答案：把 SIMT 当成 SIMD、认为可以自由分支——SIMT 只是"屏蔽了硬件细节"，分支代价仍然存在。

5. **Q：Tensor Core 和 CUDA Core 分工？**
   A：CUDA Core 做通用标量/向量运算；Tensor Core 专做矩阵乘加（MMA），是 AI 主力。A100 每 SM 64 CUDA Core + 4 Tensor Core，FP16 下 Tensor Core 312 TFLOPS 远高于通用路径。
   追问：精度怎么选？→ 训练常用 BF16/TF32，推理可下探 INT8 提吞吐，但要评估精度损失。
   陷阱答案：认为精度越低越快就一定好——INT8 快但精度风险，需量化感知或校准。

> 外部来源：[NVIDIA A100 Tensor Core GPU Architecture 白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)、[NVIDIA H100 页面（989 TFLOPS / 3.35TB/s 已核对）](https://www.nvidia.com/en-us/data-center/h100/)、[Stanford CS336（内存延迟数据）](https://cs336.stanford.edu/)。

## 自测（3 题）

1. 不看笔记，说出 A100 的 SM 数、CUDA Core 数、Tensor Core 数，以及 FP16 的 Tensor Core 算力。
2. 从快到慢排出 GPU 内存层次，并说明"为什么分层"和"分层怎么影响 kernel 性能"。
3. 用一句话说清 Prefill 是 GEMM/compute-bound、Decode 是 GEMV/memory-bound，并各给一个优化方向。

## 原文链接

https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part1/第3章_GPU入门.md
