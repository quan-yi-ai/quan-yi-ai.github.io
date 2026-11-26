# 技术文章

记录人工智能、计算系统与工程实践中的问题、思考和实现。

## 全部文章

- **[GPU 显存管理全栈导读：从 PyTorch 到推理框架与 RL 训练](/posts/gpu-memory-full-stack)** — PyTorch 分配器、TMS、SGLang/vLLM 与 slime/verl 的显存管理调用链。
- **[SGLang 服务启动日志详解：从权重加载到 CUDA Graph 与 KV Cache](/posts/sglang-startup-log-analysis)** — 基于一次真实单卡启动记录，拆解权重、KV Pool、Graph 与离线加载排障。
- **[多模态文生图性能基准：如何测量 AR、DiT 与 VAE？](/posts/multimodal-generation-benchmark)** — 分阶段 Profiler、负载 Sweep、端到端延迟与可复现 A/B 测试。
- **[RL-Kernel：MoE Router 的训推数值一致性如何验证？](/posts/rlkernel-moe-router-consistency)** — 确定性、Batch 不变性、CPU Oracle 与跨引擎首差分定位。
- **[投机解码：从无损验证到 EAGLE、MTP 与并行草稿](/posts/speculative-decoding-guide)** — 接受/拒绝采样、草稿器设计和性能模型。
- **[图解 PyTorch 编译器：从 Dynamo 到 Inductor 与 Triton](/posts/pytorch-compiler-illustrated)** — FX Graph、Guard、AOTAutograd、融合与 CUDA Graph。
- **[GPU 显存管理：从 PyTorch Allocator、CUDA VMM 到 TMS 与 Selective Offload](/posts/gpu-memory-vmm-tms-offload)** — 显存映射、模型状态卸载与正确性边界。
- **[Agentic RL Infrastructure：从单轮 Rollout 到多轮环境交互](/posts/agentic-rl-infrastructure)** — 工具调用、轨迹长尾、异步采样和 Staleness。
- **[Python 函数对象与资源协议：闭包、装饰器、生成器和 with](/posts/python-functions-context)** — 词法环境、延迟执行与资源收尾。
- **[大模型 RL 后训练框架：算法数据流与系统架构](/posts/rl-posttraining-frameworks)** — PPO/GRPO、Rollout、Advantage、同步与异步权重更新。
- **[MiniCPM-o 4.5：多模态语音推理的三阶段架构与性能边界](/posts/minicpmo-45-omni-inference)** — Thinker、Talker、Code2Wav、vLLM-Omni 与 NPU。
- **[RLinf：从 Runner、WorkerGroup 到 Agentic RL 执行链路](/posts/rlinf-architecture)** — 分布式调度、Channel、Hybrid Engine 与权重同步。
- **[Python 并发模型：从 asyncio、GIL 到多进程](/posts/python-concurrency-runtime)** — 协程、线程、TaskGroup 与独立进程的并发边界。
- **[Ray分布式计算框架](/posts/ray-distributed-framework)** — Task、Actor、ObjectRef、Raylet、GCS 与 vLLM。
- **[TransferQueue](/posts/transferqueue)** — RL 训推分离、样本字段状态、DataFlow 与存储后端。
- **[从TransferQueue重构理解python、cpp、Rust](/posts/transferqueue-python-cpp-rust)** — Rust Controller、PyO3 与传输层重构。
- **[并行方式](/posts/transformer-parallel-attention)** — Transformer 多维并行、Attention 与跨 GPU 通信。

## 探索领域

- [AI & Machine Learning](/posts/ai)
- [Systems & Computing](/posts/systems)
- [Engineering](/posts/engineering)
- [Research & Notes](/posts/research)
