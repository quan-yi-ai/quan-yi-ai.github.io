---
title: MiniCPM-o 4.5：多模态语音推理的三阶段架构与性能边界
description: 从 vLLM、vLLM-Omni、vLLM-Ascend 的框架分工出发，解析 MiniCPM-o 4.5 Thinker、Talker、Code2Wav 及流式音频推理的调度和性能瓶颈。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="minicpmo-architecture" tabindex="-1">MiniCPM-o 4.5：多模态语音推理的三阶段架构与性能边界</h1>
  <p class="asb-story-subtitle">从模型分阶段执行到 NPU 后端和流式数据交付</p>
  <p class="asb-story-deck">多模态语音生成不再是一次 LLM Decode。理解 vLLM-Omni 的 Thinker、Talker 与 Code2Wav，需要同时看模型语义、不同 Stage 的执行方式、跨 Stage 数据路由，以及 Ascend NPU 的硬件适配边界。</p>
</div>

文本自回归模型的请求通常可以概括为 Prefill → Decode → 返回 Token。然而面向音视频与语音的全模态模型存在另一条链路：**先理解输入并生成语言表示，再将语言转换为音频 Codec Token，最后从 Codec Token 合成波形**。不同阶段的计算形态、内存状态和数据粒度各不相同，直接把它们当成一个普通 Decoder 运行容易掩盖性能瓶颈。

本文综合对 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 源码的阅读和公开部署文档，讨论 MiniCPM-o 4.5 的管线与调度架构。文章分析的是公开实现的机制，不把示意命令或理论瓶颈表述为任何机器上的实测性能提升。


### Omni-Flow：在同一时间轴上看、听和说

MiniCPM-o 4.5 的模型设计以 Omni-Flow 将音视频输入、语言表示和语音输出放在共享时间轴上。多模态 Encoder 持续处理输入，语言主干结合视听表示和历史状态产生文本及控制信息，再由交错式语音解码与流式 Flow Matching Decoder 恢复音频。这一模型层面的全双工结构，需要与下文的服务阶段拆分分别理解。

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：MiniCPM-o 4.5 Omni-Flow 官方全模态流式架构"><img width="701" height="510" src="/images/figures/minicpmo-45-omni-inference-01.png" alt="MiniCPM-o 4.5 Omni-Flow 官方全模态流式架构" loading="lazy" /></button>
  <figcaption>Omni-Flow 的编码器、全双工语言主干、交错语音解码与流式波形恢复。来源：<a href="https://openbmb.cn/en/model/minicpm-o-4-5">OpenBMB / MiniCPM-o 4.5</a>。</figcaption>
</figure>

TDM 用时间分组把并行到来的模态流组织为主干可消费的序列。官方说明中的主动交互判断频率为 1 Hz；这不能解释成所有视频、音频的编码与输出都只以一秒为粒度。输入切片、主动发言判断和音频 Chunk 的粒度承担不同职责。

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：MiniCPM-o TDM 输入主动发言判断和流式输出"><img width="960" height="600" src="/diagrams/minicpm-tdm.svg" alt="MiniCPM-o TDM 输入主动发言判断和流式输出" loading="lazy" /></button>
  <figcaption>共享时间轴允许模型一边说话一边继续接收输入。依据 <a href="https://openbmb.cn/en/model/minicpm-o-4-5">官方模型说明</a>。</figcaption>
</figure>

## 一、三个代码库在解决不同问题

涉及 Ascend NPU 时，需要清楚地区分 **vLLM、vLLM-Omni、vLLM-Ascend** 的边界。

| 组件 | 主要职责 | 不应归因给它的能力 |
| --- | --- | --- |
| vLLM | 基础推理引擎、请求调度与部分执行抽象 | 不代表所有多模态 Stage 的业务语义都由 vLLM Core 处理 |
| vLLM-Omni | 多 Stage Pipeline、Stage 配置、跨阶段数据流与多模态模型适配 | 不直接替代设备底层算子实现 |
| vLLM-Ascend | Ascend NPU 硬件平台、后端算子与执行环境适配 | 不自动定义 MiniCPM-o 的音频生成算法 |

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大多模态推理软件栈"><img width="960" height="585" src="/diagrams/minicpm-stack.svg" alt="vLLM、vLLM-Omni 与 vLLM-Ascend 的功能边界" /></button>
  <figcaption>模型 Pipeline、通用推理运行时与硬件后端是三个需要分别分析的层次。</figcaption>
</figure>

在实际实现中，Stage 之间可以通过明确的 Schema 交付 Tensor、Token 或其他元数据；具体哪些对象会被物化、如何放置到 Shared Memory、何时同步，受执行后端和部署配置影响。

## 二、Thinker → Talker → Code2Wav：一次语音输出的真实阶段

根据公开的 [MiniCPM-o 4.5 Pipeline 定义](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py)，当前三阶段结构可以概括为：

1. **Thinker（Stage 0）**：处理多模态输入和语言生成，产出文本相关表示、隐藏状态或下游条件。
2. **Talker（Stage 1）**：根据来自 Thinker 的条件自回归地产生 Codec Token 或 Codec Chunk。
3. **Code2Wav（Stage 2）**：将音频 Codec 表示转换为最终波形，包含对应的声码器与重建链路。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 MiniCPM-o 4.5 三阶段流程"><img width="960" height="600" src="/diagrams/minicpm-three-stage.svg" alt="MiniCPM-o 4.5 Thinker Talker Code2Wav 三阶段推理图" loading="lazy" /></button>
  <figcaption>语言理解与文本生成、音频 Token 生成、波形恢复是不同的计算阶段。</figcaption>
</figure>

### 需要强调的版本差异

在一些较早的 vLLM-Omni 版本中，可以看到 **Thinker → Talker+Token2Wav** 的两阶段实现；而更新的公开 Pipeline 文档明确描述 **Thinker → Talker → Code2Wav** 三阶段拆分。阅读源码和做性能对比时，必须记录 Commit / Release 版本，不能将两个架构的结果直接混合比较。

源码和公开 Recipe 可参见 [当前三阶段配置](https://github.com/vllm-project/vllm-omni/blob/main/recipes/OpenBMB/MiniCPM-o-4_5.md)。

## 三、Scheduler、Executor、Worker、Model Runner 各自负责什么？

vLLM-Omni 复用了部分 vLLM 的运行时思想，但多 Stage 的复杂性要求分清四类职责：

- **Scheduler**：管理一个阶段内部或框架层面的请求推进，决定哪些请求在当前计算批次中执行。
- **Executor**：将需要运行的模型计算分派给对应 Worker，协调执行资源。
- **Worker**：表示特定设备与分布式环境上的执行实例，管理与设备相关的上下文。
- **Model Runner**：准备模型输入，执行 Forward / Decode，并处理与计算图、设备 Kernel、缓存有关的逻辑。

这些名字是**运行时角色**，不是模型的固定三阶段。Thinker、Talker 和 Code2Wav 可能各有不同的计算与状态需求，因此“有三个 Stage”不等于“有三个 GPU”或“有三套完全相同的 Scheduler”。

### 为什么跨 Stage 路由不能只是直接传 Tensor？

一个请求对应的 Talker Codec Chunk 必须送到**同一个请求关联的** Code2Wav 状态中。流式音频还需要处理边界 Chunk、参考声音条件和请求完成时的缓存清理。如果跨阶段只按执行完成顺序拼接数据，就可能在并发场景破坏请求对应关系。

当前公开实现描述了 Talker → Code2Wav 的按请求路由 Chunk 路径；部署配置和 Connector 选择决定具体的内存交付机制。

## 四、语音推理的延迟来自哪里？

音频生成很容易被一项指标误导：**首个音频 Chunk 延迟**和**完整波形生成总耗时**不是一回事。

一个简化的端到端延迟模型可以写成：

\[
T_{\text{total}} \approx T_{\text{preprocess}} + T_{\text{thinker}} + T_{\text{talker}} + T_{\text{code2wav}} + T_{\text{handoff}}
\]

这是便于分解的上界式**串行概念模型**。如果 Talker 可以持续输出 Codec Chunk，Code2Wav 又能在下一个 Chunk 到达前处理上一块，两个阶段可以流水化重叠；完整请求的 Critical Path 不再等于各阶段耗时的简单加和。

### Talker：Token 依赖与采样效率

Talker 如果以自回归方式产生 Codec Token，其生成步数、一次 Forward 覆盖的 Codec 帧数、KV Cache 和调度 Batch 形状都会影响延迟。不能简单将“音频 10 秒”直接等同于固定的 Token 个数，除非已知该版本的采样率、Codec 帧率和具体配置。

### Code2Wav：与普通 LLM 不同的计算负载

Code2Wav 包含音频特征恢复、CFM/DiT 或声码器相关步骤；其关键瓶颈可能是小算子的 Launch Overhead、形状变化、迭代步数或内存搬运，而不一定是 Transformer 的 Token Decode 速度。

对 NPU 场景，是否捕获 NPUGraph、Graph Key 是否按 Shape 区分，以及哪些预处理需要保持 Eager，均属于独立的性能与正确性问题。可参考公开 Recipe 对 Stage2 内部 Graph Boundary 的说明；不能因某阶段使用 Graph 就宣称整条语音生成流水线完全由 Graph 捕获。

## 五、如何评估优化是否真的有效？

建议为一条请求记录以下事件：接收并完成预处理、Thinker 首 Token/结束、Talker 首 Codec/结束、Code2Wav 首 Waveform Chunk/结束、完整响应完成。

| 测量指标 | 能回答的问题 |
| --- | --- |
| First Text Token Latency | Thinker 何时开始产生有效输出？ |
| First Audio Chunk Latency | 用户何时听到第一段音频？ |
| End-to-End Response Latency | 整条请求何时结束？ |
| Stage p50/p95 | 哪个阶段构成尾延迟？ |
| Codec Chunk Size / Queue Wait | 跨阶段传输与背压是否限制吞吐？ |
| NPU / GPU Memory Peak | 多 Stage 共置是否产生显存竞争？ |
| Throughput at Concurrency N | 资源利用率与 Batch 效率如何变化？ |

优化应先定位瓶颈：例如调小 Talker 的 Chunk Size 可能改善首音频延迟，却也可能增加调度/Connector 调用次数；Stage2 减少 Kernel Launch 开销可能改善波形生成，但无法解决 Stage0 很慢的请求。

## 六、Ascend NPU 平台需要单独验证的边界

在 Ascend 上运行多 Stage 管线时，至少要明确设备支持、NPU Attention Backend、Torch NPU 算子兼容、模型分片与 Graph 执行模式。CPU → NPU 的输入搬运、Stage 间共享内存、异步事件同步和混合精度也可能导致与 CUDA 不同的性能表现。

特别是 **NPUGraph 的 Capture / Replay 地址与形状要求**，不能只把 CUDA 上的执行路径逐字迁移为 NPU 行为。每个 Stage 可以有不同的可捕获范围；当 Python 请求状态、随机过程或动态 Buffer 生命周期不满足要求时，相应部分需要保留 Eager 执行。

## 七、总结：将模型结构与部署配置分离

MiniCPM-o 4.5 不是一条普通 LLM 的连续 Decode：Thinker、Talker 与 Code2Wav 有不同的输入、输出、状态和性能瓶颈。真正的系统设计必须同时明确**模型阶段边界**、**运行时 Worker/Executor 边界**以及**硬件平台边界**。

无论使用单 GPU 共置还是多 GPU 分离，都应该保持跨阶段的请求关联、正确的缓冲区与同步语义，并分别测量首音频延迟与完整请求吞吐。

## 参考资料

[vLLM-Omni 官方仓库](https://github.com/vllm-project/vllm-omni) · [MiniCPM-o 4.5 Pipeline](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py) · [MiniCPM-o 4.5 部署 Recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/OpenBMB/MiniCPM-o-4_5.md) · [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)

本文中的阶段耗时模型是概念拆分，不包含原始硬件的实测数字，也不构成某个优化 PR 的性能承诺。具体 Stage 划分和配置请与使用的 Release / Commit 核对。
