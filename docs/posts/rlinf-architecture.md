---
title: RLinf：从 Runner、WorkerGroup 到 Agentic RL 执行链路
description: 沿着 RLinf 的配置、Cluster、Placement、WorkerGroup、Channel、Runner 和 Hybrid Engine，理解 RL 分布式后训练基础设施的职责划分。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="rlinf-architecture" tabindex="-1">RLinf：从 Runner、WorkerGroup 到 Agentic RL 执行链路</h1>
  <p class="asb-story-subtitle">分布式 RL 如何连接任务编排、模型计算与环境交互</p>
  <p class="asb-story-deck">在一个强化学习训练周期里，采样、奖励、环境交互、Actor 更新与权重同步可能使用不同设备和后端。RLinf 如何将这些职责组织成可配置的分布式执行系统？沿源码的 Runner → WorkerGroup → Channel 路径理解。</p>
</div>

在传统单进程训练代码中，数据输入、模型前向、Loss、反向传播与优化器更新可以写在一个循环里。然而 RL 后训练还涉及 **Rollout 生成、环境交互、Reward 计算、训练模型和推理模型间的权重更新**。各阶段可能运行在不同进程、设备甚至机器上；因此真正复杂的不是单个 GRPO 公式，而是任务生命周期、执行资源和数据流之间的组合关系。

本文以 [RLinf 开源仓库](https://github.com/RLinf/RLinf) 的配置、Runner、Scheduler 和 Worker 层为依据。讨论的是项目架构与源码阅读路径，不把不同算法配置或后端的能力视为任何环境下都能直接复现的性能结果。

## 一、RLinf 的分层模型

RLinf 将分布式执行拆成几个相互配合的抽象：**Runner** 负责训练任务的生命周期，**Worker** 表示可远程部署的计算单元，**WorkerGroup** 表示一组并行 Worker，**Channel** 负责轨迹和其他数据交换，**ComponentPlacement** 描述组件的资源放置，**Hybrid Engine** 则连接 FSDP / Megatron 与推理后端。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 RLinf 整体架构"><img width="960" height="600" src="/diagrams/rlinf-architecture.svg" alt="从 YAML、Runner、WorkerGroup、Channel 到模型计算后端的架构示意" /></button>
  <figcaption>配置定义资源和任务，Runner 组织执行阶段，WorkerGroup 与 Channel 承载分布式计算及数据路径。</figcaption>
</figure>

一个有用的阅读框架是将系统分成四层：

| 层次 | 代表组件 | 核心问题 |
| --- | --- | --- |
| 任务配置 | Hydra YAML、`rlinf/config.py` | 这一任务允许哪些算法、模型和部署组合？ |
| 生命周期控制 | Runner | 何时启动 Rollout、何时计算 Reward、何时更新 Actor？ |
| 执行与放置 | Cluster、Placement、WorkerGroup | 哪些 Worker 在哪些 GPU / 节点运行？ |
| 数据和计算 | Channel、Actor、Rollout、Hybrid Engine | Trajectory 怎样传输？训练权重怎样更新到推理端？ |

不要把每层都归结为 Ray 的功能。Ray 负责部分分布式执行与资源管理，而 RLinf 自己定义了 RL 任务生命周期、Batch Schema、Channel 和后端适配语义。

## 二、从配置到 Worker 启动

### 配置层不只是保存超参数

一次运行通常从 `examples/` 下的脚本与 Hydra YAML 开始，经由 `rlinf/config.py` 完成默认值补全、参数推导和兼容性校验。常见配置域包含 `cluster`、`runner`、`algorithm`、`actor`、`rollout`、`inference`、`reward`、`env`、`agentloop` 等。

源码入口：[rlinf/config.py](https://github.com/RLinf/RLinf/blob/main/rlinf/config.py)。

对于分布式系统来说，配置校验是一条重要的正确性边界。例如一个后端可能不支持某种模型切分；仅修改 Worker 代码无法保证这个组合被允许启动，还需要回到配置层检查资源和算法约束。

### Cluster 与 ComponentPlacement

`Cluster` 负责和 Ray 运行时、节点信息以及硬件资源衔接；`ComponentPlacement` 则将逻辑角色映射到实际资源。原文提到了几种典型部署取舍：

- **COLLOCATED**：不同阶段可以复用相同 GPU，在显存管理和执行切换上承担额外约束。
- **DISAGGREGATED**：训练和 Rollout 可分离部署，便于独立扩展，但需要处理数据传输与参数同步。
- **HYBRID**：结合计算与环境硬件的异构部署，尤其适用于部分具身任务。
- **AUTO**：由相应的动态调度策略决定资源放置，具体可用性依赖版本和配置。

这些名称代表部署策略，而不是“哪个一定更快”的结论。比较它们需要测量 Rollout 等待、模型切换、通信与 GPU 利用率。

### Worker 与 WorkerGroup

一个 Worker 可以代表 FSDP Rank、Rollout Engine、Env 实例或 Reward 计算单元。WorkerGroup 管理同类 Worker 的创建、初始化、执行与停止，常见生命周期可以概括为：

`create_group()` → `launch()` → 计算 Rank Placement → 创建远程 Worker → `init_worker()` → 任务执行 → `stop()`。

这条抽象解决的是**分布式计算实体如何被管理**，而不是一个 Trajectory 应包含哪些字段。

## 三、Channel：为什么数据路径不能只靠调用顺序？

在 RL Pipeline 中，Rollout 可能逐条产生轨迹，Reward 可能异步打分，而 Actor 则需要按 Batch 或 Microbatch 消费数据。如果各 Worker 全部采用相互直接调用，就会把业务依赖和通信拓扑紧密绑定。

RLinf 的 `Channel` 提供数据写入、读取和路由等机制，可以管理 Producer / Consumer 关系，并按配置处理异步读写和背压（Backpressure）。具体 API 与行为参见 [Channel 源码](https://github.com/RLinf/RLinf/blob/main/rlinf/scheduler/channel/channel.py)。

**重要的是，Channel 解耦了进程通信，但并没有消除数据契约。** Rollout 产生的 trajectory、token masks、logprobs、reward、group mapping 必须与 Actor / 算法实现的输入 Schema 一致，否则即使通信成功也可能得到错误的 Loss。

Channel 也不等于一个完整的模型权重同步系统；模型参数由 Hybrid Engine / Weight Syncer 的对应路径处理。

## 四、同一套分布式底座如何支撑三类任务？

### Reasoning：以 GRPO 为例

典型数据链路是：Dataset Prompt → SGLang/vLLM Rollout → Reward → Old / Reference Logprobs → Advantage → FSDP/Megatron Actor Update → Rollout Weight Sync。

对应的源码阅读入口：[ReasoningRunner](https://github.com/RLinf/RLinf/blob/main/rlinf/runners/reasoning_runner.py)。

要注意 Rollout 时使用的策略版本、训练阶段计算的 Logprobs、优势估计和权重同步并不是可以任意互换顺序的步骤。同步与异步配置改变数据的可用时间，也可能改变算法允许的策略滞后。

### Search-R1：多轮 Agent 的动态轨迹

多轮 Agent 训练不是简单地把一个 Prompt 的 Response 长度延长。一个模型可能生成 Tool Call，环境返回观察结果，模型再继续生成，直到满足终止条件。轨迹的 Turn 数和 Token 长度都是动态的。

AgentLoop 需要保留 Tool Observation、动作、Token Mask、轨迹边界及 Group 关系。对这种可变形状的 Batch，动态 Packing 与 Loss Mask 直接影响训练正确性。对应的阅读对象包括 Agent Runner、Agent Loop 与 MA 系列 Actor Worker；具体路径与行为应按所使用的版本核对。

### Embodied：环境成为独立执行单元

具身 RL 的数据单位通常是 Trajectory：Environment Observation → Policy Action → Environment Step → Reward / Advantage → Actor 或 Critic Update。

环境可以来自模拟器，也可能涉及真实设备与外部硬件。工程上应把真实机器人 API、动作时序与安全约束视为不同于普通文本 Rollout 的边界；本文不依赖真实硬件，也不声称所有配置可以替代真实环境测试。

## 五、Hybrid Engine：训练与推理如何交换参数？

训练引擎和推理引擎并不一定使用相同的并行拓扑。例如 Megatron 的 TP/PP/EP 分片与 SGLang/vLLM 的推理布局可能不同；即使参数数值相同，**参数名、张量形状、Shard 和目标 Rank 的映射关系**仍需要明确。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 RLinf 权重同步数据流"><img width="960" height="590" src="/diagrams/rlinf-weight-sync.svg" alt="RLinf 训练权重分片、转换、Rollout Engine 更新的流程" loading="lazy" /></button>
  <figcaption>权重同步的核心不是发起一次复制，而是跨训练与推理数据布局正确映射、传输和应用模型版本。</figcaption>
</figure>

典型问题包括：训练侧的 Shard 如何聚合或分桶；Rollout 侧采用何种 TP/PP；更新期间旧模型是否仍在生成；何时允许下一批 Rollout 使用新策略；以及共置部署时是否需要 Offload / Onload。

因此，权重同步既是数据传输问题，也是模型版本和执行调度问题。不能只根据“传输结束”就推断整个 Rollout 侧已经切换为新模型。

## 六、阅读和扩展 RLinf 的建议路径

如果目标是实现某项 RL Infra 功能，从**实际入口脚本**向内追踪，往往比先读遍全部目录更有效。

1. 在 `examples/` 中确定对应任务及 YAML，识别配置和硬件资源假设。
2. 阅读 [Runner](https://github.com/RLinf/RLinf/tree/main/rlinf/runners) 的 `run()`，找出训练 step 的实际阶段顺序。
3. 确认 Runner 创建了哪些 WorkerGroup，以及它们的 Rank 与 Placement。
4. 沿 [Channel](https://github.com/RLinf/RLinf/tree/main/rlinf/scheduler/channel) 检查生产、读取、Batch Schema 与背压行为。
5. 进入 [Actor / Rollout Worker](https://github.com/RLinf/RLinf/tree/main/rlinf/workers)，核对 Logprob、Loss、Optimizer 和模型版本。
6. 最后再追踪 [Hybrid Engine](https://github.com/RLinf/RLinf/tree/main/rlinf/hybrid_engines) 的权重格式和传输细节。

对于仓库外的新算法或后端，还需要确认插件/注册模块是否在每个 Ray Worker 进程中被导入。Driver 中已经注册的 Python 对象，并不会自动传播为所有独立进程的模块状态。

## 七、总结：RLinf 的真正系统边界

RLinf 的核心不是用 Ray 将训练脚本远程执行，而是定义了 **任务级 Runner、计算级 WorkerGroup、数据级 Channel，以及训练/推理后端适配**。它们结合起来才能表达 Reasoning、Agentic Search 与 Embodied RL 等不同运行形态。

读源码时可以用四个问题检查职责划分：**谁拥有生命周期？谁放置资源？谁拥有 Trajectory 数据？谁决定参数版本何时生效？** 它们对应不同的潜在性能和一致性瓶颈。

## 参考源码

[RLinf GitHub](https://github.com/RLinf/RLinf) · [rlinf/config.py](https://github.com/RLinf/RLinf/blob/main/rlinf/config.py) · [Reasoning Runner](https://github.com/RLinf/RLinf/blob/main/rlinf/runners/reasoning_runner.py) · [Channel](https://github.com/RLinf/RLinf/blob/main/rlinf/scheduler/channel/channel.py) · [Worker](https://github.com/RLinf/RLinf/tree/main/rlinf/workers) · [Hybrid Engine](https://github.com/RLinf/RLinf/tree/main/rlinf/hybrid_engines)

本文按公开源码及配置概念总结架构，不包含特定 GPU 集群的性能测量。各模块的注册、部署和运行时行为须以所选 RLinf 版本为准。
