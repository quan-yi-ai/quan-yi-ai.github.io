---
title: Agentic RL Infrastructure：从单轮 Rollout 到多轮环境交互
description: 从 RLHF、RLVR 到 Agentic RL，讨论轨迹长尾、Tool Loop、异步采样、策略版本与训推分离的基础设施挑战。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="agentic-rl-infrastructure" tabindex="-1">Agentic RL Infrastructure：从单轮 Rollout 到多轮环境交互</h1>
  <p class="asb-story-subtitle">动态轨迹、长尾等待与策略陈旧度如何改变后训练系统</p>
  <p class="asb-story-deck">在多轮 Agent 强化学习中，模型需要反复选择动作、调用工具、读取环境反馈并继续生成。一次 Rollout 的长度不再固定，这改变了 Batch 组织、GPU 利用率、样本就绪条件和模型版本同步的工程权衡。</p>
</div>

传统文本 RL 后训练常以一条 Prompt → Response 为最直观的数据单位。但 Agent 任务可能包含多个 Tool Call、不同长度的观察反馈与多轮推理。此时一个“样本”不再只是一段最终文本，**它更像具有状态与时间结构的 Trajectory**。

## 一、从 RLHF / RLVR 过渡到 Agentic RL

RLHF 常使用人类偏好或奖励模型来指导策略更新；RLVR 强调可验证的奖励信号，例如代码测试、数学答案匹配或其他程序化判定。二者是奖励来源和训练配方的维度，不能简单把 RLHF、RLVR 和 Agentic RL 当成互斥的三个系统版本。

Agentic RL 更突出的是**策略与外部环境的交互过程**：模型动作影响未来观察，轨迹长度和执行耗时随环境变化。奖励既可能来自最终任务成功，也可能包含工具调用成本、阶段性检查和其他约束。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Agentic RL 多轮工具交互图"><img src="/diagrams/agentic-tool-loop.svg" alt="Agentic RL Prompt Policy Tool Environment Trajectory Reward 数据流" /></button>
  <figcaption>Tool Call 形成新的环境观察，再次改变模型输入；轨迹由多个 Turn 组成。</figcaption>
</figure>

## 二、Agent Trajectory 必须保留哪些信息？

一条 Agent 轨迹通常涉及：

| 数据 | 作用 |
| --- | --- |
| Prompt / Initial Observation | 初始任务和上下文 |
| Action / Tool Call | 策略做出的可执行决策 |
| Tool Result / Next Observation | 环境反馈 |
| Token / Turn Boundary | 将多个轮次映射回模型输入输出 |
| Old Policy Logprob | 训练时与当前策略进行对照 |
| Reward / Terminal State | 判断任务收益和终止条件 |
| Loss / Attention Mask | 区分可训练 Token、观察 Token 和 Padding |

这里最重要的是**数据所有权和 Loss Mask**。例如工具返回的网页文本可能是模型下一轮推理的输入，但并不是策略自己采样出的 Action Token。如果不区分两者，就可能在优化时把环境观察错误地当成模型动作。

## 三、长尾轨迹为何会制造 Rollout Bubble？

设一个 Global Batch 中有若干 Agent 轨迹，大部分在几秒内完成，少数因为多轮搜索或环境等待持续很久。如果 Trainer 必须等待整批全部完成才能开始下一阶段，完成较早的轨迹就会占据缓冲区，而训练 GPU 可能空闲。

这种现象可以从两个层次分析：

**计算长尾**：不同样本的 Token 数和工具调用次数差异大，导致 GPU 推理批次难以平衡。

**环境长尾**：工具 API、搜索服务或外部模拟环境的等待时间与 GPU Decode 无关，增加 GPU 的空闲时间。

简单增大 Batch 可能提高 GPU 推理效率，也可能放大整批等待时间。解决方案需要在吞吐、实时交付与训练稳定性间取舍。

## 四、从全批同步到流式异步

可以将典型的执行方式粗略分为：

- **同步迭代**：Rollout 收集足够的本轮样本后训练，再完成权重更新进入下一轮。数据版本边界较清楚，但可能受慢轨迹影响。
- **部分/流式 Rollout**：轨迹或部分任务完成后即可交付消费者，让 Trainer 更早获得可用数据。
- **训推分离的异步运行**：Rollout 和 Trainer 使用独立资源，并由数据系统、任务编排与权重同步机制协调各自速度。

流式数据交付并不自动解决长尾。对于一个 GRPO Group，如果训练算法要求同 Prompt 的多个 Response 一起计算组内优势，那么单条轨迹先完成不代表 Group 已经可以直接用于训练。

### 策略版本与 Staleness

异步系统里，Rollout 生成样本时使用的策略 \(\pi_{\theta_v}\) 可能落后于当前 Trainer 策略 \(\pi_{\theta_{v+k}}\)。这里的陈旧度 \(k\) 不是仅用于日志的计数，它会影响 Importance Sampling Ratio 和训练稳定性。

因此，系统需要明确记录生成版本、模型权重更新时间、样本允许的最大 Staleness，以及必要时的丢弃、重采样或算法修正策略。不能仅通过“所有 Worker 已经收到新权重”判断历史 Buffer 中的数据可无条件用于训练。

## 五、Agentic RL 对系统组件提出什么要求？

| 组件 | 从单轮文本 RL 延伸到 Agentic RL 后的主要问题 |
| --- | --- |
| Rollout Engine | 动态 Turn、变长生成、Context 维护 |
| Agent Loop | Tool Call 解析、环境状态、错误/超时处理 |
| Data Layer | 轨迹字段、Group 完整性、部分就绪、背压 |
| Reward | 最终奖励与过程反馈的时间对齐 |
| Trainer | 变长 Packing、Action Mask、策略版本约束 |
| Weight Sync | 更新策略与正在运行的 Rollout 的隔离 |
| Scheduler | 推理 GPU、Tool API 与环境 Worker 的资源协调 |

这是为什么 Agentic RL Infra 不只是“给一个 RL 框架加上工具调用 API”。系统还需要明确某个 Trajectory 何时完整、哪些字段可以被消费、数据在失败和重试后如何保持一致。

## 六、可验证的性能指标与工程权衡

不能只看“每秒采样多少 Token”。对于多轮任务，至少建议同时关注：

1. 端到端成功完成的 Trajectory/s，而不仅是 Decode Tokens/s。
2. Rollout p50/p95/p99 完成时间和 Tool Wait 占比。
3. Trainer 空闲等待时间，以及可训练 Group 的就绪速度。
4. 训练数据 Staleness 分布，尤其是长尾轨迹的版本差。
5. Tool / Env 重试、超时和失败后轨迹回收的正确性。
6. 权重同步带来的 Rollout 暂停或资源占用。

任何声称“异步必然提高吞吐”的结论，都需要把新增通信开销、策略陈旧度和算法正确性放进同一次评估。

## 七、总结：算法、数据与执行必须三层解耦

Agentic RL 系统同时管理**动态环境交互、模型训练和异步样本交付**。算法决定什么是有效 Trajectory、如何计算 Advantage；数据系统负责字段与状态；分布式运行时负责执行实体和资源放置。

理解这些边界之后，才能有针对性地优化等待气泡、工具调用长尾、权重更新与 GPU 利用率，而不只是把同步训练循环搬到更多机器上。

## 延伸资料

[verl](https://github.com/volcengine/verl) · [AReaL](https://github.com/inclusionAI/AReaL) · [RLinf](https://github.com/RLinf/RLinf) · [PPO 原论文](https://arxiv.org/abs/1707.06347)

本文为系统机制分析，并非特定厂商实现的性能报告。“同步、部分异步、分离”等术语的具体语义取决于对应框架版本。
