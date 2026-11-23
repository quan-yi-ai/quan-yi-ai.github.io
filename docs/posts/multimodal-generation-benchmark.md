---
title: 多模态文生图性能基准：如何测量 AR、DiT 与 VAE？
description: 从 AR+DiT+VAE 混合流水线出发，设计阶段级 Profiling、离线/在线 Benchmark、负载 Sweep 和可复现的性能报告。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="multimodal-generation-benchmark" tabindex="-1">多模态文生图性能基准：如何测量 AR、DiT 与 VAE？</h1>
  <p class="asb-story-subtitle">从阶段瓶颈归因到可复现的端到端 A/B 测试</p>
  <p class="asb-story-deck">AR 与 DiT/扩散模型的执行形态不同，VAE 的解码又具有新的算子和内存特征。在混合生成系统里，只测整体请求耗时很难判断优化方向。需要明确阶段边界、设备同步、负载矩阵和版本控制。</p>
</div>

多模态生成 Pipeline 可能同时包含自回归（Autoregressive）Token 生成、DiT 迭代式去噪以及 VAE 图像解码。如果仅统计从请求进入 Server 到最终图片返回的时间，即使得到一个可靠的 p95，也仍然不知道瓶颈位于哪里。

本文讨论的是**多阶段生成模型的 Benchmark 方法论**，而非对某个公开模型宣称已经实现特定优化收益。原始项目记录中含有待完成的 PR 占位编号和未填写的百分比，因此这里不复制任何未经验证的项目成果和数字。

## 一、为什么文生图 Benchmark 需要拆分阶段？

一个混合 Pipeline 的概念流程可写成：

**Prompt → AR Conditioning / Token Generation → DiT Denoising → VAE Decode → Image Encoding → Serving Response。**

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大文生图分阶段性能链路"><img src="/diagrams/multimodal-benchmark-stages.svg" alt="AR、DiT、VAE、Serving 阶段的性能测量范围" /></button>
  <figcaption>测量范围应覆盖 AR、DiT、VAE 与服务层之间的数据交付和同步开销。</figcaption>
</figure>

三个核心模型阶段的性能特性不同：

| Stage | 代表性工作负载 | 需要重点观察 |
| --- | --- | --- |
| AR | 自回归生成 Token / Conditioning | Prefill、Decode、KV Cache、Batch 策略 |
| DiT | 对 Latent 反复执行 Transformer / Denoising | 迭代步数、Kernel、Attention、Graph 与形状 |
| VAE | Latent 解码成图像像素 | Conv / Upsample、显存峰值与图像尺寸 |
| Serving | 数据路由、图片编码与响应 | Queue Wait、Serialization、网络返回 |

不能把 Stage 级 GPU Kernel 总耗时直接与端到端耗时画等号；队列等待、CPU 预处理、数据拷贝、设备同步和并行 Overlap 会改变 Critical Path。

## 二、怎样正确测量 Stage Latency？

一个常见错误是把 Python 中两次 \`time.time()\` 的差当作 GPU Kernel 时间。CUDA / NPU 等设备计算可能是异步执行的：CPU API 返回不等于设备已完成任务。

测量方法应匹配要回答的问题：

**GPU Kernel / Stage 计算时间**：选择设备 Event 或适当的 Profiler，明确同步边界，避免在每个微小 Kernel 后随意全设备同步改变负载。

**请求端到端延迟**：在 Server 收到请求与完成响应时记录 Wall-Clock Time，包含排队、数据交付、编码和返回。

**阶段跨设备等待**：分别记录 Producer 完成和 Consumer 可开始执行的时刻；跨 Stage 数据搬运与队列等待需要单独计量。

如果某两个 Stage 可并行重叠，不能直接把各阶段单独测量值相加当作请求端到端延迟。

## 三、建立一个有辨识力的负载矩阵

多模态 Benchmark 通常至少需要扫描四个维度：

1. **输出分辨率**：不同尺寸影响 DiT Latent Token 数、VAE 解码内存与像素处理成本。
2. **去噪迭代步数**：DiT Stage 的多次迭代可能对延迟近似线性贡献，但具体实现还受到缓存/图优化等影响。
3. **Prompt 长度和模型生成参数**：影响 AR Prefill 和 Decode。
4. **并发请求数**：改变 Scheduler 排队、Batch 形成与 GPU 利用率。

此外还应固定 Seed / Sampling 选项、模型权重、Precision、硬件、Runtime Version、Cache Warmup、图形捕获与 Power State。否则很难将 A/B 测量差异归因到某项代码修改。

### 哪些指标值得保留？

| 指标 | 用途 |
| --- | --- |
| End-to-End p50 / p95 / p99 | 用户可感知的尾延迟 |
| Stage AR / DiT / VAE Latency | 对比不同模型阶段的成本 |
| Requests/s 或 Images/s | 稳态总体吞吐 |
| Queue Time | 观察排队与 Batch 形成 |
| GPU HBM / 主机内存峰值 | 识别尺寸与并发导致的内存风险 |
| GPU Kernel Launch 数 | 判断是否存在 Launch-Heavy 小算子 |
| 同步和复制时间 | 检查跨 Stage 传输的隐含成本 |

## 四、离线与在线 Benchmark 不可混用

**离线生成**通常用于隔离单模型或单 Stage 的执行开销，适合找出固定形状下的 Kernel 瓶颈。

**在线 Serving**则必须把请求入队、运行时调度、并发、跨 Stage Connector、响应编码与实际网络返回纳入统计。一个离线 Kernel 优化可能不改变 Serving 的 p95，因为系统瓶颈可能仍在 AR、Queue 或网络层。

比较两套数据时应明确各自的起止时间戳，避免把不同口径的“Latency”放进同一张表里。

## 五、一次 A/B 优化实验应长什么样？

假设优化目标为 DiT Stage 中的一段 Attention Kernel：

**Baseline**：固定模型、生成参数和硬件，执行足够预热和多轮测量，记录分阶段耗时、端到端耗时与显存。

**Variant**：仅改变目标 Kernel 或编译选项，其余条件保持一致。验证数值正确性与输出质量在约定范围内。

**分析**：先看 DiT Stage 是否出现稳定收益；再看 Stage 占端到端 Critical Path 的比例；最后看不同并发与分辨率下的 p50/p95 和吞吐是否同步改善。

如果 DiT Latency 减少而总体吞吐不变，需要检查是否存在 AR Stage、VAE、队列或资源竞争造成的新瓶颈。不要将局部加速比例直接换算为全链路加速比例。

## 六、Profiling 的正确性与成本

Profiler 本身可能改变调度、同步和缓存行为。应该将短期高精度 Profiling 与长期低开销生产指标区分开，比较 Profiling 开启和关闭时的性能差异，并记录具体工具版本。

对于真实项目，报告中至少需要留下配置文件、Commit SHA、环境、模型版本、样本输入、测量脚本、Warmup 次数与原始数据。只有可复现的 Baseline，才能支撑后续优化的真实 A/B 对比。

## 七、总结

多阶段生成模型的 Benchmark 不能只输出一个端到端耗时。必须回答：**每个 Stage 做什么、什么时候真正完成、请求是否在等待上游、是否存在多个 Stage 的重叠，以及优化收益是否反映到整体用户体验**。

这一方法同样适用于多模态语音、图像与视频生成 Pipeline，只是具体 Stage 和负载矩阵会变化。

## 参考资料

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) · [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) · [CUDA Events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)

本文是可复现实验设计的技术分析，不包含未核实的 PR 编号、已完成指标或具体模型上的测量结论。
