---
title: RL-Kernel：MoE Router 的训推数值一致性如何验证？
description: 从 RL 后训练中的 Logprob 偏差出发，设计 MoE Router 的确定性、布局不变性、参考实现与跨引擎验证路径。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="rlkernel-consistency" tabindex="-1">RL-Kernel：MoE Router 的训推数值一致性如何验证？</h1>
  <p class="asb-story-subtitle">从 Token 路由差异到逐层可定位的跨引擎验证</p>
  <p class="asb-story-deck">训练引擎与推理引擎即使使用相同权重，也不总会得到逐比特一致的数值结果。对 MoE Router 而言，浮点舍入与 Top-K 边界附近的微小偏差可能改变被选中的专家。如何建立可重复、可解释、可追责的验证基础设施？</p>
</div>

大模型 RL 后训练中，Rollout 可能由 vLLM/SGLang 完成，而训练在 Megatron 等引擎上执行。算法会反复比较 Old Policy 与 Current Policy 的 Logprob，因此数值差异不只是一个 Benchmark 误差：它可能沿 Token 路由、模型输出和策略损失向后传播。

[RL-Kernel](https://github.com/RL-Align/RL-Kernel) 关注跨训练与推理引擎的算子级一致性与高性能实现。本文以 MoE Router 为案例，设计一个从最小操作到跨引擎执行的**验证方法论**。其中的分层测试方案是可复用的工程设计，不代表所有测试均已在某个仓库版本中落地。

## 一、MoE Router 为什么是敏感节点？

典型 MoE 层首先依据 Router Score 为每个 Token 选择 Top-K Expert，再执行 Expert FFN 并组合输出。

设 Router Logits 为 \(z_e\)，则某些配置会先计算

\[
p_e=\frac{\exp(z_e)}{\sum_j\exp(z_j)}
\]

再选择 Top-K Expert。具体模型还可能使用 Sigmoid、Group-Limited Routing、Bias 等不同变体，因此不能把 Softmax Top-K 视为所有 MoE 的统一公式。

当两个 Expert 的分数非常接近时，浮点误差、Reduce 顺序、Tie-Breaking 或 Tensor Layout 的变化都可能导致 Top-K 选择改变。选中 Expert 的变化会进一步改变该层的 FFN 输出，即使差异最初只有几个 ULP。

## 二、为什么跨引擎会产生差异？

常见差异来源包括：

| 层次 | 变化来源 | 可能导致的结果 |
| --- | --- | --- |
| 输入布局 | Padding、Packing、Token 排列 | Router 看到的有效 Token 集不一致 |
| 浮点实现 | FP32 / BF16 / FP16、Reduction 顺序 | Score 在近似值范围内偏移 |
| Kernel 调度 | Block Size、Warp、设备架构 | 非结合浮点运算的结果变化 |
| Top-K | 排序算法、稳定性和 Tie-Breaking | Expert ID 或顺序变化 |
| Expert Combine | Weighted Sum、累加次序 | 输出与梯度差异 |
| 框架语义 | 训练和推理的附加 Bias / Mask | 业务逻辑不一致 |

要判断某种差异是否属于 Bug，必须先定义“相同输入与配置”的约束。不同数据精度、不同 Router 语义却要求 Bitwise Equality，本身就不构成合理的测试目标。

## 三、由浅入深的四层验证

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 MoE Router 训推验证分层图"><img src="/diagrams/rlkernel-verification.svg" alt="确定性重放、Batch 不变性、CPU 参考实现和跨引擎一致性四层验证" /></button>
  <figcaption>从单实现稳定性开始，逐步增加对输入布局、参考语义和跨引擎环境的约束。</figcaption>
</figure>

### L1：同配置重复执行

固定权重、输入、随机种子和设备配置，多次执行相同 Router，记录 Expert IDs、Score、Top-K Order 等结果的摘要或逐元素比对。

L1 的目的是先排除**同一个实现自身不稳定**。如果不同次运行都不一致，直接定位跨引擎差异会很困难。

### L2：Batch / Packing / Padding 不变性

将同一组有效 Token 放入不同 Batch Size、Padding 或 Packing 方案中，比较有效 Token 的 Router 结果。在理论语义不依赖相邻 Token 的 Router 上，改变 Batch 组织不应改变有效 Token 的路由选择；如果模型显式依赖 Batch 统计、Token Group 或其他输入上下文，则需要重新定义测试约束。

例如测试时应保留原始 Token ID 与 Packed Position 的映射，避免把“比较错了位置”误判为数值漂移。

### L3a：GPU 实现与 CPU Oracle

为 Router 的关键路径构造一个尽可能明确的参考实现（Oracle），确定 Dtype、累加精度、Tie-Breaking、Top-K Order 和输出格式。根据实现设计选择**严格位级**或**容差型**比对，两者不能混用结论。

对于复杂 GPU Kernel，如果 Oracle 的浮点舍入语义不同，单纯拿默认 PyTorch CPU Softmax 与 GPU FP16 Kernel 要求 Bitwise Equality 可能并不合理。需要先建立真正对应的参考数值规范。

### L3b：训练与推理引擎对照

在参数、输入、Router 配置与数据布局对齐后，逐层比较 Megatron 与 vLLM 等引擎的结果。应从首个发生分歧的层、Token、Expert 和算子阶段定位，而不只报告最终 Loss 的相对误差。

最终输出差异只能证明“哪里出现了不同”，不能说明原因。**首差分定位**有助于确定最早在哪一层改变了路由选择。

## 四、首差分定位需要记录什么？

一个可重复的差异报告至少包含：

- 框架、Git Commit、GPU/驱动、Dtype 和 Parallel Configuration。
- 输入 Token 与 Layer / Expert ID 映射。
- Router Logits / Score、Top-K Expert IDs 与排序顺序。
- 两个实现之间首个不相同的数值或 Token 位置。
- 输入 Batch / Packing / Padding、Mask 和路由配置。
- 对应的最小复现脚本及预期不变性。

对“路由完全相同、Score 略有误差”和“Top-K 选择发生变化”，应记录不同的严重程度。后者改变计算路径，往往比单纯的微小 Score 漂移更需要进一步调查。

## 五、负向测试与回归测试

为了验证测试框架真的能发现问题，可以故意引入可控的错误：例如交换两个 Top-K 的顺序、改变 Tie-Breaking、向特定 Expert Score 注入偏差、改变输入 Token Map 或禁用预期的 Mask。

负向测试必须同时能回答“为什么失败”以及“第一处失败是什么”。但测试故障注入应限定在隔离环境中，不应进入生产训练或推理计算路径。

对真实的数值一致性问题，还应该保存有代表性的 Near-Tie Router 输入与极端 Dtype 情况，使修复以后有稳定的回归样本。

## 六、一致性与性能不能用同一个指标评价

逐比特一致可能要求更严格的运算顺序、精度或确定性 Kernel，有时会改变性能特征；而启用不同的 FP8/BF16 路径，也可能使完全位级相等不再是适当的测试目标。

因此必须区分：

**Correctness Contract**：哪些输出必须严格相等，哪些允许数值容差，哪些只要求相同的 Expert ID？

**Performance Contract**：在该 Correctness Contract 下，Kernel Latency、吞吐、显存和尾延迟是否达到目标？

如果一套验证只追求性能而未定义正确性契约，结果没有足够解释力；反之，脱离合理数值语义的 Bitwise 测试也可能错误地拒绝合法实现。

## 七、总结

MoE Router 一致性问题的核心是**把模糊的训练/推理漂移变成可复现的最小差异**。从同配置重放、Batch 布局不变性、参考数值实现到跨引擎对照，测试逐层增加约束，直到能够定位首个错误算子或 Token。

这类基础设施的价值不只是多一份单元测试，而是帮助不同计算引擎明确共享的路由语义及其可验证边界。

## 参考资料

[RL-Kernel 开源仓库](https://github.com/RL-Align/RL-Kernel) · [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) · [vLLM](https://github.com/vllm-project/vllm) · [PyTorch 数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

本文围绕 MoE Router 的可验证正确性契约讨论设计方法，具体测试覆盖和实际性能应以项目公开实现与实测结果为准。
