---
title: 大模型 RL 后训练框架：算法数据流与系统架构
description: 对比 PPO 与 GRPO 的计算流程，从 Rollout、Reward、Logprob、Advantage、Loss 到权重同步理解 RL 后训练框架的系统职责。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="rl-posttraining-frameworks" tabindex="-1">大模型 RL 后训练框架：算法数据流与系统架构</h1>
  <p class="asb-story-subtitle">从 PPO / GRPO 的共性数据路径理解训推解耦与权重同步</p>
  <p class="asb-story-deck">后训练框架真正需要管理的，不只是 Actor 的 Loss，而是 Prompt、Rollout、Reward、Logprob、Advantage 和策略版本在不同 GPU 和 Worker 之间的流动。先看算法必需的字段，再讨论共置、分离、同步与异步。</p>
</div>

为什么同一套 Rollout + Trainer 架构可以同时实现 PPO、GRPO 乃至很多其他 RL 算法？关键在于区分**相对稳定的系统数据流**与**随算法变化的优势估计、Loss 和优化策略**。但这不意味着不同算法只需替换某一个小函数：Reward 设计、Reference Policy、轨迹采样、更新次数与数据分组方式都可能影响系统执行。

## 一、从一次后训练迭代出发

大模型 RL 后训练通常需要经历：

**Prompt → Policy Rollout → Reward → 训练所需的 Logprobs / Mask → Advantage → Actor Update → 同步推理权重 → 下一批采样。**

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 RL 后训练迭代数据链路"><img width="960" height="570" src="/diagrams/rl-posttraining-flow.svg" alt="RL 后训练中 Prompt、Rollout、Reward、Advantage、Update 数据路径" /></button>
  <figcaption>无论采用何种具体策略优化算法，都需要明确各阶段输入输出与数据所有权。</figcaption>
</figure>

Rollout 侧的目标是生成轨迹；Trainer 侧的目标是根据选定的目标函数计算梯度。这两类工作负载的最优 Batch、显存使用和并行布局未必相同，因此会自然引出**共置（Colocated）**与**分离（Disaggregated）**两种部署方式。

## 二、PPO 与 GRPO：相同数据路径，不同估计方式

### PPO：Actor / Critic 的典型分工

PPO 在常见 Actor-Critic 实现中，利用价值函数估计和 GAE 得到 Advantage，并通过 Clipped Surrogate Objective 限制单次策略更新幅度。训练中可能需要 Actor、Critic、Reference Policy 和 Reward Model，但实际组件数量取决于具体实现与任务配方。

对一个采样轨迹，关键字段可包括：

| 字段 | 可能的生产者 | 主要用途 |
| --- | --- | --- |
| Prompt / Response | Dataset / Rollout | 标识生成轨迹 |
| Old Logprob | Rollout 或训练侧重算 | 构造 PPO Importance Ratio |
| Reward | 规则、奖励模型或环境 | 计算回报 |
| Value | Critic | 估计优势与价值损失 |
| Advantage / Return | 算法层 | 更新 Actor/Critic |
| Loss Mask | 数据处理 / 训练侧 | 排除无效或 Padding Token |

例如 PPO 的重要性采样比例可以写作

\[
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}
\]

如果 Old Logprob 与实际生成策略版本不一致，训练的数值含义就会发生变化。**策略版本不是单纯的工程元数据，它与目标函数正确性直接相关。**

### GRPO：组内相对奖励替代 Critic 的价值基线

GRPO 针对相同 Prompt 采样多个 Response，并通过组内奖励分布构造相对优势。常见形式可以写为：

\[
A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}
\]

这是便于理解的典型组内归一化形式；实际实现可能使用不同标准化、Clip、KL 和 Token/Sequence 级归一化选项。

与典型 PPO Actor-Critic 结构相比，GRPO 的一个重要特点是可以不训练独立的 Critic。但它增加了 **同一 Prompt 的分组采样**要求，Rollout 数据和组内索引关系不能任意打散。

### PPO 与 GRPO 的系统差异

| 问题 | 典型 PPO | 典型 GRPO |
| --- | --- | --- |
| Advantage 来源 | Value / GAE | 同 Prompt 多 Response 的相对 Reward |
| Critic | 通常需要 | 典型设计不需要 |
| Rollout 采样 | 轨迹与时间步 | 强调同 Prompt 的 Group |
| 样本元数据 | Old Policy、Value、Mask | Prompt Group、Reward、Mask、Logprobs |
| 系统重点 | Actor/Critic 协作 | 组完整性、变长序列与 Batch Packing |

这只是常见配置对照，不代表每个框架的所有算法变体都严格遵循这一张表。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：PPO 与 GRPO 按同一逻辑步骤排列的流程对照"><img width="1890" height="1651" src="/images/figures/rl-posttraining-frameworks-01.png" alt="PPO 与 GRPO 按同一逻辑步骤排列的流程对照" loading="lazy" /></button>
  <figcaption>蓝色表示共享系统职责，橙色表示典型算法差异；虚线体现可选 Reference 路径与不同循环。</figcaption>
</figure>

图中的“一条 Response / Prompt”只是 PPO 的一个采样配置示例，PPO 同样允许每个 Prompt 采多个 Response。Old Logprob 必须对应实际行为策略；训练侧重算需要匹配其版本与语义。Reward 可以来自规则或环境，不要求独立奖励模型；KL、Reference 与组内标准化也由具体配方决定。

| 流程步骤 | 典型 PPO | 典型 GRPO |
| --- | --- | --- |
| Prompt 与采样 | 从任务分布采轨迹，可每 Prompt 多条 | 同 Prompt 采一组 Response |
| Reward | 逐轨迹、逐 Token 或环境反馈 | 用组内奖励构造相对优势 |
| Old Logprob | 对应实际行为策略 | 同样须匹配生成策略 |
| Reference Logprob | 使用相关 KL 项时需要 | 使用相关 KL 项时需要 |
| Value / Advantage | Critic 估值与 GAE 等方法 | 典型设计以组内奖励替代 Critic |
| Loss / 更新对象 | Actor + Critic | Actor |
| 权重同步 | 按迭代与异步约束更新 Rollout | 同样需要版本管理 |


## 三、为何要区分两个循环？

训练程序中常见两个逻辑循环。

**外层数据循环**：获取 Prompt、生成新轨迹、计算 Reward、完成训练并推进下一轮采样。是否每次必须立刻同步最新 Actor 权重取决于同步/异步策略。

**内层更新循环**：基于当前已经采集的 Experience，按 Mini-batch 和 Epoch 执行一次或多次参数更新。PPO 的多次 Epoch、重要性采样比例和允许的 Policy Drift 会约束这一层。

如果使用异步 Rollout，则可能在 Trainer 更新时继续生成轨迹，外层循环不再具有完全串行的同步边界。这时需要明确 **Staleness、生成策略版本、参数刷新时机和旧数据回收**。

## 四、共置与分离：什么时候适合？

| 维度 | Colocated | Disaggregated |
| --- | --- | --- |
| 资源使用 | 训练/推理可复用相同 GPU | 可以按阶段独立配置 GPU |
| 显存管理 | 常需切换模型状态、Offload/Onload | 组件内存驻留可以更独立 |
| 模型布局 | 可能涉及训推 Resharding | 需要跨资源的权重更新 |
| 数据路径 | 部分情况下本机可达 | 明确存在跨进程/跨节点传输 |
| 端到端效率 | 取决于切换与空闲开销 | 取决于资源利用率和传输同步 |

分离不是天然更高效；共置也不是“简单版的错误方案”。选择需要结合 Model Size、Rollout/Update 耗时比例、硬件拓扑和允许的异步程度。

## 五、常见开源框架的架构取舍

在相关开源生态中，<a href="https://github.com/volcengine/verl">verl</a>、<a href="https://github.com/inclusionAI/AReaL">AReaL</a>、<a href="https://github.com/THUDM/slime">slime</a>、<a href="https://github.com/RLinf/RLinf">RLinf</a> 等框架都在处理 RL 训练、推理生成与资源编排的问题；其他项目如 Relax 与 UniRL 也关注不同的后训练场景。

比较框架时，我更关注五个可检验的问题，而非给项目排一个固定名次：

1. **算法层**：支持哪些 Advantage / Policy Loss，能否扩展 Agentic 或环境反馈？
2. **训练层**：采用 FSDP、Megatron 或其他后端？Checkpoint、TP/PP/EP 如何表达？
3. **生成层**：支持哪些推理后端？Token Generation / Tool Loop 与 Training 如何交接？
4. **执行层**：通过哪些调度机制管理资源、异步队列与 Backpressure？
5. **一致性层**：权重怎样同步、策略版本怎样标记、训推计算的 Logprob 是否可比？

由于这些项目持续演进，不能从一次旧版本阅读推断某框架“永远只支持某后端”或“总是最完整”。实际选型需要检查对应 Release、配置与已合并的源码。

## 六、在线、离线与 Agentic RL 的边界

**Online RL** 通常在训练过程中持续产生与当前或近期策略有关的数据；**Offline RL** 则从已收集的数据中学习，通常不会为每次更新实时执行新的 Policy Rollout。二者的采样分布与策略评估问题不同。

DPO 的偏好学习形式与 PPO/GRPO 的在线 RL Pipeline 也有区别，不宜仅仅把 DPO 看成一条“省略 Reward 模块的 PPO”。

Agentic RL 则在普通文本 Rollout 上增加了工具调用、环境状态、可变 Turn 数、任务级 Reward 和更明显的长尾轨迹。此时数据系统不能只传一个最终 Response；它需要保留轨迹与状态边界，供相应的 Loss 和采样策略使用。


### PPO、GRPO 与离线 DPO 的流程对照

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：在线 PPO GRPO 与离线 DPO 的训练流程对照"><img width="1852" height="1253" src="/images/figures/rl-posttraining-frameworks-03.png" alt="在线 PPO GRPO 与离线 DPO 的训练流程对照" loading="lazy" /></button>
  <figcaption>典型在线 RL 会生成新轨迹；离线 DPO 使用已有偏好对，在给定参考策略下优化偏好目标。</figcaption>
</figure>

图中 DPO 的“不生成”指训练更新不调用在线 Rollout，偏好数据仍需要在此前收集。Reference Logprob 可以预计算，部分变体也采用不同参考构造；它与 PPO 的行为策略 Old Logprob 承担不同角色。

### 扩展到扩散模型：保留数据主干，改变优化目标

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：FlowGRPO Flow-DPO DiffusionNFT 与 Diffusion-DPO 流程对照"><img width="2180" height="1227" src="/images/figures/rl-posttraining-frameworks-04.png" alt="FlowGRPO Flow-DPO DiffusionNFT 与 Diffusion-DPO 流程对照" loading="lazy" /></button>
  <figcaption>扩散模型后训练的概念对照：在线采样、奖励、比值或重构目标与离线偏好优化。图中名称 Flow-DPO 指该图的在线方法，不能与离线 Diffusion-DPO 混同。</figcaption>
</figure>

扩散或 Flow 模型的生成轨迹由去噪步骤构成。使用策略比值的方法需要明确随机采样过程及其转移概率；采用 ratio-free 重构目标的方法则改变损失与旧模型/EMA 的用途。离线偏好优化消费已收集的优劣样本对，不执行同样的在线奖励采集循环。

这些方法可以复用 Prompt 分发、样本交付和参数更新的系统能力，但训练端的时间步抽样、噪声构造、Logprob 或重构损失不同。图中的固定归一化、KL 和 Loss 写法是特定配方的概念表示，算法与具体定义参见 [Flow-GRPO](https://arxiv.org/abs/2505.05470)、[DiffusionNFT](https://arxiv.org/abs/2509.16117) 与 [Diffusion-DPO](https://arxiv.org/abs/2311.12908)。

## 七、总结：先画数据流，再选择训练框架

判断一个 RL 框架是否适合目标任务，应该首先明确**轨迹字段、Policy Version、数据就绪条件和训练/推理资源布局**，再检查算法和执行后端的扩展能力。

同一个 GRPO 公式可以运行于不同的架构中，但不同架构会引入完全不同的工程代价：Batch 等待、模型状态切换、权重传输、数据复制与异步样本陈旧度。这些代价决定了 RL Infra 的真实优化空间。

## 参考资料

[PPO 论文](https://arxiv.org/abs/1707.06347) · [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) · [verl](https://github.com/volcengine/verl) · [AReaL](https://github.com/inclusionAI/AReaL) · [slime](https://github.com/THUDM/slime) · [RLinf](https://github.com/RLinf/RLinf)

文中比较的是典型算法与架构机制，不代表对最新项目功能的穷尽性排名或任何一组硬件上的 Benchmark。
