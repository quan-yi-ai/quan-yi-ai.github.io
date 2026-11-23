---
title: 投机解码：从无损验证到 EAGLE、MTP 与并行草稿
description: 理解 Speculative Decoding 的草稿模型、接受-拒绝校验和概率分布保持机制，再比较 N-gram、Medusa、MTP、EAGLE、DFlash 与 DSpark 的设计取舍。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="speculative-decoding" tabindex="-1">投机解码：从无损验证到 EAGLE、MTP 与并行草稿</h1>
  <p class="asb-story-subtitle">加速自回归解码，为什么不一定改变目标模型的输出分布？</p>
  <p class="asb-story-deck">投机解码将“串行生成下一个 Token”分成草稿提议和目标模型批量验证两个阶段。其性能取决于草稿质量、提议成本和一次验证可接受多少 Token；其正确性则取决于验证算法，而不是草稿模型有多接近目标模型。</p>
</div>

自回归模型每次 Decode 需要以前面的 Token 为条件预测下一个 Token。对于小 Batch 或 Memory-Bound 的解码负载，重复发起目标模型 Forward 可能无法充分利用 GPU 的计算吞吐。**投机解码（Speculative Decoding）** 尝试以更小的成本预测一组候选 Token，再让完整目标模型一次验证多个位置，以减少昂贵的目标模型串行调用次数。

## 一、草稿与验证：加速的基本结构

设目标模型的下一个 Token 分布为 \(p(\cdot)\)，轻量草稿模型为 \(q(\cdot)\)。Draft 阶段先按 \(q\) 提出若干候选 Token；Target 阶段基于这些候选的前缀并行计算各位置的目标模型分布；Verification 阶段决定哪些候选可以接受，在哪里需要重新采样。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大投机解码 Draft Verify 流程"><img src="/diagrams/speculative-verify.svg" alt="轻量 Draft 提议 Token，目标模型并行验证并决定接受或拒绝" /></button>
  <figcaption>草稿 Token 是候选，不是直接交付给用户的输出；验证规则决定最终概率语义。</figcaption>
</figure>

一个常见误区是以为“Draft 生成的 Token 越准确，投机解码就越无损”。实际上，**无损性由接受/拒绝与校正采样规则保证**；Draft 分布越接近目标分布，通常只是提高平均接受长度和性能。

## 二、无损性：为什么 Draft 分布不同也能保持目标分布？

以标准随机采样的一个候选 Token \(x\sim q\) 为例，经典 Speculative Sampling 会以

\[
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
\]

的概率接受候选。当拒绝时，从适当归一化的剩余概率质量中重新采样：

\[
r(x)\propto\max(p(x)-q(x),0)
\]

在满足相应数学假设和实现语义时，这一接受-拒绝规则使最终 Token 分布保持为目标分布 \(p\)，而不是 Draft 分布 \(q\)。

这个结论有几个前提：目标与草稿分布使用兼容的 Token / Vocabulary 语义，验证时能取得相应的概率信息，并按照正确的剩余质量分布采样。不能将简单的 “Draft 与 Target Argmax 相同就接受，否则替换” 直接当作任意随机采样配置下的无损算法。

在 Greedy Decoding 情况下，可以用与目标模型 Greedy Token 的匹配来验证候选，但这属于不同于随机采样校正的执行语义。

## 三、性能模型：为什么越长的 Draft 不一定越快？

假设一次 Draft 提议 \(\gamma\) 个 Token，平均最终接受的有效 Token 数为 \(A\)，草稿成本为 \(T_d\)，目标验证成本为 \(T_v\)。可以用一个极简模型分析：

\[
\text{Effective Throughput}\approx\frac{A}{T_d+T_v+T_{\mathrm{overhead}}}
\]

这里 \(A\) 应按照实现中实际确认的 Token 数计算，可能包括拒绝后的校正采样或额外目标 Token，因此不能不加说明地设为 \(\gamma\)。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大投机解码性能取舍图"><img src="/diagrams/speculative-tradeoff.svg" alt="Draft 长度、接受长度、目标验证成本与实际加速收益关系" loading="lazy" /></button>
  <figcaption>Draft 成本、接受长度与目标验证成本共同决定收益，不能只根据接受率比较方法。</figcaption>
</figure>

增大 Draft Length 会增加候选数量，但如果后续位置的概率预测质量变差，拒绝可能更早出现。即使目标模型一次计算了多个位置，也需要支付相应的 KV Cache、Attention 和验证成本。因此参数需要随着模型、硬件、Batch 和采样设置变化而调节。

## 四、不同草稿器的设计空间

主要草稿器方法可以沿两条维度比较：**草稿 Token 之间是否显式建模依赖**，以及**一次草稿计算能产生多少候选**。

| 方案 | 主要思路 | 典型取舍 |
| --- | --- | --- |
| N-gram | 从历史上下文寻找候选 Token 序列 | 无须专门训练 Draft Model，但命中依赖上下文重复性 |
| 独立 Draft Model | 小模型串行提议 Token | 依赖建模较自然，增加逐 Token 草稿成本 |
| Medusa | 在目标模型上附加预测未来 Token 的多个 Head，并进行候选验证 | 减少独立模型需求，但树状候选与验证仍有实现成本 |
| MTP | 利用模型原生的 Multi-Token Prediction 能力 | 支持程度取决于模型架构与训练配置 |
| EAGLE 系列 | 利用轻量 Draft 网络预测后续特征/Token | 更强的条件依赖，可能有多步 Draft 开销 |
| DFlash | 倾向于块内并行预测 Draft Token | 减少草稿串行步数，但远端位置质量会影响接受长度 |
| DSpark | 在并行 Draft 基础上加入位置依赖修正与置信度估计 | 在质量和草稿开销之间增加新控制维度 |

这些是**机制层面的定位**，不是固定的性能排行榜。不同版本名称（如 EAGLE-3、DFlash2、DSpark 等）对应的公开实现也会继续变化。

### N-gram：为什么无需额外 Draft 网络？

当模型上下文中存在较强的重复片段时，可以复用已经出现过的后续 Token 作为候选。这类方法通常实现成本较低，但对于全新内容或低重复性任务，能够接受的候选长度可能有限。

### Medusa 与 MTP：预测未来位置的 Token

多头/多 Token 预测通过模型自身或扩展模块同时产生未来位置的候选。它们减少了独立 Draft Model 的部署负担，但候选之间的依赖与目标模型验证仍需要正确处理。

### EAGLE：条件依赖与草稿质量

EAGLE 系列通过额外的轻量模块生成未来预测，在保持目标模型主干不变的条件下提高候选质量。其效果不仅取决于 Draft 的单 Token Accuracy，还与候选路径构造、树形验证和推理 Kernel 实现有关。

### DFlash / DSpark：并行起草与依赖修正

DFlash 类方案尝试一次并行产生一个候选块，减少草稿侧串行 Forward 的次数；但块内后续位置若无法充分依赖前序 Token，接受率可能下降。

DSpark 在此类并行 Draft 骨架上增加 Markov 风格的依赖修正和用于选择验证预算的置信度信息。对应的公开实现可参考 [Speculators DSpark 文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/)。

这条演进思路并不意味着“串行 Draft 已经被取代”。最优选择仍由 Draft Cost、Target Verification Cost 与请求负载决定。

## 五、工程实现还必须处理哪些问题？

即使验证算法在数学上保持无损，生产实现仍可能因为执行细节出现问题。

**KV Cache 的归属与回滚。** Draft / Target 为被拒绝的候选预先计算的 Cache，可能需要按照最终已确认的 Token 前缀提交、回滚或重用。

**Sampling 参数一致性。** Temperature、Top-p、Top-k 等设置改变目标分布。如果 Draft 与 Verify 对最终目标分布的解释不一致，不能直接沿用最简单的无损证明。

**Batch 与动态请求。** 高吞吐服务中，多个请求可能处于不同的 Draft/Verify 步数，调度、内存预留与 Kernel Launch 需要按请求状态管理。

**准确统计性能。** 除 Draft Acceptance Rate，还应测量 Accepted Tokens per Target Forward、Draft Latency、Verification Latency、吞吐与尾延迟，以及显存峰值。

## 六、什么时候投机解码会失效或收益很小？

| 现象 | 可能原因 |
| --- | --- |
| 接受率很高但整体不快 | 草稿网络开销过高，或额外的调度/验证开销占主导 |
| Draft 长度增加却更慢 | 提议成本上升，后续位置拒绝率高 |
| 高 QPS 下收益下降 | Target Batch 已充分利用 GPU，验证不再足够便宜 |
| 输出与目标采样分布不符 | 校正采样或 Sampling 参数处理不正确 |
| 内存占用明显增加 | 多候选 KV、验证 Workspace 与图缓存开销 |

因此，投机解码并不属于“开启后总会加速”的通用开关。真正适用的场景要通过目标模型、硬件、请求长度、QPS 和 Sampling 设置的组合 Benchmark 决定。

## 七、总结：无损性与性能是两个独立问题

投机解码的**正确性目标**是保持目标模型的输出语义；**性能目标**是用更廉价的 Draft 换取更少的 Target 串行步骤。前者靠验证算法，后者靠接受长度与成本结构。

从 N-gram、Medusa、MTP 到 EAGLE、DFlash 和 DSpark，可以看到草稿算法始终在权衡条件依赖强度、一次产生的候选数以及额外计算成本。不存在脱离工作负载的固定最优草稿方法。

## 参考资料

[Speculative Decoding 原始论文](https://arxiv.org/abs/2211.17192) · [vLLM Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) · [Speculators Algorithms](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/) · [DSpark](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) · [EAGLE](https://arxiv.org/abs/2401.15077)

各方法的公开实现、性能与无损采样支持范围依赖具体版本和验证算法。本文没有给出未经实测的加速数字。
