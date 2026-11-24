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

设目标模型的下一个 Token 分布为 $p(\cdot)$，轻量草稿模型为 $q(\cdot)$。Draft 阶段先按 $q$ 提出若干候选 Token；Target 阶段基于这些候选的前缀并行计算各位置的目标模型分布；Verification 阶段决定哪些候选可以接受，在哪里需要重新采样。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大投机解码 Draft Verify 流程"><img width="960" height="600" src="/diagrams/speculative-verify.svg" alt="轻量 Draft 提议 Token，目标模型并行验证并决定接受或拒绝" /></button>
  <figcaption>草稿 Token 是候选，不是直接交付给用户的输出；验证规则决定最终概率语义。</figcaption>
</figure>

一个常见误区是以为“Draft 生成的 Token 越准确，投机解码就越无损”。实际上，**无损性由接受/拒绝与校正采样规则保证**；Draft 分布越接近目标分布，通常只是提高平均接受长度和性能。

## 二、无损性：为什么 Draft 分布不同也能保持目标分布？

以标准随机采样的一个候选 Token $x\sim q$ 为例，经典 Speculative Sampling 会以


$
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$


的概率接受候选。当拒绝时，从适当归一化的剩余概率质量中重新采样：


$
r(x)\propto\max(p(x)-q(x),0)
$


接受路径为 token x 提供的概率质量是 min(p(x),q(x))；拒绝后的校正补回剩余质量。两者相加等于 p(x)。这个结论针对保持分布的验证算法；仅仅并行计算目标模型的分数，还不足以证明无损。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：接受拒绝采样保持目标分布的手写推导"><img width="1536" height="2048" src="/images/figures/speculative-decoding-guide-01.jpg" alt="接受拒绝采样保持目标分布的手写推导" loading="lazy" /></button>
  <figcaption>单 Token 的接受质量为 min(p(t), q(t))，拒绝后的校正路径补回 max(0, p(t)−q(t))。</figcaption>
</figure>

多 token 验证沿候选前缀依次决定接受情况：首个拒绝位置输出校正 token，后续草稿作废；全部候选被接受时，可以再从目标分布产生一个额外 token。验证模型虽然同时计算了多个位置，但较后位置的分数以对应草稿前缀为条件，所以不能跳过中间拒绝的位置继续提交。

保持分布也不要求在相同随机种子下逐 token 输出同一序列，因为随机数的消费顺序可能变化。Greedy 对齐与随机采样的分布保持应分别验证。

这个结论有几个前提：目标与草稿分布使用兼容的 Token / Vocabulary 语义，验证时能取得相应的概率信息，并按照正确的剩余质量分布采样。不能将简单的 “Draft 与 Target Argmax 相同就接受，否则替换” 直接当作任意随机采样配置下的无损算法。

在 Greedy Decoding 情况下，可以用与目标模型 Greedy Token 的匹配来验证候选，但这属于不同于随机采样校正的执行语义。

## 三、性能模型：为什么越长的 Draft 不一定越快？

假设一次 Draft 提议 $\gamma$ 个 Token，每轮平均最终提交的 Token 数为 $A$，草稿成本为 $T_d$，目标验证成本为 $T_v$。可以用一个极简模型分析：


$
\text{Effective Throughput}\approx\frac{A}{T_d+T_v+T_{\mathrm{overhead}}}
$


这里 A 包括已接受草稿，以及拒绝后的校正 token 或全接受时的额外目标 token；它与“接受了几个草稿”是两个统计口径。在不提前结束且每轮生成额外 token 的标准链式算法中，一轮提交量最多为 gamma+1。

若单 token 普通解码耗时为 T_base，则每 token 平均耗时约为 (T_d+T_v+T_overhead)/A，加速比约为 A T_base/(T_d+T_v+T_overhead)。例如基线为 10 ms/token，一轮草稿、验证和管理共耗时 15 ms、提交 3 个 token，则约为 5 ms/token。这是成本模型示例。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大投机解码性能取舍图"><img width="960" height="620" src="/diagrams/speculative-tradeoff.svg" alt="Draft 长度、接受长度、目标验证成本与实际加速收益关系" loading="lazy" /></button>
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

### Lookahead：Jacobi 迭代与候选池

[Lookahead Decoding](https://arxiv.org/abs/2402.02057)使用目标模型的并行迭代轨迹收集 n-gram，再验证候选。它不需要另一个草稿网络，但仍需神经网络计算；与直接从已有文本检索 n-gram 的路径不同。

### Medusa：共享上下文，多头预测

Medusa 在共享隐藏状态上增加多个未来位置预测头。各头共享上下文，但预测较远位置时没有像自回归草稿那样显式消费刚采样的前一候选。例如上下文允许“of course”或“no problem”时，独立位置预测可能组合出不协调的续写。

候选按选定树形组织，而不必穷举全部笛卡尔积。Tree Attention 让每个节点只读取自己的祖先；验证后提交满足规则的连续前缀。Medusa 的 typical acceptance 属于另一种接受策略，生成质量相近与严格保持随机采样分布需要分别讨论。参见[Medusa 论文](https://arxiv.org/abs/2401.10774)。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：Medusa 共享主干隐藏状态和多个预测头"><img width="748" height="616" src="/images/figures/speculative-decoding-guide-02.png" alt="Medusa 共享主干隐藏状态和多个预测头" loading="lazy" /></button>
  <figcaption>Medusa 多头预测与候选组合。来源：<a href="https://github.com/FasterDecoding/Medusa">Medusa 项目</a>。</figcaption>
</figure>

### MTP：把前一预测深度的表示接入下一深度

MTP 是多 token 预测这一类目标与结构的统称。以 [DeepSeek-V3](https://arxiv.org/html/2412.19437v2#S2.SS2)为例，主干位置 i 的表示预测 t_(i+1)，第 k 个额外模块组合上一深度表示和 t_(i+k) 的嵌入，预测 t_(i+k+1)：

$$
z_i^k=M_k[\operatorname{RMSNorm}(h_i^{k-1});\operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))],
\qquad h^k=\operatorname{TRM}_k(z^k).
$$


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：DeepSeek-V3 MTP 模块的深度与 Token 索引"><img width="960" height="590" src="/diagrams/speculative-mtp-depth.svg" alt="DeepSeek-V3 MTP 模块的深度与 Token 索引" loading="lazy" /></button>
  <figcaption>固定位置 i，额外预测深度 k 将目标推进到 t_(i+k+1)。依据 <a href="https://arxiv.org/html/2412.19437v2#S2.SS2">DeepSeek-V3 §2.2</a>。</figcaption>
</figure>

嵌入与输出头和主模型共享。训练使用真实序列，起草时使用候选，因此可能存在分布偏移；影响大小要结合训练方法与实际接受长度衡量。较长草稿还受模块容量、顺序执行成本和验证预算制约，不存在通用的固定长度上限。

### EAGLE：带 token 条件的特征预测

原始 [EAGLE](https://arxiv.org/abs/2401.15077)通过轻量模块预测目标模型特征，并结合向前错开一步的 token 信息缓解特征预测的不确定性。描述输入层和 decoder 层时需要限定到具体版本；后续版本对特征选择、训练和候选构造有变化。

它的取舍是以轻量的顺序草稿计算换取更好的条件依赖。更长候选需要更多草稿工作，但延迟还受树形结构、批处理和硬件影响。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：EAGLE 论文比较四种草稿生成机制"><img width="1624" height="734" src="/images/figures/speculative-decoding-guide-04.png" alt="EAGLE 论文比较四种草稿生成机制" loading="lazy" /></button>
  <figcaption>Speculative Sampling、Lookahead、Medusa 与原始 EAGLE 的机制对照。图中附带的 Lookahead 注释应结合前文的 Jacobi 迭代理解；它仍运行目标网络。来源：<a href="https://arxiv.org/html/2401.15077v3#S3">EAGLE 原论文</a>。</figcaption>
</figure>

以已确认前缀“I love”为例：目标模型已有末尾特征并采样出 cats，EAGLE 将 cats 的嵌入与上一位置特征对齐，预测 cats 所在位置的特征，再提出 eat。条件 Token 与特征错开一步，才能把实际采样分支的信息传给草稿器。

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：EAGLE 偏移 Token 与预测特征的对应关系"><img width="960" height="600" src="/diagrams/speculative-eagle-shift.svg" alt="EAGLE 偏移 Token 与预测特征的对应关系" loading="lazy" /></button>
  <figcaption>原始 EAGLE 的偏移条件与两步草稿示例；一条分支只展示索引关系，实际实现可生成树形候选。</figcaption>
</figure>

### DFlash / DSpark：并行主干与轻量位置依赖

[DFlash](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)以目标隐藏状态和 mask 位置为条件，一次草稿前向预测候选块。草稿器内部允许非因果注意力，目标验证仍遵循自己的因果语义。mask 占位符不等于已经得到前一位置的真实候选值，块内位置相关性仍是重要问题。

[DSpark](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/)增加低秩 Markov 修正和接受概率预测头，将前一候选的身份加入后续位置，并为验证长度提供信号。神经网络主干并行起草，位置修正仍可能带有轻量顺序依赖。自适应验证预算也已有 [Sequoia](https://arxiv.org/abs/2402.12374)等先前工作；不同方法的价值应结合自身机制分析。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：DSpark 并行块串行修正与前缀调度"><img width="1624" height="1172" src="/images/figures/speculative-decoding-guide-06.png" alt="DSpark 并行块串行修正与前缀调度" loading="lazy" /></button>
  <figcaption>并行块提供初始 Logits，轻量顺序修正建立位置依赖，调度器选择送入目标模型的候选前缀。来源：<a href="https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/">DSpark / Speculators</a>。</figcaption>
</figure>

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

各方法的采样语义由对应验证算法决定，性能则需在相同模型、硬件与负载下比较。
