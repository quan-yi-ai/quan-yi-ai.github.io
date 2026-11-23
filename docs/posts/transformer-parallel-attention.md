---
title: 并行方式
description: Transformer 并行方式与 Attention 实现：从单卡计算到 TP、PP、DP、CP、EP，深入理解 FlashAttention、PagedAttention、Ring Attention 和 NCCL。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">全部文章</a><span>/</span><span>Systems & Computing</span></div>
  <div class="asb-editorial-heading"><span class="asb-pulse"></span>TECHNICAL NOTES <span class="asb-header-divider">/</span> PARALLEL COMPUTING</div>
  <h1 id="并行方式" tabindex="-1">并行方式</h1>
  <p class="asb-story-subtitle">Transformer 并行方式与 Attention 实现</p>
  <p class="asb-story-deck">从单卡的 Attention 与 FFN 计算出发，沿着层、样本、Token、Head、特征和专家等切分轴，逐步理解多卡执行、注意力算法与集合通信。</p>
  <div class="asb-story-meta"><span>人工智能 × 计算系统</span><span class="asb-meta-dot"></span><span>深度技术笔记</span><span class="asb-meta-dot"></span><a href="https://mcn1ijk3pml3.feishu.cn/wiki/Mujfwy0wsi0bjIkbReScJvhKn3M" target="_blank" rel="noopener noreferrer">飞书原文 ↗</a></div>
</div>

<figure class="asb-figure asb-figure--cover">
  <a href="/diagrams/parallel-overview.svg" target="_blank" rel="noopener noreferrer" title="在新标签中查看 SVG 原图">
    <img src="/diagrams/parallel-overview.svg" alt="并行方式的原创矢量概念图：从单卡 Transformer 的 Attention 和 FFN 到 TP、PP、DP、CP、EP、SP 切分轴，再到四张 GPU 的分布式执行" />
  </a>
  <figcaption><span>FIG. 01</span> 从单卡计算到分布式并行的六种切分方式 <a href="/diagrams/parallel-overview.svg" target="_blank" rel="noopener noreferrer" aria-label="放大查看原图">查看 SVG ↗</a></figcaption>
</figure>

本文先建立单卡 Transformer 的计算基准，再介绍切分轴及其通信代价，之后依次讨论 Attention 的片上 IO 优化、跨 GPU Attention 和 KV Cache 管理，最后回到多维并行与 NCCL 通信。

**阅读路线**：`单卡公式 → 张量切分 → 分布式 Attention → KV Cache → 多维并行与通信`。其中，**并行策略**回答“计算与参数如何分配”，**Attention 实现**回答“注意力算子如何高效执行”，**缓存管理**回答“历史 K/V 如何存放与复用”。同一个系统可以同时使用这些不同层面的优化。

## 一　单卡上的 Transformer block

<figure class="asb-figure asb-figure--source"><a href="/images/parallel/figure-1.svg" target="_blank" rel="noopener noreferrer" title="放大查看原文图 1"><img src="/images/parallel/figure-1.svg" alt="飞书原文技术示意图 1" loading="lazy" /></a><figcaption><span>SOURCE FIG. 01</span> 飞书原文技术图 · 点击可查看大图</figcaption></figure>

设 X 的布局为 [B, S, H]。B 是 batch，S 是 token 数，H 是模型宽度，L 是 block 层数，F 是 FFN 中间宽度。Nₕ 是 MHA 的 head 数，Dₕ 是每个 head 的宽度，A = Nₕ · Dₕ 是合并后的 Attention 宽度。线性层的权重写成 [输入宽度, 输出宽度]，批和序列可展平为 M = B · S 行。前向公式省略 bias 和 dropout。

### Attention 的计算公式

$$
\begin{aligned} Q &= X W_Q,\qquad K = X W_K,\qquad V = X W_V \\[6pt] S_h &= \frac{Q_h\,K_h^{\mathsf T}}{\sqrt{D_h}} + \mathrm{Mask} \\[6pt] O_h &= \operatorname{softmax}_{\mathrm{key}}(S_h)\,V_h \\[6pt] Y &= \operatorname{Concat}_{\mathrm{heads}}(O_h)\,W_O \end{aligned}
$$

自注意力先做 Q、K、V 三组投影。MHA 的每组权重为 [H, A]，输出各重排成 [B, S, Nₕ, Dₕ]。逐 head 的分数形状为 [S, S]。因果掩码将未来 key 的分数设为 −∞，softmax 沿 key 轴计算。各 head 输出为 [S, Dₕ]，拼接并做输出投影后回到 [B, S, H]。

### FFN 的计算公式

$$
\begin{aligned} &\text{FFN} \\[3pt] U &= \phi(X\,W_{\mathrm{up}}) \\[4pt] Y &= U\,W_{\mathrm{down}} \\[9pt] &\text{SwiGLU} \\[3pt] U &= \operatorname{SiLU}(X\,W_{\mathrm{gate}}) \odot (X\,W_{\mathrm{up}}) \\[4pt] Y &= U\,W_{\mathrm{down}} \end{aligned}
$$

上投影和下投影的权重形状为 [H, F] 和 [F, H]。展平后激活依次为 [M, H] → [M, F] → [M, H]。SwiGLU 的 gate 和 up 沿同一中间宽度 F 切分，逐元素乘法在本地完成。

常见 pre-norm 残差流写作 X′ = X + Attention(Norm(X))，Y = X′ + FFN(Norm(X′))。Attention 在 token 间混合信息，FFN 在每个 token 内变换特征。

## 二　多卡前向的切分轴与张量形状

T、P、D、C、G 分别表示 TP、PP、DP、CP、EP 的并行度。b 是 microbatch 的样本数。下表每行独立看，使用均匀切分，SP 是 Megatron 中与 TP 配合的用法。形状以每个 rank 为单位。

<div class="asb-table-kicker">原文嵌入表格 1 · 已恢复单元格数据</div>

| 并行方式与切分轴 | 每个 rank 的局部对象 |
| --- | --- |
| DP · batch | [B / D, S, H] |
| PP · block 层号 | 约 L / P 层；阶段边界激活 [b, S, H] |
| TP · head 或特征 | Q、K、V [B, S, Nₕ / T, Dₕ]；FFN 中间 [B, S, F / T] |
| SP · 部分激活的 token | 残差流与 Norm 等位置 [B, S / T, H] |
| CP · 上下文 token | 本地 Q [B, S / C, Nₕ, Dₕ] |
| EP · 专家编号 | 约 E / G 个专家；专家 e 的输入 [Tₑ, H] |

### 并行方式确定分工，具体切法可以变化

并行方式给出分工的对象，具体布局还要看算子和上下游。PP 决定每个阶段放哪些 block；DP 让各模型副本处理不同样本，普通 DDP 复制训练状态，ZeRO 或 FSDP 还可以对训练状态分片。EP 决定专家放在哪些 rank，同一个专家还可以叠加专家 TP，拆分内部 GEMM。

SP（按 Megatron 的用法）通常复用 TP 组，把部分激活按 token 切分，在需要完整输入的线性层前收齐。CP 把上下文 token 分给各 rank。Attention 可以保留本地 Q、交换 K、V，也可以通过 All-to-All 在 token 分片和 head 分片之间重排。

### 从切分轴到通信：Column / Row Parallel GEMM

同一个 GEMM 有两种常见 TP 切法。设 X 的形状为 [M, K]，权重 W 为 [K, N]，TP 组大小为 T。

**按输出特征 N 切（Column Parallel）。** 各 rank 做 [M, K] × [K, N / T] → [M, N / T]，得到不同的输出列。下游可直接消费分片时就保持原布局。只有下游需要完整输出时，才沿输出特征做 AllGather。若上游用了 SP，还可能需要在 GEMM 前沿 token 做 AllGather。这两次 AllGather 的对象和切分轴不同。

**按输入特征 K 切（Row Parallel）。** 各 rank 做 [M, K / T] × [K / T, N] → [M, N]，得到同一组输出位置的部分和。完整结果需要跨 rank 相加。若每个 rank 都要完整结果，用 AllReduce。若下游只保留本 rank 的 token 段，用 ReduceScatter，输出 [M / T, N]。

Dense FFN 通常把这两种切法连用。第一层按输出特征切，本地激活后直接交给第二层，第二层按输入特征切。Attention 的常规 MHA 投影沿用相同思路，Q、K、V 投影按整颗 head 分配输出特征，输出投影再归约各 rank 的部分和。

### PP 和 DP 怎样跨 block

PP 把连续的一组完整 block 放到每个阶段。在未叠加 TP、SP 或 CP 的例子里，阶段之间点对点传递形状 [b, S, H] 的 microbatch 激活。反向沿相反方向传激活梯度。多个 microbatch 可以同时在不同阶段运行。普通 DDP 的每个副本都跑全部 L 层，各自处理不同样本，反向后同步权重梯度。

### TP 在 Attention 里切 head 数

<figure class="asb-figure asb-figure--source"><a href="/images/parallel/figure-2.svg" target="_blank" rel="noopener noreferrer" title="放大查看原文图 2"><img src="/images/parallel/figure-2.svg" alt="飞书原文技术示意图 2" loading="lazy" /></a><figcaption><span>SOURCE FIG. 02</span> 飞书原文技术图 · 点击可查看大图</figcaption></figure>

每个 rank 的 Q、K、V 权重按输出特征列切为 [H, A / T]，都处理完整 S 个 token。MHA 的每张卡负责 Nₕ / T 个完整 head，head 宽度 Dₕ 不变。本地分数形状为 [B, Nₕ / T, S, S]。本地 head 拼成 [M, A / T]，乘本地输出投影权重 [A / T, H] 得到 [M, H] 部分和。纯 TP 用 AllReduce 求和，TP+SP 用 ReduceScatter 求和并把 token 分回各卡。对 GQA/MQA，KV head 数比 Q head 少，KV 如何分布或复制要看实现。

### TP 在 FFN 里切中间特征

第一层每卡做 [M, H] × [H, F / T] → [M, F / T]；各卡产出不同特征列，激活在本地做。第二层每卡做 [M, F / T] × [F / T, H] → [M, H]；各卡产出同一组输出位置的部分和。纯 TP 走 AllReduce，TP+SP 走 ReduceScatter。开启 SP 时，Q、K、V 或 FFN 第一层入口先 AllGather，恢复本线性层所需的 token。

### CP 切 token EP 切专家

CP 中每卡只负责 S / C 个 query token，head 数不因 CP 自身减少。FFN 逐 token 算本地分片。Attention 则跨卡取得可见的 K、V。计算全局分数的逻辑块形状为 [B, Nₕ, S / C, S]，可以逐块计算。

EP 中 E 代表专家总数，Tₑ 代表路由到专家 e 的 token 数。若专家权重完整放在一张卡上，本地专家做 [Tₑ, H] → [Tₑ, Fₑ] → [Tₑ, H]。流程是 Dispatch → 专家 GEMM → Combine。如果同一个专家还做专家 TP，Megatron 的 All-to-All dispatcher 可以在这两次 EP 通信之间，插入专家 GEMM 前的 AllGather 和 GEMM 后的 ReduceScatter。

## 三　MHA、GQA、MLA：模型结构决定 K/V 规模

这些名称描述模型结构。用 n、k 表示 query 和 KV head 数。在均匀分组下，每组 n / k 个 query head 共享一个 KV head。MHA 有 k = n，GQA 有 1 < k < n，MQA 有 k = 1。等宽 K、V、每元素 w 字节时，普通 KV payload 为 2 · k · Dₕ · w 字节每 token 每层，忽略框架副本和分配开销。

MLA（Multi-head Latent Attention）使用低秩潜变量压缩 KV。以 DeepSeek-V2 为例，缓存可保留宽度 Lᵥ 的 KV latent 和宽度 R 的独立 RoPE key 分量。两者精度相同时，对应的 payload 是 (Lᵥ + R) · w。实现可以吸收部分上投影，避免每步都显式展开全部历史 K、V。

## 四　单卡 Attention 的 IO 优化：Online Softmax 与 FlashAttention

### Online Softmax 怎样跨块合并

普通 Softmax。固定一个 batch、head 和 query，令 sᵢ 为该 query 对第 i 个可见 key 的注意力分数，pᵢ 为归一化后的权重。

$$
\begin{aligned} p_i &= \frac{\exp(s_i)}{\sum_j \exp(s_j)} \\[8pt] m &= \max_j s_j, \qquad p_i = \frac{\exp(s_i-m)}{\sum_j \exp(s_j-m)} \end{aligned}
$$

求和沿 key 维进行。分子和分母同乘 exp(−m)，结果相同，并能避免指数上溢。掩码位置的权重为 0。下面的 Online Softmax 把跨所有可见 key 的归一化改成逐块更新。

<figure class="asb-figure asb-figure--source"><a href="/images/parallel/figure-3.svg" target="_blank" rel="noopener noreferrer" title="放大查看原文图 3"><img src="/images/parallel/figure-3.svg" alt="飞书原文技术示意图 3" loading="lazy" /></a><figcaption><span>SOURCE FIG. 03</span> 飞书原文技术图 · 点击可查看大图</figcaption></figure>

$$
\begin{aligned} m' &= \max(m,\; \max s_j) \\[5pt] \alpha &= e^{m-m'} \\[5pt] p_j &= e^{s_j-m'} \\[5pt] \ell' &= \alpha\,\ell + \sum p_j \\[5pt] a' &= \alpha\,a + p_j\,V_j \\[5pt] O &= \frac{a}{\ell} \end{aligned}
$$

固定某个 query 行，设 key 块 j 的有效分数是 sⱼ、对应 value 是 Vⱼ。每行维护运行最大值 m、指数和 ℓ、未归一化的加权和 a，初始为 (−∞, 0, 0)。处理每个有效块时：

把旧累积量乘 α，是为了把旧块和新块的指数权重放在同一个最大值基准下。第一个有效块的旧权重取零。全部有效块处理完后 O = a / ℓ。全掩码块可跳过，全掩码行的输出由实现约定，这里取零。

### FlashAttention 前向伪代码

以下是核心前向的数据流，略去投影层、dropout、反向和硬件调度细节。m、ℓ 按 query 行更新，a 的每行是一个 value 向量。

```python
for Q_i in query_tiles(Q):
    m ← -∞, l ← 0, a ← 0
    for (K_j, V_j) in key_value_tiles(K, V):
        s ← Q_i @ transpose(K_j) / sqrt(D_h) + mask(i, j)
        valid ← rows_with_visible_keys(s)
        # Update valid rows only. Keep masked rows unchanged.
        m_new ← max(m, rowmax(s))
        alpha ← exp(m - m_new), or 0 for first valid tile
        p ← exp(s - m_new)
        l ← alpha * l + rowsum(p)
        a ← alpha * a + p @ V_j
        m ← m_new
    write O_i ← a / l for rows with l > 0, else 0
```

Q、K、V 按 tile 搬到片上，分数和概率就地消费，避免把完整 [S, S] 写入 HBM。密集 Attention 仍有平方级计算。伪代码说明的是分块和归一化原则，实际 kernel 会改变循环顺序、并行方式和流水排程。

<div class="asb-table-kicker">原文嵌入表格 2 · 已恢复单元格数据</div>

| 版本 | 主要改变 |
| --- | --- |
| FA1 | IO 分块与在线归一化，降低 HBM 流量。 |
| FA2 | 改善线程块和 warp 的工作分配，减少非矩阵乘操作。 |
| FA3 | 面向 Hopper，利用 TMA 和 Tensor Core 异步流水，支持 FP8 路径。 |
| FA4 | 面向 Blackwell，联合设计算法和流水，使用指数近似等手段缓解计算单元增长不均衡。 |

### FlashAttention v1 到 v4 分别优化什么

**v1 · 减少显存读写。** 分块加载 Q、K、V，用 Online Softmax 在片上累积结果，避免把整张注意力矩阵写回 HBM。

**v2 · 改善并行和线程分工。** 增加沿序列维的并行度，调整线程块内的 warp 分工，减少非矩阵乘操作和共享内存通信。

**v3 · 面向 Hopper 的异步流水。** 将数据搬运、矩阵乘与 Softmax 交错执行，结合 warp specialization，并提供 FP8 路径。

**v4 · 面向 Blackwell 的流水与归约。** 使用更大的 tile 和异步 MMA，减少指数计算与重缩放开销，并用 Tensor Memory 和双 CTA MMA 降低反向共享内存流量和原子累加开销。

## 五　跨 GPU 的 Attention：Ring、Tree 与 Ulysses

当上下文被拆到多个 GPU 上时，每张卡只能直接访问本地的 K/V，Attention 却需要汇聚所有可见 Key 的贡献。这里先讨论**如何完成跨卡 Attention**，再到第六节单独说明推理时的 KV Cache 存储和复用。

### Ring Attention 是 CP 的一种实现

<figure class="asb-figure asb-figure--source"><a href="/images/parallel/figure-5.svg" target="_blank" rel="noopener noreferrer" title="放大查看原文图 5"><img src="/images/parallel/figure-5.svg" alt="飞书原文技术示意图 5" loading="lazy" /></a><figcaption><span>SOURCE FIG. 05</span> 飞书原文技术图 · 点击可查看大图</figcaption></figure>

设 CP 组有 C 个 rank。rank r 保留本地 Qᵣ，大小是 [B, S / C, Nₕ, Dₕ]，并持有自己的 Kᵣ、Vᵣ。每轮用当前 K、V 块计算局部 Attention，同时把 K、V 转给环上的邻居并接收下一块。mask 决定哪些 query-key 对有效。各 rank 用前面的 Online Softmax 合并各块的贡献，计算结束时只产生自己 query 分片的输出。

**这里的 Ring 不是 Ring AllGather。** Ring Attention 沿设备环传递 K/V 分片，让每个 Query 分片依次访问全局上下文；Ring AllGather 则是集合通信的一种实现算法，传输的可能是线性层输入激活。两者可以使用相似的网络拓扑，但通信对象、计算依赖和最终目的并不一样。

### Tree Attention：跨 GPU 归约 Attention 统计量

Ring Attention 的思路是把 K/V 分片沿环传到需要计算的位置；另一条路线是不搬运整段 K/V，而是让设备分别计算当前 Query 对本地 K/V 分片的贡献，再对结果进行归约。**Tree Attention**（这里指 *Topology-aware Decoding for Long-Context Attention on GPU Clusters*）主要面向分布式长上下文 **decode**，利用树形归约以及分层互连来降低跨设备传输与同步开销。

假设各 GPU 持有不同的 K/V 分片，而本次解码的 Query 可以在参与计算的设备上获得。GPU `r` 对其本地可见的 Key 计算三个统计量：局部最大值 `m_r`、以该最大值为基准的指数和 `l_r`，以及尚未归一化的加权向量 `a_r`。这是前文 Online Softmax 的同一套数学量。两组统计量 `A`、`B` 可以按以下公式合并：

$$
\begin{aligned}
m &= \max(m_A,m_B) \\
\ell &= e^{m_A-m}\ell_A + e^{m_B-m}\ell_B \\
a &= e^{m_A-m}a_A + e^{m_B-m}a_B \\
O &= a/\ell
\end{aligned}
$$

这使得 Attention 结果能够按树形结构逐层归约：设备不必传输完整的历史 K/V，只需交换与当前 Query 对应的归约统计量（具体实现还需计入 Query 的分发等开销）。归约时仍需处理因果掩码、数值稳定性和全掩码行等情况。

<figure class="asb-figure asb-figure--cover">
<a href="/diagrams/ring-vs-tree-attention.svg" target="_blank" rel="noopener noreferrer" title="在新标签中查看 SVG 原图"><img src="/diagrams/ring-vs-tree-attention.svg" alt="Ring Attention 通过环形传递 KV 分片；Tree Attention 计算局部统计量后执行树形归约的原创矢量对比图" loading="lazy" /></a>
<figcaption><span>FIG. 07</span> Ring 与 Tree Attention 的通信思路对比 <a href="/diagrams/ring-vs-tree-attention.svg" target="_blank" rel="noopener noreferrer">查看 SVG ↗</a></figcaption>
</figure>

两者不能简单判断“哪个总是更快”：Ring Attention 的长序列训练 / prefill 路径与 Tree Attention 重点优化的长上下文 decode 路径并不完全相同，性能也受互连拓扑、序列长度、批大小和实现细节影响。

参考：[Tree Attention 原论文](https://arxiv.org/abs/2408.04093) · [代码实现](https://github.com/Zyphra/tree_attention)

### Ulysses：通过 All-to-All 重排序列与 Head

还有一种不同的分布式 Attention 思路：**DeepSpeed Ulysses** 在设备间进行 All-to-All，使数据由序列维切分转换为 Head 维切分；每张卡随后对自己的 Head 计算 Attention，再进行反向的 All-to-All 还原布局。

可以这样归纳：**Ring 交换 K/V 块，Tree 归约 Attention 统计量，Ulysses 重排切分布局。** 它们解决的是跨设备获取全局上下文的问题，但具体通信方式不同。实际系统还可以组合多种方法。

参考：[DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)

::: tip 同名概念：Tree-structured Attention
上面的 Tree Attention 是一种**跨设备的精确 Attention 归约算法**。在推测解码、多分支推理或 Agent RL 中，“Tree Attention”还可能指根据**共享前缀 / 树状生成路径**复用计算的技术，例如 [DeFT](https://arxiv.org/abs/2404.00242) 和 [AREAL-DTA](https://arxiv.org/abs/2602.00482)。后者的“Tree”来自输入序列的分叉关系，而不是 GPU 归约拓扑。两类算法不能混为一谈。
:::

## 六　推理 KV Cache 的组织与分层管理

前面的 Ring、Tree 和 Ulysses 讨论**跨设备怎样完成 Attention 计算**。推理服务还会遇到另一个独立问题：长时间运行的请求怎样存放、查找、复用并迁移历史 K/V。这里转到 KV Cache 的存储布局和缓存层级。

### PagedAttention 管理推理 KV Cache

<figure class="asb-figure asb-figure--source"><a href="/images/parallel/figure-4.svg" target="_blank" rel="noopener noreferrer" title="放大查看原文图 4"><img src="/images/parallel/figure-4.svg" alt="飞书原文技术示意图 4" loading="lazy" /></a><figcaption><span>SOURCE FIG. 04</span> 飞书原文技术图 · 点击可查看大图</figcaption></figure>

将你提到的“配置的 Attention”按 PagedAttention 理解。若每个 KV block 容纳 J 个 token，逻辑位置 t 的块号是 floor(t/J)，块内偏移是 t mod J。块表把逻辑块号映射到物理块，物理块可不连续。请求增长时按需申请，支持前缀共享，减少碎片和预留浪费。每个有效 token 的 KV payload 由模型结构和精度决定。

### LMCache、Mooncake 与 HiCache：缓存管理的不同层次

<div class="asb-table-kicker">三种 KV Cache 相关方案 · 原文表格</div>

| 项目 | 是什么 | 层级 |
| --- | --- | --- |
| LMCache | 独立的 KV Cache 管理层，挂在 vLLM/SGLang 外面 | 引擎外置的"存储中间件" |
| Mooncake | 以 KVCache 为中心的分离式推理架构（PD 分离 + 分布式缓存池） | 集群级架构 |
| HiCache | SGLang 内置的三级缓存模块 | 引擎内部机制 |

**LMCache：引擎外置的 KV Cache 管理层。** 它关注 KV Cache 跨请求的复用、卸载和跨存储层迁移。常见层级包括 GPU HBM（活跃工作集）、CPU DRAM（热缓存）、本地 SSD / NVMe，以及远端存储。前缀复用、缓存卸载和 Prefill/Decode 间的传输是相关但不同的使用方式。

**Mooncake：面向 Prefill/Decode 分离的分布式 KV Cache 架构。** 它利用集群的 CPU 内存、存储设备与网络资源，跨 Prefill 和 Decode 阶段传输、管理 KV Cache，从系统级减少重复计算和资源闲置。

**HiCache：推理引擎内的分层缓存机制。** 可以按 GPU HBM、CPU 主机内存和外部 / 分布式存储的层级理解。它与 PagedAttention 不是互斥方案：前者关注缓存层次与迁移，后者关注 KV 的逻辑块和物理块映射。

参考资料

[Megatron Core 并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) · [Megatron Core 上下文并行](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) · [Megatron Core MoE Dispatcher](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html)

[GQA](https://arxiv.org/abs/2305.13245) · [DeepSeek-V2 与 MLA](https://arxiv.org/abs/2405.04434) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [FA2](https://arxiv.org/abs/2307.08691) · [FA3](https://arxiv.org/abs/2407.08608) · [FA4](https://arxiv.org/abs/2603.05451)

[PagedAttention](https://arxiv.org/abs/2309.06180) · [Ring Attention](https://arxiv.org/abs/2310.01889) · [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)

## 七　多维并行组合与计算通信重叠

单个并行轴及其通信方式弄清楚之后，才能理解多个轴的组合。常见 3D 组合是 DP×TP×PP。加入 CP 的独立轴配置，world size 可以写成 D·T·P·C。固定其他坐标，只沿 tp 坐标变化得到 TP 组，只沿 cp 坐标变化得到 CP 组。MoE 项目可能把 EP 也计入“几 D”，具体以该框架的进程组为准。SP 通常复用 TP 组，EP 与普通 DP、专家 DP 和专家 TP 的排组需要单独核对。

回到通算融合，入口是 **AllGather → GEMM**，通信先产出可读的输入块。出口是 **GEMM → ReduceScatter 或 AllReduce**，计算先产出部分和。数据布局决定箭头方向，块级实现决定通信和另一块的计算能否同时进行。MoE 可以重叠 Dispatch/Combine 与专家 GEMM，Ring Attention 可以重叠 K、V 传输与 Attention tile 计算。

### Ring 集合通信的每 rank 通信量

设通信组有 R 个 rank，完整逻辑张量大小为 U 字节，均匀分成 R 块。以下计量每个 rank 发出的 payload，接收量相同。若统计收发总量，需要再乘以 2。

**Ring AllGather**：单 rank 输入 U/R，输出 U。每轮发送 U/R，共 R−1 轮，发送量为 (R−1)·U/R。

**Ring ReduceScatter**：单 rank 输入 U，归约后保留 U/R。每轮发送 U/R，共 R−1 轮，发送量为 (R−1)·U/R。

**Ring AllReduce**：单 rank 输入和输出都是 U。先做 ReduceScatter，再做 AllGather，发送量为 2·(R−1)·U/R。

对应 TP+SP 的 AllGather，U 是收齐后的输入激活大小。对于 GEMM 后的 ReduceScatter 或 AllReduce，U 是每 rank 的完整部分和张量大小。这两处的 U 要分别按张量形状计算。


## 八　NCCL 集合通信：从拓扑到数据传输

以下保留原文整理的关键环节，帮助把前面 AllGather、ReduceScatter、AllReduce 的逻辑通信映射到实际执行。**注意：这不是一次 AllReduce 必定按顺序执行的七个固定阶段**。初始化、拓扑发现、协议选择、通道组织、GPU Kernel 与网络传输在不同硬件和协议下的执行路径可能不同。

**原文整理的 NCCL 环节（含待核对项）**

| 阶段 | 核心操作 | 备注 |
| --- | --- | --- |
| 初始化（建链） | 初始化通信域及连接资源 | 原文“TCP 单节点？”有待进一步核对 |
| 拓扑发现 | 发现 GPU 和网络互连拓扑 | NVLink、PCIe、InfiniBand 等 |
| Ring/Tree 等通信组织 | 根据算法和拓扑组织通信关系 | 具体选择因规模与硬件而异 |
| channel分配 | 数据切chunk，多路并行传 | 每个chunk走一个channel |
| Proxy 线程 | 某些通信协议或网络路径需要 CPU 端 Proxy 协作 | 并非每个 Channel 都必然有独立线程 |
| QP 建立 | RDMA 网络路径可能需要建立 Queue Pair | 单机 NVLink/PCIe 路径不一定涉及 QP |
| 数据传输 | 通过选定协议与互连传输数据 | RDMA 只是可能的传输方式之一 |
