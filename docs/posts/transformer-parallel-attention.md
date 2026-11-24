---
title: 并行方式
description: 从单卡 Transformer 的张量形状出发，推导 TP、DP、PP、SP、CP、EP 的切分与通信，以 8 GPU 示例串联 Ring/Tree Attention 和 NCCL 通信成本。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="并行方式" tabindex="-1">并行方式</h1>
  <p class="asb-story-subtitle">Transformer 并行方式与 Attention 实现</p>
  <p class="asb-story-deck">如何把一个 Transformer 拆到多张 GPU 上？沿着每个张量的切分轴，理解 TP、DP、PP、SP、CP、EP 的计算分工、跨卡依赖和通信代价。</p>
</div>

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：并行方式的原创矢量概念图：从单卡 Transformer 的 Attention 和 FFN 到 TP、PP、DP、CP、EP、SP 切分轴，再到四张 GPU 的分布式执行"><img width="1100" height="420" src="/diagrams/parallel-overview.svg" alt="并行方式的原创矢量概念图：从单卡 Transformer 的 Attention 和 FFN 到 TP、PP、DP、CP、EP、SP 切分轴，再到四张 GPU 的分布式执行" /></button>
  <figcaption>从单卡计算到分布式并行的六种切分方式</figcaption>
</figure>

模型并行需要明确以下问题：**每张 GPU 保留什么，独立计算什么，在哪个算子边界必须交换什么数据？** 理解这一点，比单纯记住 TP / PP / DP 等缩写更重要。

全文使用同一组记号和一个 8 GPU 示例，从单卡公式开始推导切分、通信和进程组；再讨论长上下文 Attention 为何需要跨设备交换 K/V 或归约 Softmax 状态。FlashAttention 演进及 KV Cache 分层管理属于相关但不同的问题，放在文末延伸部分。

## 一　Transformer 并行的目标与切分维度

单卡运行 Transformer，至少要容纳模型权重、激活、KV Cache；训练还涉及梯度与优化器状态。即使显存足够，单卡算力、内存带宽或上下文长度也可能成为瓶颈。

不同并行技术分担的对象并不相同：

| 约束或目标 | 优先考虑 | 核心切分与新增依赖 |
| --- | --- | --- |
| 增加样本吞吐 | DP | 每个副本处理不同样本；训练时同步梯度或分片状态 |
| 单层权重 / GEMM 太大 | TP | 按 Head 或矩阵特征切；产生归约或收集通信 |
| 模型层数太多 | PP | 按层切；跨 Stage 传激活与梯度 |
| TP 组中激活重复占用显存 | SP | 在部分算子之间切 Token；必要时 AllGather / ReduceScatter |
| 长上下文无法在单卡高效计算 | CP | 切上下文 Token；Attention 需要跨卡访问 K/V 的贡献 |
| MoE 专家太多 | EP | 按专家分配；需要 Token Dispatch / Combine |

这些是**逻辑上的计算与数据分工**，不等于 NCCL 必须采用何种 Ring / Tree 通信算法。DP、TP、PP、CP、EP 在条件合适时可以组合；SP 在 Megatron 常见实现中通常复用 TP 组，并非总是新增独立维度。

### 八 GPU 组合并行示例

假设一个用于**讲解张量形状**的稠密 Transformer：

| 参数 | 示例值 |
| --- | --- |
| GPU 总数 | 8 |
| Transformer 层数 | 24 |
| 隐藏维度 \(H\) | 4096 |
| Attention Head 数 \(N_h\) / Head 宽度 \(D_h\) | 32 / 128 |
| FFN 中间维度 \(F\) | 11008 |
| 序列长度 \(S\) | 4096 |
| 逻辑并行配置 | DP = 2，TP = 2，PP = 2 |
| 每个 DP 副本的 Microbatch \(b\) | 2 |

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：8 GPU DP × TP × PP 示例中的进程组与流水阶段"><img width="1120" height="540" src="/diagrams/parallel-8gpu-layout.svg" alt="8 GPU DP × TP × PP 示例中的进程组与流水阶段" loading="lazy" /></button>
  <figcaption>8 GPU 逻辑并行拓扑：两个 DP 副本，每个副本划分两个 PP Stage，每个 Stage 包含一个两卡 TP 组</figcaption>
</figure>

对于这个示例，每个 PP Stage 负责约 12 层；每个 TP Rank 负责 16 个 Attention Head、5504 个 FFN 中间特征。DP 两个副本处理不同的样本。

该配置用于推导张量形状与通信量，并非某款模型的推荐部署。示例省略词嵌入、负载不均、优化器分片、显存余量和流水线气泡等因素。

## 二　单卡 Transformer 的计算与张量布局

<figure class="asb-figure asb-figure--source">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：单卡 Transformer Attention 与 FFN 前向数据流"><img width="1080" height="430" src="/images/parallel/figure-1.svg" alt="单卡 Transformer 的 Attention 与 FFN 前向计算流程" loading="lazy" /></button>
  <figcaption>单卡 Transformer 的 Attention 与 FFN 计算流程</figcaption>
</figure>

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

在这个计算图中，**Attention 沿 Token 混合信息，FFN 沿特征变换表示**。因此，对 FFN/投影层进行特征分片，与对上下文 Token 进行分片，产生的通信依赖完全不同。接下来先研究最基本的矩阵切法。

## 三　张量并行：GEMM 切分与通信依赖

TP 的核心不是“每张卡算一半”这么简单，而是要问：**两张卡产出的结果是彼此不同的列，还是相同输出位置上的部分和？** 前者可能直接供后续算子消费；后者最终必须归约。

### Column Parallel 与 Row Parallel GEMM

同一个 GEMM 有两种常见 TP 切法。设 X 的形状为 [M, K]，权重 W 为 [K, N]，TP 组大小为 T。

**按输出特征 N 切（Column Parallel）。** 各 rank 做 [M, K] × [K, N / T] → [M, N / T]，得到不同的输出列。下游可直接消费分片时就保持原布局。只有下游需要完整输出时，才沿输出特征做 AllGather。若上游用了 SP，还可能需要在 GEMM 前沿 token 做 AllGather。这两次 AllGather 的对象和切分轴不同。

**按输入特征 K 切（Row Parallel）。** 各 rank 做 [M, K / T] × [K / T, N] → [M, N]，得到同一组输出位置的部分和。完整结果需要跨 rank 相加。若每个 rank 都要完整结果，用 AllReduce。若下游只保留本 rank 的 token 段，用 ReduceScatter，输出 [M / T, N]。

Dense FFN 通常把这两种切法连用。第一层按输出特征切，本地激活后直接交给第二层，第二层按输入特征切。Attention 的常规 MHA 投影沿用相同思路，Q、K、V 投影按整颗 head 分配输出特征，输出投影再归约各 rank 的部分和。

Attention 与 FFN 的典型 TP 实现都可以由 Column Parallel 和 Row Parallel 两种 GEMM 组合得到。下面分别沿 Head 和 FFN 中间维度推导。

### Attention Head 切分与输出归约

<figure class="asb-figure asb-figure--source">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：张量并行的 Attention Head 切分与部分和归约"><img width="1100" height="460" src="/diagrams/tp-attention-heads.svg" alt="TP Attention 的 Head 切分和两个 GPU 的输出投影部分和归约矢量图" loading="lazy" /></button>
  <figcaption>Tensor Parallelism：每卡处理部分 Attention Head，输出投影后归约部分和</figcaption>
</figure>

每个 rank 的 Q、K、V 权重按输出特征列切为 [H, A / T]，都处理完整 S 个 token。MHA 的每张卡负责 Nₕ / T 个完整 head，head 宽度 Dₕ 不变。本地分数形状为 [B, Nₕ / T, S, S]。本地 head 拼成 [M, A / T]，乘本地输出投影权重 [A / T, H] 得到 [M, H] 部分和。纯 TP 用 AllReduce 求和，TP+SP 用 ReduceScatter 求和并把 token 分回各卡。对 GQA/MQA，KV head 数比 Q head 少，KV 如何分布或复制要看实现。

### FFN 中间特征切分与输出归约

第一层每卡做 [M, H] × [H, F / T] → [M, F / T]；各卡产出不同特征列，激活在本地做。第二层每卡做 [M, F / T] × [F / T, H] → [M, H]；各卡产出同一组输出位置的部分和。纯 TP 走 AllReduce，TP+SP 走 ReduceScatter。开启 SP 时，Q、K、V 或 FFN 第一层入口先 AllGather，恢复本线性层所需的 token。

以上示例采用均匀切分，默认投影维度可整除 TP 度。GQA/MQA 因 KV Head 数较少，不应直接假设每个 Rank 都能得到同样数量的 KV Head；部分实现会复制 KV Head 或采用其他布局。

**通信依赖的来源：** Column Parallel 后各卡拥有不同列，Row Parallel 后各卡拥有同一输出位置的部分和。通信来自下游算子所要求的**数据布局变化**，而不是为了满足某个通信框架的固定套路。

## 四　数据并行、流水线并行与序列并行

TP 减少了单卡处理的特征宽度，但它没有减少模型层数，也不自动提高不同样本之间的并行度。因此需要再引入几种分工方式。

### Data Parallel：复制计算路径，切分样本

DP 的基本形式是让每个副本拥有一套完整的逻辑模型，对不同样本执行前向与反向。对于前面的 8 GPU 示例，DP=2 并不代表每张 GPU 都有完整模型：每个**四卡 DP 副本**内部还做 TP=2 和 PP=2。训练时，在**相同 TP、PP 坐标**的 Rank 之间同步对应参数的梯度。若采用 ZeRO/FSDP 等技术，状态的所有权与通信路径还会改变。

### Pipeline Parallel：按层分阶段

PP 把连续的一组完整 block 放到每个阶段。在未叠加 TP、SP 或 CP 的例子里，阶段之间点对点传递形状 [b, S, H] 的 microbatch 激活。反向沿相反方向传激活梯度。多个 microbatch 可以同时在不同阶段运行。普通 DDP 的每个副本都跑全部 L 层，各自处理不同样本，反向后同步权重梯度。

在 8 GPU 示例中，Stage 0 计算第 0–11 层，Stage 1 计算第 12–23 层。若暂时忽略 TP/SP 对阶段边界布局的影响，一个 Microbatch 的边界激活形状为 \([2,4096,4096]\)。使用 BF16，每次单向传输的数据量为：

$$
2\times4096\times4096\times2\ \mathrm{bytes}=64\ \mathrm{MiB}.
$$

这只是**一个 Microbatch 的单个 Stage 边界激活**，不包括反向梯度、额外通信或流水调度开销。PP 的关键工程问题是 Microbatch 数量、Stage 负载均衡与 Pipeline Bubble。




### Sequence Parallel：让部分激活不必在整个 TP 组复制

在 Megatron 风格的 TP+SP 中，常将 Norm / 残差等位置的激活沿序列维保留为 \([B,S/T,H]\)。进入需要完整 Token 的 Column Parallel GEMM 前，先在 TP 组 AllGather；Row Parallel GEMM 后使用 ReduceScatter，直接把输出恢复成 Token 分片。

这解释了一个容易混淆的点：**SP 和 CP 都出现“Token 切分”，但它们的目的和 Attention 跨卡依赖并不相同。** SP 常用于降低 TP 场景中的激活冗余；CP 则要让超长上下文的 Attention 跨卡完成。

## 五　上下文并行与专家并行



<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：张量并行切 Head 和上下文并行切 Token 的局部形状"><img width="960" height="410" src="/diagrams/tp-cp-shapes.svg" alt="张量并行切 Head 和上下文并行切 Token 的局部形状" loading="lazy" /></button>
  <figcaption>标准 MHA 中，TP 沿 Head 维分组，CP 沿序列维分段；每个 Head 的宽度 D_h 不变。</figcaption>
</figure>

### Context Parallel：切 Token，但 Attention 还要看到全局上下文

CP 中每卡只负责 S / C 个 query token，head 数不因 CP 自身减少。FFN 逐 token 算本地分片。Attention 则跨卡取得可见的 K、V。计算全局分数的逻辑块形状为 [B, Nₕ, S / C, S]，可以逐块计算。

设四张 GPU 组成 CP 组，每张卡持有 \(S/4\) 个 Query 和相应的本地 K/V。对于本地 Query，它的注意力分母仍取决于**所有可见 Key**，而不是只有本地的 \(S/4\) 个 Key。因此 CP 的通信不仅是“把输入分开”，还必须实现跨 GPU 的 Attention。

这里有两类思路：让 K/V 分片到 Query 所在 GPU 上参与计算，或让各 GPU 计算本地 Attention 统计量再合并。下一章会用 Online Softmax 把这两种方法联系起来。

### Expert Parallel：把 Token 派发到拥有专家权重的 GPU

EP 中 E 代表专家总数，Tₑ 代表路由到专家 e 的 token 数。若专家权重完整放在一张卡上，本地专家做 [Tₑ, H] → [Tₑ, Fₑ] → [Tₑ, H]。流程是 Dispatch → 专家 GEMM → Combine。如果同一个专家还做专家 TP，Megatron 的 All-to-All dispatcher 可以在这两次 EP 通信之间，插入专家 GEMM 前的 AllGather 和 GEMM 后的 ReduceScatter。

与 CP 的 Token-to-Token 注意力依赖不同，EP 的跨 GPU 通信来自 **Router 选出的 Token-to-Expert 映射**。实际通信量取决于专家分布、Token 数、Top-k 与负载不均；不是简单由专家数 \(E\) 除以 EP 度就能判断耗时。没有 MoE 层的稠密模型也不需要 EP。

## 六　分布式 Attention 与跨设备归一化

要理解 Ring 与 Tree Attention，先回答一个问题：**一个 Query 的 Key 分散在多张 GPU 上，为什么不能把每张卡独立计算的 Softmax 输出直接相加？**

因为 Softmax 的分母是对**全部可见 Key**计算的。各 GPU 只对自己的 Key 归一化，会得到不同的局部分母；它们的局部输出直接求和或取平均，一般不等于全局 Attention。

可以用只有两个 Key 的例子看出差别：GPU 0 持有 \(s_0=0,v_0=0\)，GPU 1 持有 \(s_1=2,v_1=10\)。各 GPU 对自己的唯一 Key 做 Softmax，局部输出分别为 0 和 10；直接取平均得到 5。然而真正的全局输出是

$
O=\frac{e^0\cdot 0+e^2\cdot 10}{e^0+e^2}\approx 8.81.
$

因此，跨 GPU 需要合并的不只是“算完的向量”，还包括能够恢复全局 Softmax 归一化权重的信息。

### Online Softmax：将全局归一化变成可合并的状态

普通 Softmax。固定一个 batch、head 和 query，令 sᵢ 为该 query 对第 i 个可见 key 的注意力分数，pᵢ 为归一化后的权重。

$$
\begin{aligned} p_i &= \frac{\exp(s_i)}{\sum_j \exp(s_j)} \\[8pt] m &= \max_j s_j, \qquad p_i = \frac{\exp(s_i-m)}{\sum_j \exp(s_j-m)} \end{aligned}
$$

求和沿 key 维进行。分子和分母同乘 exp(−m)，结果相同，并能避免指数上溢。掩码位置的权重为 0。下面的 Online Softmax 把跨所有可见 key 的归一化改成逐块更新。

<figure class="asb-figure asb-figure--source">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：Online Softmax 的跨块归一化状态更新"><img width="1120" height="470" src="/diagrams/online-softmax.svg" alt="Online Softmax 逐块遍历 KV Tile，并更新运行最大值、指数和与加权输出的原创矢量数据流图" loading="lazy" /></button>
  <figcaption>Online Softmax：逐块扫描 KV 并维护归一化状态</figcaption>
</figure>

$$
\begin{aligned} m' &= \max(m,\; \max s_j) \\[5pt] \alpha &= e^{m-m'} \\[5pt] p_j &= e^{s_j-m'} \\[5pt] \ell' &= \alpha\,\ell + \sum p_j \\[5pt] a' &= \alpha\,a + \sum_j p_j\,V_j \\[5pt] O &= \frac{a'}{\ell'} \end{aligned}
$$

固定某个 query 行，设 key 块 j 的有效分数是 sⱼ、对应 value 是 Vⱼ。每行维护运行最大值 m、指数和 ℓ、未归一化的加权和 a，初始为 (−∞, 0, 0)。处理每个有效块时：

把旧累积量乘 α，是为了把旧块和新块的指数权重放在同一个最大值基准下。第一个有效块的旧权重取零。全部有效块处理完后 O = a / ℓ。全掩码块可跳过，全掩码行的输出由实现约定，这里取零。

### FlashAttention 的分块前向过程

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

把单卡分块过程延伸到多卡，每张 GPU 都可以对本地 K/V 计算 \((m,\ell,a)\)。局部最大值与指数和不能直接求和，但可以通过**最大值重标定**合并；这正是下面 Tree Attention 使用的数学基础。

不过分布式 Attention 不一定都选择传输统计量：Ring Attention 的主要做法是让 K/V 块流向保留 Query 的 GPU，然后在 Query 所在 GPU 上持续更新 Online Softmax 状态。下面先对比它们各自交换的数据。

### Ring、Tree 与 Ulysses 的通信对象

当上下文被切到多个 GPU 时，每个 Query 都需要汇聚所有可见 Key 的贡献。不同算法对通信对象的选择，决定了性能特征。

### Ring Attention：K/V 分片轮转

<figure class="asb-figure asb-figure--source">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：四 GPU Ring Attention 的 K/V 分片轮转"><img width="1080" height="430" src="/images/parallel/figure-5.svg" alt="四 GPU Ring Attention 中的 K/V 分片轮转" loading="lazy" /></button>
  <figcaption>Context Parallelism 中的 Ring Attention 数据交换</figcaption>
</figure>

设 CP 组有 C 个 rank。rank r 保留本地 Qᵣ，大小是 [B, S / C, Nₕ, Dₕ]，并持有自己的 Kᵣ、Vᵣ。每轮用当前 K、V 块计算局部 Attention，同时把 K、V 转给环上的邻居并接收下一块。mask 决定哪些 query-key 对有效。各 rank 用前面的 Online Softmax 合并各块的贡献，计算结束时只产生自己 query 分片的输出。

**这里的 Ring 不是 Ring AllGather。** Ring Attention 沿设备环传递 K/V 分片，让每个 Query 分片依次访问全局上下文；Ring AllGather 则是集合通信的一种实现算法，传输的可能是线性层输入激活。两者可以使用相似的网络拓扑，但通信对象、计算依赖和最终目的并不一样。

### Tree Attention：Softmax 统计量的树形归约

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
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：Ring Attention 通过环形传递 KV 分片；Tree Attention 计算局部统计量后执行树形归约的原创矢量对比图"><img width="1120" height="450" src="/diagrams/ring-vs-tree-attention.svg" alt="Ring Attention 通过环形传递 KV 分片；Tree Attention 计算局部统计量后执行树形归约的原创矢量对比图" loading="lazy" /></button>
  <figcaption>Ring 与 Tree Attention 的通信思路对比</figcaption>
</figure>

两者不能简单判断“哪个总是更快”：Ring Attention 的长序列训练 / prefill 路径与 Tree Attention 重点优化的长上下文 decode 路径并不完全相同，性能也受互连拓扑、序列长度、批大小和实现细节影响。

参考：[Tree Attention 原论文](https://arxiv.org/abs/2408.04093) · [代码实现](https://github.com/Zyphra/tree_attention)

### Ulysses：基于 All-to-All 的维度重排

还有一种不同的分布式 Attention 思路：**DeepSpeed Ulysses** 在设备间进行 All-to-All，使数据由序列维切分转换为 Head 维切分；每张卡随后对自己的 Head 计算 Attention，再进行反向的 All-to-All 还原布局。

可以这样归纳：**Ring 交换 K/V 块，Tree 归约 Attention 统计量，Ulysses 重排切分布局。** 它们解决的是跨设备获取全局上下文的问题，但具体通信方式不同。实际系统还可以组合多种方法。

参考：[DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)

::: tip 同名概念：Tree-structured Attention
上面的 Tree Attention 是一种**跨设备的精确 Attention 归约算法**。在推测解码、多分支推理或 Agent RL 中，“Tree Attention”还可能指根据**共享前缀 / 树状生成路径**复用计算的技术，例如 [DeFT](https://arxiv.org/abs/2404.00242) 和 [AREAL-DTA](https://arxiv.org/abs/2602.00482)。后者的“Tree”来自输入序列的分叉关系，而不是 GPU 归约拓扑。两类算法不能混为一谈。
:::

| 方法 | 跨卡主要交换什么 | 适合从哪个角度理解 |
| --- | --- | --- |
| Ring Attention | K/V 分片 | 保留本地 Query，边接收 K/V 边算并累积 |
| Tree Attention | Query 分发及每个 Query 的局部 Attention 统计量 | KV 留在持有它的 GPU，上层合并 Softmax 状态 |
| Ulysses | 经 All-to-All 重排的序列 / Head 分片 | 先改变布局，再执行局部 Head Attention |

**Ring Attention ≠ NCCL Ring AllReduce，Tree Attention ≠ NCCL Tree AllReduce。** 前者是完成 Attention 的算法与数据依赖，后者是通信库实现集合操作时可能采用的拓扑 / 算法。通信库内部使用什么 Ring、Tree 或其他算法，不能直接从 Attention 名称反推。

对于长上下文、单步 Query 很少的 Decode，发送与 Query 数相关的统计量可能比沿环搬运大块历史 K/V 更有利；而 Prefill/训练的大块 Query 计算和硬件互连会改变性能权衡。不能脱离 Query 长度和设备拓扑直接断言哪一种总是更快。

## 七　多维并行进程组与通信成本

把前文分工写成坐标：\((d,p,t)\) 分别对应 DP、PP、TP。8 GPU 示例的 \(D=2,P=2,T=2\)；固定其他坐标、只改变 \(t\) 得到 TP 组；只改变 \(p\) 得到 PP 流水；只改变 \(d\) 得到对应参数分片的 DP 同步组。

若再引入独立 CP 维度，可扩展为 \((d,p,t,c)\)。但 **SP 通常复用 TP 组**，不能盲目把 SP 度再乘进 GPU 总数；EP 的具体映射还要考虑 Expert DP、Expert TP 等框架选择。

### TP AllReduce 的通信量示例

在示例中，单个 Microbatch 的输出激活形状是 \([2,4096,4096]\)，BF16 大小约为 **64 MiB**。使用 TP=2、且不启用 SP 的 Row Parallel 输出投影时，每个 Rank 各持有这个形状的部分和，因此需要归约后才能让两个 Rank 都获得完整结果。

对于理想化 Ring AllReduce、组大小 \(R=2\)、每 Rank 完整张量大小 \(U=64\ \mathrm{MiB}\)，单 Rank 发送量为：

$$
2\frac{R-1}{R}U = 64\ \mathrm{MiB}.
$$

接收量相同；真实时间还受到通信启动、链路拓扑、内核融合和重叠程度影响。这个公式不能直接代替端到端性能测量。

### Ring 集合通信的每 Rank 数据量

设通信组有 R 个 rank，完整逻辑张量大小为 U 字节，均匀分成 R 块。以下计量每个 rank 发出的 payload，接收量相同。若统计收发总量，需要再乘以 2。

**Ring AllGather**：单 rank 输入 U/R，输出 U。每轮发送 U/R，共 R−1 轮，发送量为 (R−1)·U/R。

**Ring ReduceScatter**：单 rank 输入 U，归约后保留 U/R。每轮发送 U/R，共 R−1 轮，发送量为 (R−1)·U/R。

**Ring AllReduce**：单 rank 输入和输出都是 U。先做 ReduceScatter，再做 AllGather，发送量为 2·(R−1)·U/R。

对应 TP+SP 的 AllGather，U 是收齐后的输入激活大小。对于 GEMM 后的 ReduceScatter 或 AllReduce，U 是每 rank 的完整部分和张量大小。这两处的 U 要分别按张量形状计算。

### 通信与计算重叠的布局约束

典型入口依赖是 **AllGather → GEMM**，先收齐算子要求的输入；出口依赖是 **GEMM → ReduceScatter / AllReduce**，先得到部分和再归约。拆成更小的 Chunk 后，某些输入收集、计算与输出归约可能流水化；但是否能重叠，要看 Buffer 生命周期、流同步、通信协议与硬件资源竞争。

## 八　八 GPU 示例的前向与反向流程

沿一个 Microbatch 的数据流，可以将 8 张 GPU 的前向与反向执行对应到以下阶段：

1. **DP 分样本。** 两个 DP 副本获得不同的 Microbatch；每个副本由四张 GPU 合作执行同一个逻辑模型。
2. **PP 分层。** Stage 0 计算前 12 层，随后把边界激活传给 Stage 1；Stage 1 计算后 12 层。
3. **TP 分算子。** 每个 Stage 的两张 GPU 各负责 16 个 Attention Head 和 5504 个 FFN 中间特征；输出投影产生相同位置的部分和，通过 TP 归约形成下一算子所需布局。
4. **反向沿相反依赖传播。** PP 在阶段之间传回激活梯度；TP 的各个分片分别计算局部梯度和必要的集合通信。
5. **DP 同步梯度。** 在对应 TP / PP 坐标之间，对相同参数分片的梯度执行同步或使用相应的分布式优化器方案更新状态。

本示例是**稠密模型**且 CP=EP=1。因此不会发生跨 GPU 的专家 Dispatch，也不需要 Ring/Tree Attention 的 CP 数据流。如果后续把上下文再沿 CP=2 切分，必须重新计算 world size、每 Rank Q/K/V 形状以及跨设备 Attention 的通信代价；不能在原有 8 GPU 拓扑上凭空再添一个独立维度。

### 关键机制辨析

**Q：Column Parallel 后是不是一定要 AllGather？** 不一定。若下游直接消费分片（例如紧接的 Row Parallel），可以保持分片；是否收集取决于布局需求。

**Q：为什么 Row Parallel 经常接 AllReduce / ReduceScatter？** 因为各 Rank 计算的是同一输出位置的不同部分和，需要跨 Rank 求和。下一阶段是否保留完整 Token 决定归约后的输出布局。

**Q：为什么 Ring/Tree Attention 不能直接等同于 NCCL 的 Ring/Tree？** 因为前者规定 Attention 如何组织 Query、K/V 与 Softmax 统计量；后者是集合通信的一种底层实现路线，两者处于不同抽象层。

## 结论：并行的核心是数据所有权与依赖

从单卡到多卡，最重要的不是列出多少种并行名称，而是把每个边界写成三个问题：

- **谁拥有参数与激活？** 按样本、层、特征、Token，还是专家切分？
- **算子需要什么输入布局？** 是本地独立可算、需要别的分片，还是需要归约同一输出位置的部分和？
- **交换什么，交换多少，何时交换？** 是激活、梯度、K/V、路由 Token，还是 Softmax 统计量？通信能否与计算重叠？

这套判断方法可以继续用于理解分布式训练、长上下文推理、MoE 路由和强化学习后训练框架中的训推数据流。

## 扩展：Attention Kernel 与 KV Cache 管理

以下内容与并行设计相关，但属于**不同抽象层**：FlashAttention 主要关注单 GPU Attention 的 IO / Kernel 执行，PagedAttention 关注推理 KV Cache 的逻辑块映射，LMCache / Mooncake / HiCache 关注缓存的层次与跨系统迁移。它们可以和 TP / CP 等并行策略同时存在，不应混作并行度的独立轴。

### FlashAttention v1–v4 演进

<div class="asb-table-kicker">FlashAttention 各版本的核心优化</div>

| 版本 | 主要改变 |
| --- | --- |
| FA1 | IO 分块与在线归一化，降低 HBM 流量。 |
| FA2 | 改善线程块和 warp 的工作分配，减少非矩阵乘操作。 |
| FA3 | 面向 Hopper，利用 TMA 和 Tensor Core 异步流水，支持 FP8 路径。 |
| FA4 | 面向 Blackwell，联合设计算法和流水，使用指数近似等手段缓解计算单元增长不均衡。 |

**各版本的变化可归纳为：**

**v1 · 减少显存读写。** 分块加载 Q、K、V，用 Online Softmax 在片上累积结果，避免把整张注意力矩阵写回 HBM。

**v2 · 改善并行和线程分工。** 增加沿序列维的并行度，调整线程块内的 warp 分工，减少非矩阵乘操作和共享内存通信。

**v3 · 面向 Hopper 的异步流水。** 将数据搬运、矩阵乘与 Softmax 交错执行，结合 warp specialization，并提供 FP8 路径。

**v4 · 面向 Blackwell 的流水与归约。** 使用更大的 tile 和异步 MMA，减少指数计算与重缩放开销，并用 Tensor Memory 和双 CTA MMA 降低反向共享内存流量和原子累加开销。

### MHA、GQA、MQA、MLA 如何影响并行布局与 KV 体积

这些名称描述模型结构。用 n、k 表示 query 和 KV head 数。在均匀分组下，每组 n / k 个 query head 共享一个 KV head。MHA 有 k = n，GQA 有 1 < k < n，MQA 有 k = 1。等宽 K、V、每元素 w 字节时，普通 KV payload 为 2 · k · Dₕ · w 字节每 token 每层，忽略框架副本和分配开销。

MLA（Multi-head Latent Attention）使用低秩潜变量压缩 KV。以 DeepSeek-V2 为例，缓存可保留宽度 Lᵥ 的 KV latent 和宽度 R 的独立 RoPE key 分量。两者精度相同时，对应的 payload 是 (Lᵥ + R) · w。实现可以吸收部分上投影，避免每步都显式展开全部历史 K、V。

对于 TP 切分，必须核对 Q Head 数、KV Head 数与 TP 度的整除关系；GQA/MQA 的 KV Head 很少时，具体实现可能采取复制、共享或其他策略，而不是强行平均分配。

### PagedAttention 与多级 KV Cache

推理服务还需要解决另一个独立问题：历史 K/V 怎样存放、查找、复用并迁移，而不是只考虑一次 Attention 如何在 GPU 之间计算。

<figure class="asb-figure asb-figure--source">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：PagedAttention 逻辑 KV 块到物理块的映射"><img width="1080" height="430" src="/images/parallel/figure-4.svg" alt="PagedAttention 中逻辑 KV 块与物理存储块的映射" loading="lazy" /></button>
  <figcaption>PagedAttention 的 KV Cache 块映射</figcaption>
</figure>

PagedAttention 关注的是 KV Cache 的**逻辑位置与物理存储块之间的映射**，不是新的一种跨 GPU Attention 并行轴。若每个 KV block 容纳 J 个 token，逻辑位置 t 的块号是 floor(t/J)，块内偏移是 t mod J。块表把逻辑块号映射到物理块，物理块可不连续。请求增长时按需申请，支持前缀共享，减少碎片和预留浪费。每个有效 token 的 KV payload 由模型结构和精度决定。

### LMCache、Mooncake 与 HiCache：缓存管理的不同层次

<div class="asb-table-kicker">KV Cache 相关系统对比</div>

| 项目 | 是什么 | 层级 |
| --- | --- | --- |
| LMCache | 独立的 KV Cache 管理层，挂在 vLLM/SGLang 外面 | 引擎外置的"存储中间件" |
| Mooncake | 以 KVCache 为中心的分离式推理架构（PD 分离 + 分布式缓存池） | 集群级架构 |
| HiCache | SGLang 内置的三级缓存模块 | 引擎内部机制 |

**LMCache：引擎外置的 KV Cache 管理层。** 它关注 KV Cache 跨请求的复用、卸载和跨存储层迁移。常见层级包括 GPU HBM（活跃工作集）、CPU DRAM（热缓存）、本地 SSD / NVMe，以及远端存储。前缀复用、缓存卸载和 Prefill/Decode 间的传输是相关但不同的使用方式。

**Mooncake：面向 Prefill/Decode 分离的分布式 KV Cache 架构。** 它利用集群的 CPU 内存、存储设备与网络资源，跨 Prefill 和 Decode 阶段传输、管理 KV Cache，从系统级减少重复计算和资源闲置。

**HiCache：推理引擎内的分层缓存机制。** 可以按 GPU HBM、CPU 主机内存和外部 / 分布式存储的层级理解。它与 PagedAttention 不是互斥方案：前者关注缓存层次与迁移，后者关注 KV 的逻辑块和物理块映射。

### NCCL：通信算法还依赖具体拓扑与协议

以下列出 NCCL 的一些关键环节，帮助把前面 AllGather、ReduceScatter、AllReduce 的逻辑通信映射到实际执行。**注意：这不是一次 AllReduce 必定按顺序执行的七个固定阶段**。初始化、拓扑发现、协议选择、通道组织、GPU Kernel 与网络传输在不同硬件和协议下的执行路径可能不同。

**NCCL 通信路径涉及的典型组件**

| 阶段 | 核心操作 | 备注 |
| --- | --- | --- |
| 初始化（建链） | 初始化通信域及连接资源 | 单节点通信路径取决于硬件互连，不一定使用 TCP |
| 拓扑发现 | 发现 GPU 和网络互连拓扑 | NVLink、PCIe、InfiniBand 等 |
| Ring/Tree 等通信组织 | 根据算法和拓扑组织通信关系 | 具体选择因规模与硬件而异 |
| channel分配 | 数据切chunk，多路并行传 | 每个chunk走一个channel |
| Proxy 线程 | 某些通信协议或网络路径需要 CPU 端 Proxy 协作 | 并非每个 Channel 都必然有独立线程 |
| QP 建立 | RDMA 网络路径可能需要建立 Queue Pair | 单机 NVLink/PCIe 路径不一定涉及 QP |
| 数据传输 | 通过选定协议与互连传输数据 | RDMA 只是可能的传输方式之一 |

## 参考资料与进一步阅读

[Megatron Core 并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) · [Megatron Core 上下文并行](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) · [Megatron Core MoE Dispatcher](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html)

[GQA](https://arxiv.org/abs/2305.13245) · [DeepSeek-V2 与 MLA](https://arxiv.org/abs/2405.04434) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [FA2](https://arxiv.org/abs/2307.08691) · [FA3](https://arxiv.org/abs/2407.08608) · [FA4](https://arxiv.org/abs/2603.05451)

[PagedAttention](https://arxiv.org/abs/2309.06180) · [Ring Attention](https://arxiv.org/abs/2310.01889) · [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
