---
title: Transformer 并行方式与 Attention 实现
description: 从单卡计算到 TP、PP、DP、CP、EP，并理解 FlashAttention、PagedAttention、Ring Attention 与 NCCL 通信原理。
outline: deep
---

<div class="asb-editorial-heading">DEEP DIVE / PARALLEL COMPUTING</div>

# Transformer 并行方式与 Attention 实现

*从单卡计算、分布式切分轴到 Attention 算法与通信机制，沿着张量形状重新理解系统设计。*

<figure class="asb-figure">
<svg viewBox="0 0 880 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="从单卡 Transformer 到多卡并行策略的原创概念图">
  <defs><linearGradient id="parallel-flow" x1="0" x2="1"><stop stop-color="#49c8a3"/><stop offset="1" stop-color="#6c9fe6"/></linearGradient></defs>
  <rect x="12" y="38" width="220" height="151" rx="15" fill="none" stroke="#67c9af" stroke-opacity=".65"/>
  <text x="32" y="67" font-family="sans-serif" fill="#6ce6bf" font-size="14" font-weight="700">01 / Single device</text>
  <rect x="32" y="89" width="180" height="34" rx="7" fill="#268871" fill-opacity=".2" stroke="#3fbd9d"/>
  <rect x="32" y="137" width="180" height="34" rx="7" fill="#3288b3" fill-opacity=".16" stroke="#559fc8"/>
  <text x="122" y="112" fill="currentColor" font-size="13" text-anchor="middle">Attention</text>
  <text x="122" y="160" fill="currentColor" font-size="13" text-anchor="middle">FFN</text>
  <path d="M249 113 H309" stroke="url(#parallel-flow)" stroke-width="2" stroke-dasharray="6 5"/>
  <path d="M302 107l8 6-8 6" stroke="#6ce6bf" fill="none" stroke-width="2"/>
  <rect x="325" y="38" width="222" height="151" rx="15" fill="none" stroke="#67c9af" stroke-opacity=".65"/>
  <text x="345" y="67" font-family="sans-serif" fill="#6ce6bf" font-size="14" font-weight="700">02 / Partition axes</text>
  <g fill="#123b3b" stroke="#43bd9a"><rect x="344" y="90" width="60" height="31" rx="6"/><rect x="417" y="90" width="60" height="31" rx="6"/><rect x="490" y="90" width="40" height="31" rx="6"/><rect x="344" y="135" width="60" height="31" rx="6"/><rect x="417" y="135" width="60" height="31" rx="6"/><rect x="490" y="135" width="40" height="31" rx="6"/></g>
  <g fill="#c6fff0" text-anchor="middle" font-size="12"><text x="374" y="110">TP</text><text x="447" y="110">PP</text><text x="510" y="110">DP</text><text x="374" y="155">CP</text><text x="447" y="155">EP</text><text x="510" y="155">SP</text></g>
  <path d="M564 113 H624" stroke="url(#parallel-flow)" stroke-width="2" stroke-dasharray="6 5"/><path d="M617 107l8 6-8 6" stroke="#6ce6bf" fill="none" stroke-width="2"/>
  <rect x="640" y="38" width="227" height="151" rx="15" fill="none" stroke="#67c9af" stroke-opacity=".65"/>
  <text x="660" y="67" font-family="sans-serif" fill="#6ce6bf" font-size="14" font-weight="700">03 / Distributed execution</text>
  <g fill="#244767" stroke="#6badd6"><rect x="663" y="88" width="76" height="37" rx="6"/><rect x="762" y="88" width="76" height="37" rx="6"/><rect x="663" y="140" width="76" height="37" rx="6"/><rect x="762" y="140" width="76" height="37" rx="6"/></g>
  <g fill="#e6f8ff" font-size="12" text-anchor="middle"><text x="701" y="111">GPU 0</text><text x="800" y="111">GPU 1</text><text x="701" y="163">GPU 2</text><text x="800" y="163">GPU 3</text></g>
  <path d="M740 106H761 M740 159H761 M700 126V139 M800 126V139" stroke="#76e4c1" stroke-width="1.7"/>
</svg>
<figcaption>原创概念图：从单卡 Transformer block 到并行切分轴，再到分布式执行。</figcaption>
</figure>

从单卡的 Attention 和 FFN 公式出发，沿层、样本、token、head、特征和专家几个轴推导多卡形状，再看 FlashAttention、PagedAttention 和 Ring Attention 各自改变了哪一步。

## 一　单卡上的 Transformer block

<figure class="asb-figure"><img src="/images/parallel/figure-1.svg" alt="飞书原始技术示意图 1" loading="lazy" /><figcaption>原文图 1 · Transformer 并行与 Attention 技术示意</figcaption></figure>

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

::: warning 原文嵌入电子表格 1
此处在飞书原文中为嵌入式电子表格。当前转换流程尚未取得表格单元格数据，暂时无法保证表格无损还原。可先[查看飞书原文](https://mcn1ijk3pml3.feishu.cn/wiki/Mujfwy0wsi0bjIkbReScJvhKn3M)。
:::

### 并行方式确定分工，具体切法可以变化

并行方式给出分工的对象，具体布局还要看算子和上下游。PP 决定每个阶段放哪些 block；DP 让各模型副本处理不同样本，普通 DDP 复制训练状态，ZeRO 或 FSDP 还可以对训练状态分片。EP 决定专家放在哪些 rank，同一个专家还可以叠加专家 TP，拆分内部 GEMM。

SP（按 Megatron 的用法）通常复用 TP 组，把部分激活按 token 切分，在需要完整输入的线性层前收齐。CP 把上下文 token 分给各 rank。Attention 可以保留本地 Q、交换 K、V，也可以通过 All-to-All 在 token 分片和 head 分片之间重排。

同一个 GEMM 有两种常见 TP 切法。 设 X 的形状为 [M, K]，权重 W 为 [K, N]，TP 组大小为 T。

按输出特征 N 切。 各 rank 做 [M, K] × [K, N / T] → [M, N / T]，得到不同的输出列。下游可直接消费分片时就保持原布局。只有下游需要完整输出时，才沿输出特征做 AllGather。若上游用了 SP，还可能需要在 GEMM 前沿 token 做 AllGather。这两次 AllGather 的对象和切分轴不同。

按输入特征 K 切。 各 rank 做 [M, K / T] × [K / T, N] → [M, N]，得到同一组输出位置的部分和。完整结果需要跨 rank 相加。若每个 rank 都要完整结果，用 AllReduce。若下游只保留本 rank 的 token 段，用 ReduceScatter，输出 [M / T, N]。

Dense FFN 通常把这两种切法连用。第一层按输出特征切，本地激活后直接交给第二层，第二层按输入特征切。Attention 的常规 MHA 投影沿用相同思路，Q、K、V 投影按整颗 head 分配输出特征，输出投影再归约各 rank 的部分和。

### PP 和 DP 怎样跨 block

PP 把连续的一组完整 block 放到每个阶段。在未叠加 TP、SP 或 CP 的例子里，阶段之间点对点传递形状 [b, S, H] 的 microbatch 激活。反向沿相反方向传激活梯度。多个 microbatch 可以同时在不同阶段运行。普通 DDP 的每个副本都跑全部 L 层，各自处理不同样本，反向后同步权重梯度。

### TP 在 Attention 里切 head 数

<figure class="asb-figure"><img src="/images/parallel/figure-2.svg" alt="飞书原始技术示意图 2" loading="lazy" /><figcaption>原文图 2 · Transformer 并行与 Attention 技术示意</figcaption></figure>

每个 rank 的 Q、K、V 权重按输出特征列切为 [H, A / T]，都处理完整 S 个 token。MHA 的每张卡负责 Nₕ / T 个完整 head，head 宽度 Dₕ 不变。本地分数形状为 [B, Nₕ / T, S, S]。本地 head 拼成 [M, A / T]，乘本地输出投影权重 [A / T, H] 得到 [M, H] 部分和。纯 TP 用 AllReduce 求和，TP+SP 用 ReduceScatter 求和并把 token 分回各卡。对 GQA/MQA，KV head 数比 Q head 少，KV 如何分布或复制要看实现。

### TP 在 FFN 里切中间特征

第一层每卡做 [M, H] × [H, F / T] → [M, F / T]；各卡产出不同特征列，激活在本地做。第二层每卡做 [M, F / T] × [F / T, H] → [M, H]；各卡产出同一组输出位置的部分和。纯 TP 走 AllReduce，TP+SP 走 ReduceScatter。开启 SP 时，Q、K、V 或 FFN 第一层入口先 AllGather，恢复本线性层所需的 token。

### CP 切 token EP 切专家

CP 中每卡只负责 S / C 个 query token，head 数不因 CP 自身减少。FFN 逐 token 算本地分片。Attention 则跨卡取得可见的 K、V。计算全局分数的逻辑块形状为 [B, Nₕ, S / C, S]，可以逐块计算。

EP 中 E 代表专家总数，Tₑ 代表路由到专家 e 的 token 数。若专家权重完整放在一张卡上，本地专家做 [Tₑ, H] → [Tₑ, Fₑ] → [Tₑ, H]。流程是 Dispatch → 专家 GEMM → Combine。如果同一个专家还做专家 TP，Megatron 的 All-to-All dispatcher 可以在这两次 EP 通信之间，插入专家 GEMM 前的 AllGather 和 GEMM 后的 ReduceScatter。

## 三　MHA GQA MLA 与 KV 缓存结构

这些名称描述模型结构。用 n、k 表示 query 和 KV head 数。在均匀分组下，每组 n / k 个 query head 共享一个 KV head。MHA 有 k = n，GQA 有 1 < k < n，MQA 有 k = 1。等宽 K、V、每元素 w 字节时，普通 KV payload 为 2 · k · Dₕ · w 字节每 token 每层，忽略框架副本和分配开销。

MLA（Multi-head Latent Attention）使用低秩潜变量压缩 KV。以 DeepSeek-V2 为例，缓存可保留宽度 Lᵥ 的 KV latent 和宽度 R 的独立 RoPE key 分量。两者精度相同时，对应的 payload 是 (Lᵥ + R) · w。实现可以吸收部分上投影，避免每步都显式展开全部历史 K、V。

## 四　Online Softmax 与 FlashAttention

### Online Softmax 怎样跨块合并

普通 Softmax。固定一个 batch、head 和 query，令 sᵢ 为该 query 对第 i 个可见 key 的注意力分数，pᵢ 为归一化后的权重。

$$
\begin{aligned} p_i &= \frac{\exp(s_i)}{\sum_j \exp(s_j)} \\[8pt] m &= \max_j s_j, \qquad p_i = \frac{\exp(s_i-m)}{\sum_j \exp(s_j-m)} \end{aligned}
$$

求和沿 key 维进行。分子和分母同乘 exp(−m)，结果相同，并能避免指数上溢。掩码位置的权重为 0。下面的 Online Softmax 把跨所有可见 key 的归一化改成逐块更新。

<figure class="asb-figure"><img src="/images/parallel/figure-3.svg" alt="飞书原始技术示意图 3" loading="lazy" /><figcaption>原文图 3 · Transformer 并行与 Attention 技术示意</figcaption></figure>

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

::: warning 原文嵌入电子表格 2
此处在飞书原文中为嵌入式电子表格。当前转换流程尚未取得表格单元格数据，暂时无法保证表格无损还原。可先[查看飞书原文](https://mcn1ijk3pml3.feishu.cn/wiki/Mujfwy0wsi0bjIkbReScJvhKn3M)。
:::

### FlashAttention v1 到 v4 分别优化什么

**v1 · 减少显存读写。** 分块加载 Q、K、V，用 Online Softmax 在片上累积结果，避免把整张注意力矩阵写回 HBM。

**v2 · 改善并行和线程分工。** 增加沿序列维的并行度，调整线程块内的 warp 分工，减少非矩阵乘操作和共享内存通信。

**v3 · 面向 Hopper 的异步流水。** 将数据搬运、矩阵乘与 Softmax 交错执行，结合 warp specialization，并提供 FP8 路径。

**v4 · 面向 Blackwell 的流水与归约。** 使用更大的 tile 和异步 MMA，减少指数计算与重缩放开销，并用 Tensor Memory 和双 CTA MMA 降低反向共享内存流量和原子累加开销。

## 五　PagedAttention 和 Ring Attention

### PagedAttention 管理推理 KV Cache

<figure class="asb-figure"><img src="/images/parallel/figure-4.svg" alt="飞书原始技术示意图 4" loading="lazy" /><figcaption>原文图 4 · Transformer 并行与 Attention 技术示意</figcaption></figure>

将你提到的“配置的 Attention”按 PagedAttention 理解。若每个 KV block 容纳 J 个 token，逻辑位置 t 的块号是 floor(t/J)，块内偏移是 t mod J。块表把逻辑块号映射到物理块，物理块可不连续。请求增长时按需申请，支持前缀共享，减少碎片和预留浪费。每个有效 token 的 KV payload 由模型结构和精度决定。

### Ring Attention 是 CP 的一种实现

<figure class="asb-figure"><img src="/images/parallel/figure-5.svg" alt="飞书原始技术示意图 5" loading="lazy" /><figcaption>原文图 5 · Transformer 并行与 Attention 技术示意</figcaption></figure>

设 CP 组有 C 个 rank。rank r 保留本地 Qᵣ，大小是 [B, S / C, Nₕ, Dₕ]，并持有自己的 Kᵣ、Vᵣ。每轮用当前 K、V 块计算局部 Attention，同时把 K、V 转给环上的邻居并接收下一块。mask 决定哪些 query-key 对有效。各 rank 用前面的 Online Softmax 合并各块的贡献，计算结束时只产生自己 query 分片的输出。

举例对照两种通信依赖：Ring Attention 沿环传 K、V 块，服务于 Attention 的跨 token 依赖。TP/SP 的 Ring AllGather 若选用 ring 算法，传的是线性层输入的激活分片。另一种长上下文实现 Ulysses 用 All-to-All 把序列分片重排成 head 分片，完成 Attention 后再换回去。

## 六　3D 4D 5D 并行与通算融合

常见 3D 组合是 DP×TP×PP。加入 CP 的独立轴配置，world size 可以写成 D·T·P·C。固定其他坐标，只沿 tp 坐标变化得到 TP 组，只沿 cp 坐标变化得到 CP 组。MoE 项目可能把 EP 也计入“几 D”，具体以该框架的进程组为准。SP 通常复用 TP 组，EP 与普通 DP、专家 DP 和专家 TP 的排组需要单独核对。

回到通算融合，入口是 **AllGather → GEMM**，通信先产出可读的输入块。出口是 **GEMM → ReduceScatter 或 AllReduce**，计算先产出部分和。数据布局决定箭头方向，块级实现决定通信和另一块的计算能否同时进行。MoE 可以重叠 Dispatch/Combine 与专家 GEMM，Ring Attention 可以重叠 K、V 传输与 Attention tile 计算。

### Ring 集合通信的每 rank 通信量

设通信组有 R 个 rank，完整逻辑张量大小为 U 字节，均匀分成 R 块。以下计量每个 rank 发出的 payload，接收量相同。若统计收发总量，需要再乘以 2。

**Ring AllGather**：单 rank 输入 U/R，输出 U。每轮发送 U/R，共 R−1 轮，发送量为 (R−1)·U/R。

**Ring ReduceScatter**：单 rank 输入 U，归约后保留 U/R。每轮发送 U/R，共 R−1 轮，发送量为 (R−1)·U/R。

**Ring AllReduce**：单 rank 输入和输出都是 U。先做 ReduceScatter，再做 AllGather，发送量为 2·(R−1)·U/R。

对应 TP+SP 的 AllGather，U 是收齐后的输入激活大小。对于 GEMM 后的 ReduceScatter 或 AllReduce，U 是每 rank 的完整部分和张量大小。这两处的 U 要分别按张量形状计算。

## NCCL中执行一次All Reduce内部进行的7个阶段

::: warning 原文复杂表格 1
本处为飞书结构化表格；单元格文字仍在补充提取中。请参照[飞书原文](https://mcn1ijk3pml3.feishu.cn/wiki/Mujfwy0wsi0bjIkbReScJvhKn3M)核对。
:::

初始化（建链）

建立通信域，所有GPU互相知道彼此的存在

TCP 单节点？

拓扑发现

探测连接方式 NVLink？PCIe？IB？

ring/tree构建

排通信顺序 谁连谁

channel分配

数据切chunk，多路并行传

每个chunk走一个channel

proxy线程

每个channel启动一个收发线程

QP建立

建RDMA连接，端点就绪

TCP

RDMA传输

数据收发，kernel bypass

::: warning 原文嵌入电子表格 3
此处在飞书原文中为嵌入式电子表格。当前转换流程尚未取得表格单元格数据，暂时无法保证表格无损还原。可先[查看飞书原文](https://mcn1ijk3pml3.feishu.cn/wiki/Mujfwy0wsi0bjIkbReScJvhKn3M)。
:::

- LMcache
官方定位是面向LLM推理的KV cache的管理层，把KV cache从临时状态变成可复用、可持久存储、可跨引擎共享的东西。里面也是会做分层

GPU显存：活跃工作集

CPU DRAM：用pin memory做热缓存

本地存储：磁盘、NVMe、GDS

远程存储：Redis、MoonCake、infinistore

支持两种模式：cache offloading（跨请求前缀复用）、PD分离（跨引擎、GPU的缓存传输）

- MoonCake
把 Prefill 和 Decode 拆成两个集群，中间用分布式 KVCache 池连接，池子靠集群里闲置的 CPU、DRAM、SSD、NIC 拼出来——这叫"以存换算"。

- HiCache
SGLang的三级缓存

仿CPU的三级缓存，L1 GPU HBM L2 CPU host RAM  L3分布式存储

参考资料

[Megatron Core 并行指南](https%3A%2F%2Fdocs.nvidia.com%2Fmegatron-core%2Fdeveloper-guide%2Flatest%2Fuser-guide%2Fparallelism-guide.html) · [Megatron Core 上下文并行](https%3A%2F%2Fdocs.nvidia.com%2Fmegatron-core%2Fdeveloper-guide%2Flatest%2Fuser-guide%2Ffeatures%2Fcontext_parallel.html) · [Megatron Core MoE Dispatcher](https%3A%2F%2Fdocs.nvidia.com%2Fmegatron-core%2Fdeveloper-guide%2Flatest%2Fapidocs%2Fcore%2Fcore.transformer.moe.token_dispatcher.html)

[GQA](https%3A%2F%2Farxiv.org%2Fabs%2F2305.13245) · [DeepSeek-V2 与 MLA](https%3A%2F%2Farxiv.org%2Fabs%2F2405.04434) · [FlashAttention](https%3A%2F%2Farxiv.org%2Fabs%2F2205.14135) · [FA2](https%3A%2F%2Farxiv.org%2Fabs%2F2307.08691) · [FA3](https%3A%2F%2Farxiv.org%2Fabs%2F2407.08608) · [FA4](https%3A%2F%2Farxiv.org%2Fabs%2F2603.05451)

[PagedAttention](https%3A%2F%2Farxiv.org%2Fabs%2F2309.06180) · [Ring Attention](https%3A%2F%2Farxiv.org%2Fabs%2F2310.01889) · [DeepSpeed Ulysses](https%3A%2F%2Farxiv.org%2Fabs%2F2309.14509)

