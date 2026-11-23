---
title: 从TransferQueue重构理解python、cpp、Rust
description: 以 TransferQueue 的 Rust Controller 和 PayloadTransfer 重构为线索，理解 Python、C++、Rust 在 RL 基础设施中的分工、瓶颈与工程权衡。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="从transferqueue重构理解python-cpp-rust" tabindex="-1">从TransferQueue重构理解python、cpp、Rust</h1>
  <p class="asb-story-subtitle">从 RL 数据管道的控制面、数据面到编程语言选型</p>
  <p class="asb-story-deck">为什么一个以 Python 为主的 RL 数据基础设施，会考虑用 Rust 重构高频控制路径，并为 SimpleStorage 引入可插拔的传输层？关键不只是语言运行速度，而是瓶颈位置、状态表示与模块边界。</p>
</div>

TransferQueue 的基础架构、样本就绪状态与存储后端机制见 [《TransferQueue》](/posts/transferqueue)。

在 RL 后训练系统中，Rollout、Reward、Training 等组件可能以不同速度运行。训练端需要从生产者持续获得可用样本，生产者又需要知道数据何时可消费。随着并发请求、样本数量和数据规模增加，**控制请求的处理效率**和**实际数据的搬运效率**可能分别成为瓶颈。

[TransferQueue](https://github.com/Ascend/TransferQueue) 最近提出两条相关但独立的重构方向：以 Rust 实现 Controller 的高频元数据处理，以及给 SimpleStorage 增加可插拔的 PayloadTransfer 接口。这为理解 Python、C++ 与 Rust 在系统基础设施中的分工提供了一个具体案例。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 TransferQueue 架构图"><img src="/diagrams/transferqueue-rearchitecture.svg" alt="TransferQueue 控制面与数据面两条重构路线的架构图" /></button>
  <figcaption>同一条数据管道中，控制元数据的流转与 Tensor Payload 的传输不是同一件事。</figcaption>
</figure>

## 一　TransferQueue 的控制面与数据面

系统里有两类容易混淆的流量：

| 路径 | 传递什么 | 关注什么 |
| --- | --- | --- |
| 控制面（Controller） | 样本就绪状态、索引、批次元数据、消费状态、KV 元数据请求等 | 高频 RPC、状态更新、并发调度、编码和尾延迟 |
| 数据面（Storage） | Tensor、字符串等实际字段值 | 传输吞吐、内存布局、缓冲区生命周期、跨节点复制 |

按照 [Rust Controller RFC #220](https://github.com/Ascend/TransferQueue/issues/220) 的描述，Ray 的主要作用是启动、发现和配置 Controller Actor；Controller 客户端与存储组件取得 Endpoint 后，运行时的元数据 RPC 可以直接经 ZMQ 通信。存储后端则在另一条路径上移动实际 Payload。

因此，**用 Rust 重写 Controller 不等于使用 RDMA 传 Tensor**，而给 SimpleStorage 增加可插拔传输接口，也不等于让 Controller 自动获得更高并发。

## 二　Python Controller 的性能瓶颈

性能分析的起点是定位**高频请求实际经过的执行路径**，而非仅根据语言判断效率。

RFC #220 指出了几个具体瓶颈：

1. **单线程请求循环。** 高频的 GET_META、NOTIFY_DATA_UPDATE 和 KV 相关 RPC 汇聚到 Python 的请求处理循环；在其描述的 CPU 密集型状态处理路径中，GIL 和单线程串行执行会限制扩展。
2. **非阻塞 Polling。** 当可用样本还不足一个 Batch 时，消费者可能持续发出 GET_META 请求，放大控制面的请求频率。
3. **状态表示和扫描方式。** 分区状态使用按索引跨度分配的稀疏 torch.int8 Tensor，某些准备状态检查会构造临时 Mask 并扫描较大的索引范围。
4. **元数据编解码。** 原有部分 BatchMeta、dtype 等元数据需要 Python 侧的序列化路径，增加请求热路径开销。

这说明潜在收益至少有三部分：**执行模型、数据结构、线协议**。即使不换语言，优化索引表示和轮询策略也可能带来收益；即使换成 C++ 或 Rust，若仍保留昂贵的扫描与锁竞争，也不能保证高吞吐。

## 三　Rust Controller 的热路径重构

[PR #221](https://github.com/Ascend/TransferQueue/pull/221) 和 [PR #223](https://github.com/Ascend/TransferQueue/pull/223) 分别涉及元数据线协议和 Rust Controller 的实现方向。

### Python Actor Shell 与 Rust 引擎的职责

方案没有要求把整个 RL 框架改写成 Rust。Ray Actor 仍承担启动、发现和配置等工作；Rust 引擎通过 PyO3 集成在同一个 Actor 进程内，处理高频 ZMQ 元数据请求。

这里的分工是：

- **Python / Ray：** 与现有 RL 框架集成，保留成熟的调度和应用 API。
- **Rust：** 负责对 CPU 热路径更敏感的协议处理、状态数据结构与并发请求执行。
- **PyO3：** 在两套运行时之间明确生命周期和数据调用边界；必要时调用 Python 侧能力。

这种设计不是“完全摆脱 Python”，而是**把最频繁的工作放在更合适的执行环境中**。自定义 Python Sampler 等能力仍需要考虑 GIL 和回调开销。PR #223 当前描述的 Rust Controller 初始实现以 SequentialSampler 为主，不能据此假设 RFC 中全部能力均已完成。

### Dense Row、Bitset 与并发请求处理

RFC 和 PR 描述的 Rust 实现将活跃样本映射到较稠密的本地 Row，并以 u64 Bitset 维护状态。与按最大索引跨度扫描 Tensor 相比，位图操作可以利用按机器字批量处理的方式，并减少索引稀疏度对工作量的影响。

理解这组性能结果的关键在于：**如果新旧实现的数据结构、编解码与并发模型都不同，就不能把全部收益归因于“语言从 Python 换成 Rust”。**

## 四　Controller 性能测试与结果分析

PR #223 提供了 Controller-only ZMQ Metadata RPC 的测试结果。在其报告的 **128 并发连接、Mixed Profile、Closed-loop** 场景下：

| 索引场景 | Python Controller | Rust Controller | 报告的吞吐提升 |
| --- | ---: | ---: | ---: |
| Dense，index span = 4,096 | 1,831 req/s | 50,846 req/s | 27.8× |
| Sparse，index span = 262,144 | 156 req/s | 50,103 req/s | 320.5× |

这些数字来自该 PR 的性能报告，**不是独立复测，也不是端到端 RL 训练加速比**。测试还限定了机器、CPU 亲和性、ZMQ 请求类型、连接数、索引布局与延迟测量方式。

稀疏索引场景的吞吐差异更为明显。根据实现说明，Python 路径的某些扫描成本依赖索引跨度，而 Rust 的 Dense Row / Bitset 设计更接近按活跃分区规模处理。因而这里比较的不只是语言性能，也包括**算法和状态结构的复杂度差异**。

如果后续要严谨分析，应补充单独消融：只换编码格式、只换状态结构、只换执行语言/并发模型时，吞吐与 P99 分别变化多少。当前公开的这些结果不足以精确拆分每一种改动的贡献。

## 五　Python、C++ 与 Rust 的系统选型

三种语言的讨论应该落到真实的组件边界，而不是做脱离工作负载的排行榜。

| 维度 | Python | C++ | Rust |
| --- | --- | --- | --- |
| 典型优势 | RL 生态、Ray 集成、快速试验 | 原生库生态、硬件/通信接口、成熟性能工程工具链 | 原生执行、所有权系统、并发状态管理 |
| 需要关注 | CPU 热路径、对象与序列化开销、GIL 适用边界 | 内存与资源生命周期、并发安全维护成本 | 学习曲线、绑定/构建复杂度、异步与跨语言交互 |
| 本案例中的角色 | 已存在的 Controller / Actor 外壳 | 值得作为选型对照，但没有同条件 Controller Benchmark | PR 中的原生 Controller 热路径 |

**上述测试不能证明 Rust 在相同工作负载下比 C++ 更快。** 它只能说明 TransferQueue 选择了 Rust 作为该控制面重构方案的一部分，以及这一实现组合在 PR 测试条件下报告了怎样的性能。

判断语言是否合适，至少还要看：团队能否长期维护、依赖与构建能否稳定交付、FFI 是否侵入应用接口，以及多语言边界到底放在何处。

## 六　数据面抽象与 PayloadTransfer

另一个方向见 [RFC #208](https://github.com/Ascend/TransferQueue/issues/208) 与 [PR #214](https://github.com/Ascend/TransferQueue/pull/214)。它们关注 SimpleStorage 的 PUT / GET / UPDATE 如何不依赖某个固定网络后端。

提案引入 **PayloadTransfer**，使 Manager 继续负责路由与结果组装，StorageUnit 继续负责字段读取、解析和写入，而传输适配器负责连接、消息协议、数据搬运与完成语义。

抽象之所以重要，是因为 ZMQ、UCX、NIXL 或 RDMA 相关方案的差别不只是更换一个 Socket：它们可能使用不同的 Buffer 注册、内存区域描述、直接写入最终输出、请求完成确认和失败恢复机制。

例如 GET 时，若接收方已经知道 Tensor 的布局并预先分配最终输出 Buffer，支持 Direct Placement 的后端可能直接填充指定 Span；但前提是要正确处理 **dtype / shape 校验、内存所有权、Buffer 生命周期、异步完成和失败后的重试**。接口不清晰，低层通信再快也可能造成数据覆盖或过早释放。

PR #214 当前主要引入抽象并适配既有 ZMQ 路径；**不能把它当作 UCX、NIXL 或 RDMA 已经全部完成集成**。其报告的双机测试重点是验证抽象引入后没有明显性能退化，而非宣称新的传输协议已经获得加速收益。

## 七　总结：控制面与数据面的架构取舍

TransferQueue 的这两条路线形成了一个有用的对照：

- **控制面重构**：高频请求集中在 Python 热路径，问题涉及串行处理、编解码和稀疏状态结构。解决思路是 Rust 引擎与数据结构、协议共同重构。
- **数据面解耦**：业务逻辑绑定既有 ZMQ 通信路径，问题是后续扩展不同传输后端困难。解决思路是先定义可替换的接口，再在可靠的内存与完成语义下实现不同后端。

基础设施性能问题可以沿以下顺序分析：先定位**控制面还是数据面**，再判断瓶颈来自**算法复杂度、数据表示、运行时并发还是通信带宽**，最后才决定用 Python、C++、Rust 或新的网络后端。

对 RL 后训练基础设施而言，真正重要的是让高频调度、样本状态管理与大规模 Payload 传输各自有清晰的所有权和扩展边界。

## 参考与进一步阅读

- [TransferQueue 项目](https://github.com/Ascend/TransferQueue)
- [RFC #220：并发 Rust Controller](https://github.com/Ascend/TransferQueue/issues/220)
- [PR #221：Native MessagePack Metadata](https://github.com/Ascend/TransferQueue/pull/221)
- [PR #223：Rust Controller](https://github.com/Ascend/TransferQueue/pull/223)
- [RFC #208：PayloadTransfer 接口](https://github.com/Ascend/TransferQueue/issues/208)
- [PR #214：SimpleStorage 传输层抽象](https://github.com/Ascend/TransferQueue/pull/214)

上述 RFC / PR 的实现范围和状态以对应链接为准；文中的性能数字均为其报告结果，未经独立复测。
