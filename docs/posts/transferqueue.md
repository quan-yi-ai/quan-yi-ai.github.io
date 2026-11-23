---
title: TransferQueue
description: 从一条 RL 经验样本的写入、状态更新到微批消费，理解 TransferQueue 的控制面、数据面、异步编排和跨节点存储路径。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="transferqueue" tabindex="-1">TransferQueue</h1>
  <p class="asb-story-subtitle">从一条经验样本出发，拆解 RL 流式数据系统</p>
  <p class="asb-story-deck">在 Rollout、Reward 与 Trainer 分开运行时，谁负责决定哪些样本已经就绪、由谁消费，又怎样把真实 Tensor 高效地送到下一个任务？从控制面与数据面的分工理解 TransferQueue。</p>
</div>

TransferQueue 解决的不只是“把 Tensor 从 A 传到 B”的问题。它还需要知道一个样本的哪些字段已经被生产、哪个任务准备消费、怎样组织 Microbatch，以及数据存放在哪个 StorageUnit 上。理解这些职责，才能把它与普通消息队列、显存卸载工具和底层 RDMA 传输区分开。

本文先解释训推分离产生的数据需求，再沿 Controller → Metadata → StorageManager → Storage Backend 的路径理解完整的样本生命周期，最后讨论异步策略版本、存储后端与性能验证。


## 一、训推分离与 RL 数据管理需求

在一次 RL 后训练迭代中，Rollout 生成 Response，Reward 计算反馈，Trainer 消费相应字段并更新模型。它们可以部署在同一组 GPU 上分时运行，也可以由独立进程甚至独立集群运行。

分离式架构是常见的工程选择之一，其核心是让 Actor Rollout 和 Actor Update 使用不同的计算资源。它的潜在动机包括提升资源利用效率、降低切换开销，以及复用现有训推集群；是否更优仍取决于负载与集群条件。

从计算特性来看，训练往往有大量矩阵乘运算，而推理 Decode 阶段常受 KV Cache 访问与内存带宽影响。模型规模、批大小和序列长度会改变各自的瓶颈，因此资源分离可能更容易针对不同阶段独立调优。

在部分共置式实现中，Rollout 与 Update 交替运行还可能涉及模型状态加载、卸载与 Resharding，因为训练与推理的最优并行布局未必一致。模型越大、并行策略越复杂，这些切换成本越值得单独测量。

从部署成本来看，分离式架构有机会复用既有训推资源，采用不同计算设备；能否降低总成本取决于资源利用率以及额外的传输与同步开销。

分离式架构的挑战不只有节点间传输，还包括把全局样本组织为能够立即消费的 Microbatch，以及对 On-policy 程度和策略版本差异进行管理。

数据管理的难点不仅在于跨设备传输，还在于维护下面这种状态：**同一个 Global Batch 内，某些样本已经产生 Response，某些已经有 Reward，另一些还在生成；训练端需要从中选出所需字段都就绪的样本。**

因此，系统需要明确三类职责：谁维护全局样本状态、谁决定哪个任务消费哪些样本、谁负责传输真实 Tensor。TransferQueue 正是在这些边界上提供统一的数据管理机制。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：Rollout、Reward 和 Trainer 使用 TransferQueue 交接样本字段"><img src="/diagrams/tq-figure-1.svg" alt="Rollout、Reward 和 Trainer 使用 TransferQueue 交接样本字段" loading="lazy" /></button>
  <figcaption>计算任务与数据系统解耦：生产者写入样本，消费者按字段就绪状态读取</figcaption>
</figure>

## 二、TransferQueue 的控制面与数据面

TransferQueue 将**控制面**与**数据面**分离。控制面由 Controller 跟踪分区、样本索引、各字段的生产/消费状态与 Batch 选择；数据面由 StorageManager 与 StorageUnit 承载字段的存储、读取和传输。Ray 可以承担进程启动、发现等基础设施职责，但频繁的数据读写不等于每次都经过 Ray 对象传输。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：TransferQueue 的控制面和数据面"><img src="/diagrams/tq-figure-2.svg" alt="TransferQueue 的控制面和数据面" loading="lazy" /></button>
  <figcaption>Controller 负责逻辑状态，StorageManager / StorageUnit 负责实际数据路径</figcaption>
</figure>

### 样本索引、字段就绪与消费状态

把一个 Global Batch 看成一张逻辑表：**行是样本索引，列是 Prompt、Response、Reward、Logprobs 等字段**。不同任务接力生产字段，Controller 维护相应状态；真实字段值存放在数据面，而不是把所有 Tensor 都塞进 Metadata。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：四个样本的字段就绪和消费状态矩阵"><img src="/diagrams/tq-sample-lifecycle.svg" alt="四个样本的字段就绪和消费状态矩阵" loading="lazy" /></button>
  <figcaption>同一时刻不同样本可能具有不同的字段就绪状态；图中数据是教学示例</figcaption>
</figure>

例如，某 Trainer 要求字段 `response`、`reward` 和 `logprobs`，那么仅有 Prompt 或 Response 的样本都不能直接满足该请求。**Ready 不是一个笼统的“样本完成”布尔值，而要结合请求字段和消费任务判断。** 消费状态也需要区分 Task，不能把一个任务消费过直接理解成所有下游任务都不可再读。

## 三、PUT / GET 请求的完整执行路径

这是理解 TransferQueue 最重要的一节。下面基于项目现有的 [TransferQueueClient](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/client.py)、[Python Controller](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/controller.py) 和 [AsyncSimpleStorageManager](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/managers/simple_storage_manager.py) 路径，展示基本的调用关系；省略异常恢复、字段链接、定制解析器和不同后端的分支。

### 写入：分配索引、存储字段、通知状态

1. **生产者发起写入**。调用客户端的 `async_put(data, metadata=...)`；当没有 Metadata 时，客户端可以借助 Controller 的 `GET_META(mode="insert")` 获取或分配 Global Index。
2. **数据面选择 StorageUnit**。SimpleStorageManager 根据全局索引路由，把字段值发送给对应的 StorageUnit。
3. **控制面更新字段状态**。写入完成后的 Metadata / 生产状态通知使消费者能判断哪些字段已经就绪；要区分实际 Payload 落盘（或落内存）和 Metadata 就绪这两个事件。

这里的 `BatchMeta` 不是样本内容本身。它记录索引、字段 Schema 等信息，为后续读写提供依据。

### 读取：先获取 Metadata，再获取真实数据

读取操作首先向 Controller 请求满足条件的 Batch Metadata：

```python
# 示例调用形状来自 TransferQueueClient API；省略 tq.init() 等初始化
meta = await client.async_get_meta(
    data_fields=["response", "reward", "logprobs"],
    batch_size=4,
    partition_id="train_0",
    mode="fetch",
    task_name="trainer",
)
if meta.size:
    batch = await client.async_get_data(meta)
```

对应的核心源码职责：

| 代码入口 | 关键职责 |
| --- | --- |
| [`TransferQueueClient.async_get_meta`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/client.py#L243) | 向 Controller 发送 `GET_META` ZMQ 请求并接收 `BatchMeta` |
| [`TransferQueueController.get_meta`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/controller.py#L1413) | `fetch` 时筛选字段已 Ready 且未被当前任务消费的索引，运行 Sampler 并生成 Metadata |
| [`TransferQueueClient.async_get_data`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/client.py#L566) | 根据 `BatchMeta` 调用 `storage_manager.get_data()` |
| [`AsyncSimpleStorageManager.get_data`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/managers/simple_storage_manager.py#L816) | 路由到 StorageUnit，处理字段读取与返回数据组装 |
| [`StorageUnitData.get_data`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/simple_storage.py#L220) | 按字段名和 Global Index 读取对应数据 |

如果尚未凑齐请求的 Batch，`fetch` 在一般情况下可能返回空 `BatchMeta`，等待后续生产者完成写入；并不是每次请求都会立即收到可训练的 Microbatch。不同 Sampler 还可能有缓存结果或可变大小 Batch 的行为。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：Producer、Controller、StorageManager 和 StorageUnit 的 PUT GET 时序图"><img src="/diagrams/tq-put-get-sequence.svg" alt="Producer、Controller、StorageManager 和 StorageUnit 的 PUT GET 时序图" loading="lazy" /></button>
  <figcaption>先取 Metadata，再取 Payload；写入同样区分索引状态与实际字段的传输</figcaption>
</figure>

### 多 StorageUnit 路由与结果顺序恢复

假设一次 GET 请求的 Global Index 顺序是 `[12, 7, 4, 9]`。使用两个 SimpleStorageUnit 且按 `global_index % 2` 路由：SU0 接收 `[12, 4]`，SU1 接收 `[7, 9]`。数据可能先后从不同 Unit 返回，但消费者预期的 Batch 顺序仍是 `[12, 7, 4, 9]`。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：Global Index 通过取模路由至两个 StorageUnit 并按原始位置组装"><img src="/diagrams/tq-storage-routing.svg" alt="Global Index 通过取模路由至两个 StorageUnit 并按原始位置组装" loading="lazy" /></button>
  <figcaption>示例使用两个 Unit，演示按索引路由和输出位置恢复；不是固定的物理集群拓扑</figcaption>
</figure>

这段逻辑在 [`_group_by_hash`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/managers/simple_storage_manager.py#L291) 中明确可见：一边记录各 Unit 需要的 `global_indexes`，一边保留它们在原 Batch 的 `batch_positions`。**并发读写不仅要搬对数据，还要把它放回正确位置。**

## 四、基于 DataFlow 的异步任务编排

有了样本索引、字段就绪状态和按需读取机制，就可以让不同阶段的计算任务按自身进度推进，而不要求每次都由上游直接调用下游 RPC。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：从 Prompt 到 Rollout Reward Trainer 的 DataFlow"><img src="/diagrams/tq-figure-4.svg" alt="从 Prompt 到 Rollout Reward Trainer 的 DataFlow" loading="lazy" /></button>
  <figcaption>字段就绪驱动任务消费，但 Iteration 边界仍由上层管理</figcaption>
</figure>

以一个 Global Batch 的生命周期为例：

1. 主控启动 Rollout、Reward、Trainer 等 Worker，再向数据系统写入 Prompts。
2. Rollout 读到需要的 Prompt 后产生 Response、Logprobs 等字段，写回对应样本。
3. Reward Worker 在所需字段就绪时读取数据、产生 Reward；Trainer 按所需字段和 Sampler 策略获取 Microbatch。
4. 主控按算法要求管理本轮训练完成条件、指标、旧数据回收与下一轮 Prompts。

对训练侧，可以把上述读取逻辑进一步包装成 DataLoader / Iterator，让模型训练代码关注 Batch，而不必直接了解 Controller 与 StorageUnit 的通信协议。

### 策略版本与一步异步约束

DataFlow 让**任务执行时间**更灵活，但不自动保证 On-policy 条件。所谓“一步异步”，通常是让训练使用由上一策略版本生成的数据；它是一种需要明确约束的调度方式，并非 TransferQueue 的默认语义或所有算法都能无条件接受的策略。

例如要在训练更新模型时继续运行 Rollout，就必须知道每条样本由哪一个策略版本生成，并制定样本准入、权重更新和旧数据清理规则。**数据系统提供状态与交付能力，算法层决定可接受的策略滞后。**

### Microbatch 采样与负载均衡

Response 长度不同会造成每个 DP Worker 的 Token 工作量不同。合理的采样和 Microbatch 分配有机会减少长尾等待，而不仅是平均分配相同数目的样本。能否实现及收益大小取决于 Sampler 的具体实现、元数据字段和运行配置。

## 五、存储后端：SimpleStorage 与 MooncakeStore

完成 Batch 选择之后，下一问题是**真实 Tensor 存放在哪里、怎么跨节点交付**。StorageManager 把上层请求与不同 Backend 分开，数据面并不是 Controller 内部的一个普通 Python List。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：StorageManager 对多个 Backend 暴露统一数据操作"><img src="/diagrams/tq-figure-7.svg" alt="StorageManager 对多个 Backend 暴露统一数据操作" loading="lazy" /></button>
  <figcaption>相同逻辑数据请求可以交由不同后端处理，但它们的内存与网络路径不同</figcaption>
</figure>

### SimpleStorage 的索引存储与访问

SimpleStorage 是 TransferQueue 常用的基础存储后端，经典路径以 CPU 内存为主。它以多个 SimpleStorageUnit 组织样本数据，AsyncSimpleStorageManager 负责协调；Storage Unit 可分布在不同 Ray 节点上。其基础路径采用 ZMQ 通信，并不承诺端到端零拷贝；部分版本还提供可选的 SSD 扩展，应根据实际配置辨别数据驻留位置。

在基础实现里，`StorageUnitData.put_data` 以 Global Index 为 Key 写入每个字段，`get_data` 根据字段名与索引读回结果。多个 StorageUnit 分担存储与网络请求，并不等于每个 Unit 都保存一份完整的 Global Batch。


#### 适用场景与瓶颈

- 优点：依赖相对少、部署与调试成本低；很适合本地调试、功能验证和初期分布式训练。
- 限制：经典内存路径主要使用 CPU DRAM；跨节点传输、序列化和 Host↔GPU 拷贝可能成为吞吐瓶颈，启用可选 SSD 路径时还需分析磁盘 I/O。
- **分布式能力**：SimpleStorage 可通过多个 StorageUnit 分担存储与请求，并不局限于单机字典。

### MooncakeStore：分层与远程传输

MooncakeStore 是 TransferQueue 的高性能 KV 型存储 Backend，支持层次化存储及可配置的 TCP/RDMA 数据传输。它通常需要额外的 Mooncake 软件组件、元数据协调服务、内存段配置以及网络能力。文档将其标为 beta，具体接口和配置应以所用版本为准。

MooncakeStore 更像一个可远程访问的共享数据层：写入方以 Key 标识数据，读取方使用对应 Key/元信息找到数据。和 SimpleStorage 一样，训练框架可以通过上层统一接口访问它，但底层的数据放置、迁移和传输路径不同。


#### 配置示意与必要条件

```yaml
backend:
  storage_backend: MooncakeStore
  MooncakeStore:
    protocol: rdma          # 也可配置为 tcp
    use_gdr: false         # GPU Direct RDMA 另有硬件要求
    metadata_server: localhost:50050
    master_server_address: localhost:50051
```

上述配置用于说明关键参数，并非完整部署文件。MooncakeStore 的实际传输协议以配置为准：启用该后端不意味着必然使用 RDMA，GPUDirect RDMA 还依赖受支持的 GPU、NIC、驱动与初始化条件。

### 存储后端的选型维度

<div class="asb-table-kicker">SimpleStorage 与 MooncakeStore 对比</div>

| 评估项 | SimpleStorage | MooncakeStore |
| --- | --- | --- |
| 核心定位 | 易使用的 CPU 内存存储后端 | 支持分层存储和可配置网络传输的 KV 后端 |
| 网络路径 | 原有实现以 ZMQ 为主 | 可根据配置选择 TCP / RDMA |
| 部署前提 | 相对较少的外部依赖 | 需要额外组件、元信息服务与内存段配置 |
| 重点核对 | CPU 拷贝、序列化、网络瓶颈 | RDMA/GDR 是否真的启用、注册与缓冲区开销 |
| 性能结论 | 需结合实测 | 不能仅凭后端名称断言一定更快 |


选择后端不能仅看“是否支持 RDMA”。需要结合真实的字段类型、单次 Payload 大小、并发量、网络资源、部署维护成本和训练端的等待时间。

## 六、RDMA、零拷贝与性能验证

分析跨节点数据交付效率，需要分别明确四个问题：**源数据位于哪块内存、有没有中间 Buffer、网络使用哪种协议、最终 Tensor 在哪块设备上供训练消费。**

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大图片：Tensor 从生产端到网络、接收端的可能内存路径"><img src="/diagrams/tq-memory-path.svg" alt="Tensor 从生产端到网络、接收端的可能内存路径" loading="lazy" /></button>
  <figcaption>启用 RDMA 不等于全流程没有 CPU Staging、布局转换和额外拷贝</figcaption>
</figure>

<div class="asb-table-kicker">RDMA、零拷贝与 GPUDirect 的区别</div>

| 概念 | 含义 | 不能直接推断的事 |
| --- | --- | --- |
| RDMA | 在支持的网络、内存注册与协议条件下进行远程内存访问 | 使用 RDMA 不代表全链路零拷贝 |
| CPU Host Buffer | 数据可先经过可访问、可注册的 CPU 内存缓冲区 | 可能仍存在 CPU 复制或布局转换 |
| GPUDirect RDMA | 在硬件与驱动支持下，让 NIC 直接访问 GPU 相关内存 | 不代表应用层没有序列化、Staging 或额外拷贝 |
| End-to-end Zero Copy | 需要逐段确认数据路径与内存所有权 | 不能只看网络协议名称作出判断 |

例如：即使网络阶段使用 RDMA，某个 Tensor 也可能先被复制到 CPU 的已注册缓冲区，再经网络传输。真正想证明“GPU 到 GPU 零拷贝”，需要检查每个缓冲区的位置、数据转换/序列化步骤，以及源和目的端的 GPU Direct 路径。

TransferQueue 的 Mooncake GDR 文档明确指出：GDR 是可选路径，依赖 protocol: rdma、use_gdr: true 以及硬件支持；它还会为每个启用 GDR 的进程准备固定 GPU staging buffer。因而即使使用 GDR，也不应把整条训练数据流水线简化成“永远没有拷贝”。

### 性能验证方法

性能评估应固定模型、采样数量、响应长度、GPU 数、节点和网络条件，综合观测：Rollout→Trainer 交付延迟（p50/p95）、单位时间可交付样本数、CPU DRAM 峰值、GPU 占用、序列化耗时、网络吞吐和 Trainer 等待数据的空闲时间。还要区分冷启动与稳态，避免把元数据服务启动时间混进每批传输耗时。

后续验证建议将延迟拆成三个层次：Controller Metadata RPC 延迟、Storage Backend 传输延迟，以及从 Rollout 完成到 Trainer 实际可用 Batch 的**端到端交付延迟**。一项后端网络加速不一定等价于训练吞吐同比例提高。

## 七、TransferQueue 核心源码路径

下列源码入口分别对应前文讨论的元数据选择、样本路由和字段访问流程。

| 阅读顺序 | 代码 | 应当确认的问题 |
| --- | --- | --- |
| 1 | [`interface.py`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/interface.py) | 对上层训练/推理 Worker 暴露了哪些 API？ |
| 2 | [`client.py`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/client.py) | `async_get_meta`、`async_get_data` 与 `async_put` 如何分离请求？ |
| 3 | [`controller.py`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/controller.py) | `fetch` 如何判断 Ready/Consumed，并交给 Sampler？ |
| 4 | [`simple_storage_manager.py`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/managers/simple_storage_manager.py) | Global Index 如何路由到 Unit、Batch 顺序如何恢复？ |
| 5 | [`simple_storage.py`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/simple_storage.py) | 字段值如何按 Index 真正写入和取回？ |
| 6 | [`config.yaml`](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/config.yaml) | 实际使用何种 Backend、协议和内存选项？ |

**阅读时要注意版本边界。** 这些路径和行为根据本次查阅的 GitHub 默认分支实现整理；未来重构可能使 Controller 的语言、模块位置和部分接口发生变化。项目代码的提交版本和运行配置决定实际行为。

## 八、总结：TransferQueue 的能力与系统边界

TransferQueue 的核心价值可以归结为三件事：用全局样本索引和字段状态管理生产/消费；让不同计算任务按需组装 Microbatch；使数据路由与具体的 Storage Backend 解耦。

但它**不自动保证**所有 RL 算法的 On-policy 条件，不自动让所有 Backend 具备 GPUDirect 或端到端零拷贝，也不会替代模型权重和优化器状态的显存管理。

### 样本存储与模型显存管理的区别

<div class="asb-table-kicker">样本存储与显存卸载对比</div>

| 对象 | TransferQueue Storage Backend | TMS / Selective Offload |
| --- | --- | --- |
| 管理内容 | Prompt、Response、Reward、Logprobs 等经验样本字段 | 模型参数或其他参与训练/推理的 GPU 内存占用 |
| 主要目标 | 解耦生产者与消费者，提供可靠的数据存储、调度与传输 | 控制模型内存驻留，支持训推切换或资源复用 |
| 核心问题 | 样本何时就绪、由谁消费、怎样跨节点移动 | 何时卸载 / 恢复、地址与缓冲区如何保持有效 |
| 二者关系 | 可以与内存管理工具同时使用 | 不替代经验样本的数据系统 |

这实际上是两个正交维度：TransferQueue 的 Storage Backend 处理“样本如何从生产者送到消费者”；TMS/Selective Offload 处理“训推切换时哪些模型显存应当释放、如何恢复”。即使同时部署，也不能把 MooncakeStore 当成模型权重卸载器，或把 TMS 当成经验数据队列。

性能诊断需要区分 Trainer 等待 Controller 选择样本与等待 Storage Backend 搬运 Tensor 两种情况。是由于字段就绪不齐导致 Batch 被阻塞，还是网络与拷贝真的达到了瓶颈？这些问题对应不同的优化方向。

### 延伸：Controller 与传输层重构

若要进一步研究 Python Controller 在高并发 Metadata 请求中的瓶颈、Rust/PyO3 重构以及可插拔传输层，可继续阅读 [《从TransferQueue重构理解python、cpp、Rust》](/posts/transferqueue-python-cpp-rust)。本文讨论的是数据系统的核心机制，下一篇讨论的是它在更高并发与不同传输后端下的工程扩展。

## 参考资料

TransferQueue README：[github.com/Ascend/TransferQueue](https://github.com/Ascend/TransferQueue)

TransferQueue 配置文件（默认 SimpleStorage、MooncakeStore 配置）：[github.com/Ascend/TransferQueue/blob/main/transfer_queue/config.yaml](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/config.yaml)

MooncakeStore Client 源码：[github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/clients/mooncake_client.py](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/clients/mooncake_client.py)

Mooncake GPUDirect RDMA 使用说明：[github.com/Ascend/TransferQueue/blob/main/docs/storage_backends/mooncake_gdr.md](https://github.com/Ascend/TransferQueue/blob/main/docs/storage_backends/mooncake_gdr.md)

verl TransferQueue 文档：[github.com/verl-project/verl/blob/main/docs/data/transfer_queue.md](https://github.com/verl-project/verl/blob/main/docs/data/transfer_queue.md)

Relax Selective Offload PR #85：[github.com/redai-studio/Relax/pull/85](https://github.com/redai-studio/Relax/pull/85)

这些架构图用于解释数据流与组件边界，不代表运行时 Profiling 结果；关于吞吐、延迟和零拷贝的结论，应以具体设备与部署条件下的测试为准。
