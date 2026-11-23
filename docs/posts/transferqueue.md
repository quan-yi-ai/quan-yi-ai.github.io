---
title: TransferQueue
description: 从 RL 训推分离的样本交付问题出发，解析 TransferQueue 的控制面、数据面、二维样本状态、DataFlow 编排、SimpleStorage 与 MooncakeStore、RDMA 和一步异步。
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
  <p class="asb-story-subtitle">从 RL 训推分离到流式数据调度与存储</p>
  <p class="asb-story-deck">在 Rollout、Reward 与 Trainer 分开运行时，谁负责决定哪些样本已经就绪、由谁消费，又怎样把真实 Tensor 高效地送到下一个任务？从控制面与数据面的分工理解 TransferQueue。</p>
</div>

TransferQueue 解决的不只是“把 Tensor 从 A 传到 B”的问题。它还需要知道一个样本的哪些字段已经被生产、哪个任务准备消费、怎样组织 Microbatch，以及数据存放在哪个 StorageUnit 上。理解这些职责，才能把它与普通消息队列、显存卸载工具和底层 RDMA 传输区分开。

本文先解释训推分离产生的数据需求，再沿 Controller → Metadata → StorageManager → Storage Backend 的路径理解完整的样本生命周期，最后讨论异步策略版本、存储后端与性能验证。


## 一、训推分离为什么需要数据系统

大模型强化学习后训练可采用共置式（colocated）或分离式（separated）架构。共置式通常在同一组资源上分时运行 Rollout 和 Update；分离式则把两类任务部署在不同资源上。

分离式架构是常见的工程选择之一，其核心是让 Actor Rollout 和 Actor Update 使用不同的计算资源。它的潜在动机包括提升资源利用效率、降低切换开销，以及复用现有训推集群；是否更优仍取决于负载与集群条件。

效率上：训练与推理的负载特性不同：训练往往有大量矩阵乘运算，而推理 Decode 阶段常受 KV Cache 访问与内存带宽影响。模型规模、批大小和序列长度会改变各自的瓶颈，因此资源分离可能更容易针对不同阶段独立调优。

在部分共置式实现中，Rollout 与 Update 交替运行还可能涉及模型状态加载、卸载与 Resharding，因为训练与推理的最优并行布局未必一致。模型越大、并行策略越复杂，这些切换成本越值得单独测量。

成本上：分离式架构有可能将RL框架作为一个顶层的调度模块，从而复用已有训推集群，并利用异构计算资源进行后训练，从而降低开销。

分离式架构的挑战不只有节点间传输，还包括把全局样本组织为能够立即消费的 Microbatch，以及对 On-policy 程度和策略版本差异进行管理。


## 二、TransferQueue 架构与样本读写

TransferQueue：一个具有全景视角、流式调度能力的数据系统

数据上游生产者将处理好的数据装箱发送至TransferQueue，TransferQueue将这些数据拆箱并保存起来，并在下游消费者请求数据时，重新打包并发送过去，实现计算任务间数据的动态路由。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：Rollout 生产经验，TransferQueue 管理样本状态，Trainer 按需消费"><img src="/diagrams/tq-figure-1.svg" alt="Rollout 生产经验，TransferQueue 管理样本状态，Trainer 按需消费" loading="lazy" /></button>
  <figcaption>Rollout 生产经验，TransferQueue 管理样本状态，Trainer 按需消费</figcaption>
</figure>

好处：

1.各个计算任务之间不再有显式的数据依赖关系

2.数据调度粒度变得更精细，仍然要凑够Micro Batch的数据

3.TransferQueue具有整个Global Batch的全景视角


### 控制平面与数据平面

将TransferQueue分为控制平面和数据平面两部分，控制平面用于管理训练数据的全局调度，而数据平面则以分布式的方式，存储真正的训练数据并进行高效传输。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：控制面负责就绪状态与元数据，数据面负责实际 Payload"><img src="/diagrams/tq-figure-2.svg" alt="控制面负责就绪状态与元数据，数据面负责实际 Payload" loading="lazy" /></button>
  <figcaption>控制面负责就绪状态与元数据，数据面负责实际 Payload</figcaption>
</figure>

为了实现流式、细粒度的数据管理，TransferQueue采用了一种二维数据结构：

每行对应一个训练样本，索引（Index）标识该样本在全局批次（Global Batch）中的位置。

每列代表对应RL任务的输入输出，如Prompts、Responses、Reference Log Prob等

这样设计是因为 在 RL 任务中，每个训练样本是由各个计算任务接力生成的。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：按样本索引和字段组织训练数据，逐步补齐就绪状态"><img src="/diagrams/tq-figure-3.svg" alt="按样本索引和字段组织训练数据，逐步补齐就绪状态" loading="lazy" /></button>
  <figcaption>按样本索引和字段组织训练数据，逐步补齐就绪状态</figcaption>
</figure>

在控制平面，通过一系列Metadata记录每条样本在每个阶段的生产消费情况。

以数据读取过程为例，我们介绍TransferQueue系统的交互流程：

1.某路DP完成计算任务，向这个任务对应的TransferQueue Controller发起数据读取请求

2.TransferQueue Controller扫描自身所存储的生产消费状态，筛选出可被这次数据读取请求消费的样本，并且基于负载均衡策略，动态组织成一个Micro-Batch，将对应Metadata返回给数据消费者。

3.数据消费者根据MetaData，向分布式存储平面发起数据读取请求，获取实际数据。


## 三、数据流调度与算法编排

基于 TransferQueue 这类具有全局 Metadata 视角的数据管理组件，算法主控可以采用 DataFlow 范式：

1. 异步启动各个 RL 计算任务，让它们按自身进度尝试消费所需字段。
2. 把第一个 Global Batch 的 Prompts 写入 TransferQueue。
3. 当对应样本的输入字段就绪时，各任务读取数据、执行计算，并写回新产生的字段；Controller 维护生产与消费状态。
4. 主控在当前 Global Batch 的训练阶段结束后记录指标、逐出旧样本，并加入下一轮 Prompts，继续推进迭代。

这种设计并不意味着完全不需要算法层同步：Iteration 边界、策略版本以及回收时机仍然需要由上层明确控制。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：DataFlow 方式让计算任务由字段就绪状态驱动"><img src="/diagrams/tq-figure-4.svg" alt="DataFlow 方式让计算任务由字段就绪状态驱动" loading="lazy" /></button>
  <figcaption>DataFlow 方式让计算任务由字段就绪状态驱动</figcaption>
</figure>

StreamRL、AReal 等方向也关注流式 Rollout 数据交付。与仅讨论 Rollout → Trainer 传输相比，TransferQueue 进一步强调对样本字段就绪、任务消费状态和不同计算阶段的数据依赖进行统一管理；各项目的实际能力仍应以所使用的版本和实现为准。

将 TransferQueue 读取逻辑封装成 PyTorch DataLoader / Iterator 接口后，训练端可以迭代获取 Microbatch，而无需直接处理 Controller 与 Storage Backend 的内部协议。

例如，对接 Megatron 训练后端时，上层消费的仍是符合训练逻辑的 Batch，底层则负责按需选样、路由与装配数据。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：通过 DataLoader 将流式样本组织为训练微批次"><img src="/diagrams/tq-figure-5.svg" alt="通过 DataLoader 将流式样本组织为训练微批次" loading="lazy" /></button>
  <figcaption>通过 DataLoader 将流式样本组织为训练微批次</figcaption>
</figure>

TransferQueue 使用 Ray 承载部分生命周期管理和分布式组件协作。理解整体性能时，需要区分 Ray 层的启动与发现开销、Controller 的 Metadata 热路径，以及 Storage Backend 的 Payload 传输成本。

以写入经验数据为例，`put_experience` 可以从概念上拆成三个步骤（具体接口与调用顺序以代码版本为准）：

1. 将 Batch 数据按 Sample 粒度整理。
2. 与 Controller 交互，确定样本索引与 Metadata。
3. 根据 Metadata 选择存储位置，把实际数据交给分布式 Storage Backend。

Controller 还可以通过采样 / 负载均衡策略，在读请求到来时可以根据所指定的负载均衡策略来提高训练效率。例如，在需要处理变长 Response 的场景中，可以按 Token 工作量分配给不同 DP Worker，以减少部分 Worker 长时间等待；具体是否启用及其收益需要结合所使用的框架实现验证。


## 四、策略版本与一步异步

在部分大模型 RL 训练配置中，策略版本不一定要求严格同步；是否允许滞后，取决于算法、数据使用方式和稳定性要求。


### 一步异步的策略版本约束

所谓“一步异步”，指的是训练和推理模型最多只差一个版本。更准确地说，是训练最新模型时的输入数据，是由上一个版本的推理模型所产生的。


## 五、数据平面实现：存储后端与传输路径

前文用一步异步说明了训练端会消费哪个版本的 Rollout 样本。样本选定后，还要解决另一个问题：Rollout、Reward 和 Trainer 分处不同进程或节点时，实际数据怎样交到下一个任务手里？

沿用前文的控制平面与数据平面分工：Controller 判断哪些字段已就绪，存储后端负责保存和传输实际数据。下文以 SimpleStorage 和 MooncakeStore 为例，追踪样本的读写路径，说明 RDMA 和零拷贝分别影响哪些环节。

比较两个后端时，重点看样本存放在哪一层、跨节点如何传输，以及数据进入训练设备前经过哪些拷贝。


### 样本字段与存储范围

以 GRPO / PPO 为例，推理侧先从 Prompt 采样 Response，计算或收集 Reward、Logprobs 等数据，训练侧再据此计算 Advantage、Loss、梯度并更新参数。在训推分离或异步运行时，生产和消费数据的进程、设备、甚至机器不同；这时仅靠某个 Python 进程中的列表，不足以表达全局样本生命周期。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：从 Prompt 到 Reward、Trainer 的经验样本生命周期"><img src="/diagrams/tq-figure-6.svg" alt="从 Prompt 到 Reward、Trainer 的经验样本生命周期" loading="lazy" /></button>
  <figcaption>从 Prompt 到 Reward、Trainer 的经验样本生命周期</figcaption>
</figure>

这里的“存储”不是保存模型权重或 Adam 优化器状态，而是保存经验样本（experience）及字段：prompt、response、reward、logprobs、mask 等。一个样本可能被多个阶段逐步补齐字段，训练端需要知道哪些字段已就绪。


### StorageManager 与可替换后端

上文的读写流程以元数据连接调度与实际数据传输。数据平面通过 StorageManager 封装后端差异，并提供 put_data、get_data 和 clear_data 等操作。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：统一 StorageManager API 与可替换的存储后端"><img src="/diagrams/tq-figure-7.svg" alt="统一 StorageManager API 与可替换的存储后端" loading="lazy" /></button>
  <figcaption>统一 StorageManager API 与可替换的存储后端</figcaption>
</figure>


```python
class StorageManager:  # 接口形状示意，签名以源码版本为准
    async def put_data(self, data, metadata): ...
    async def get_data(self, metadata): ...
    async def clear_data(self, metadata): ...
```

同一套控制逻辑可以选择不同 Backend。SimpleStorage、MooncakeStore 并非互相对立的 RL 算法，而是针对不同部署条件实现的可替换数据承载方案。项目还列出了 Yuanrong、RayRDT 等选择。


### SimpleStorage 的数据路径

SimpleStorage 是 TransferQueue 默认的基础 CPU 内存存储后端。它以多个 SimpleStorageUnit 组织样本数据，AsyncSimpleStorageManager 负责协调；Storage Unit 可分布在不同 Ray 节点上。底层配置说明其基于 ZMQ 的内存通信，重点在易用和较少的额外依赖，而不是承诺完全零拷贝。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：多 StorageUnit 中的样本与字段分布"><img src="/diagrams/tq-figure-8.svg" alt="多 StorageUnit 中的样本与字段分布" loading="lazy" /></button>
  <figcaption>多 StorageUnit 中的样本与字段分布</figcaption>
</figure>

一种有用的理解方式是二维表：每个 row 对应一个样本 ID，每个 column 对应任务产出的某个字段。Rollout 阶段写入 response，Reward 阶段补充 score，Trainer 只在需要的时刻读取对应字段。实际对象组织与 API 版本可能演进，但这一行列模型是文档明确描述的设计动机。


#### SimpleStorage 的适用场景与瓶颈

- 优点：依赖相对少、部署与调试成本低；很适合本地调试、功能验证和初期分布式训练。
- 限制：数据主要落在 CPU DRAM；跨节点传输、序列化和多次 Host↔GPU 拷贝可能成为吞吐瓶颈。
- 不要混淆：SimpleStorage 叫 Simple，不代表只有单机一个 Python dict；它可以使用多个分布式 Storage Unit。

### MooncakeStore 的数据路径

MooncakeStore 是 TransferQueue 的高性能 KV 型存储 Backend，支持层次化存储及可配置的 TCP/RDMA 数据传输。它通常需要额外的 Mooncake 软件组件、元数据协调服务、内存段配置以及网络能力。文档将其标为 beta，具体接口和配置应以所用版本为准。

MooncakeStore 更像一个可远程访问的共享数据层：写入方以 Key 标识数据，读取方使用对应 Key/元信息找到数据。和 SimpleStorage 一样，训练框架可以通过上层统一接口访问它，但底层的数据放置、迁移和传输路径不同。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大技术图：SimpleStorage 与 MooncakeStore 的概念数据路径"><img src="/diagrams/tq-figure-9.svg" alt="SimpleStorage 与 MooncakeStore 的概念数据路径" loading="lazy" /></button>
  <figcaption>SimpleStorage 与 MooncakeStore 的概念数据路径</figcaption>
</figure>


#### MooncakeStore 的核心配置


```yaml
backend:
  storage_backend: MooncakeStore
  MooncakeStore:
    protocol: rdma          # 也可配置为 tcp
    use_gdr: false         # GPU Direct RDMA 另有硬件要求
    metadata_server: localhost:50050
    master_server_address: localhost:50051
```

这是用于理解的配置摘录，不保证可直接运行。特别是协议默认为 TCP 的配置示例中，不能仅凭启用了 MooncakeStore 就宣称正在使用 RDMA；开启 GPU Direct RDMA 还要求支持它的 GPU/NIC 与正确初始化。


### RDMA、零拷贝与 GPUDirect RDMA

<div class="asb-table-kicker">RDMA、零拷贝与 GPUDirect 的区别</div>

| 概念 | 含义 | 不能直接推断的事 |
| --- | --- | --- |
| RDMA | 在支持的网络、内存注册与协议条件下进行远程内存访问 | 使用 RDMA 不代表全链路零拷贝 |
| CPU Host Buffer | 数据可先经过可访问、可注册的 CPU 内存缓冲区 | 可能仍存在 CPU 复制或布局转换 |
| GPUDirect RDMA | 在硬件与驱动支持下，让 NIC 直接访问 GPU 相关内存 | 不代表应用层没有序列化、Staging 或额外拷贝 |
| End-to-end Zero Copy | 需要逐段确认数据路径与内存所有权 | 不能只看网络协议名称作出判断 |

例如：即使网络阶段使用 RDMA，某个 Tensor 也可能先被复制到 CPU 的已注册缓冲区，再经网络传输。真正想证明“GPU 到 GPU 零拷贝”，需要检查每个缓冲区的位置、数据转换/序列化步骤，以及源和目的端的 GPU Direct 路径。

TransferQueue 的 Mooncake GDR 文档明确指出：GDR 是可选路径，依赖 protocol: rdma、use_gdr: true 以及硬件支持；它还会为每个启用 GDR 的进程准备固定 GPU staging buffer。因而即使使用 GDR，也不应把整条训练数据流水线简化成“永远没有拷贝”。


## 六、后端选型与性能验证

<div class="asb-table-kicker">SimpleStorage 与 MooncakeStore 对比</div>

| 评估项 | SimpleStorage | MooncakeStore |
| --- | --- | --- |
| 核心定位 | 易使用的 CPU 内存存储后端 | 支持分层存储和可配置网络传输的 KV 后端 |
| 网络路径 | 原有实现以 ZMQ 为主 | 可根据配置选择 TCP / RDMA |
| 部署前提 | 相对较少的外部依赖 | 需要额外组件、元信息服务与内存段配置 |
| 重点核对 | CPU 拷贝、序列化、网络瓶颈 | RDMA/GDR 是否真的启用、注册与缓冲区开销 |
| 性能结论 | 需结合实测 | 不能仅凭后端名称断言一定更快 |


### 性能验证方法

不要只比“成功启动”或“某一段网络带宽”。建议固定模型、采样数量、响应长度、卡数、节点与网络，观测：Rollout→Trainer 交付延迟（p50/p95）、单位时间可交付样本数、CPU DRAM 峰值、GPU 占用、序列化耗时、网络吞吐和 Trainer 等待数据的空闲时间。还要区分冷启动与稳态，避免把元数据服务启动时间混进每批传输耗时。


## 七、样本存储与显存卸载的边界

<div class="asb-table-kicker">样本存储与显存卸载对比</div>

| 对象 | TransferQueue Storage Backend | TMS / Selective Offload |
| --- | --- | --- |
| 管理内容 | Prompt、Response、Reward、Logprobs 等经验样本字段 | 模型参数或其他参与训练/推理的 GPU 内存占用 |
| 主要目标 | 解耦生产者与消费者，提供可靠的数据存储、调度与传输 | 控制模型内存驻留，支持训推切换或资源复用 |
| 核心问题 | 样本何时就绪、由谁消费、怎样跨节点移动 | 何时卸载 / 恢复、地址与缓冲区如何保持有效 |
| 二者关系 | 可以与内存管理工具同时使用 | 不替代经验样本的数据系统 |

这实际上是两个正交维度：TransferQueue 的 Storage Backend 处理“样本如何从生产者送到消费者”；TMS/Selective Offload 处理“训推切换时哪些模型显存应当释放、如何恢复”。即使同时部署，也不能把 MooncakeStore 当成模型权重卸载器，或把 TMS 当成经验数据队列。


## 八、源码阅读路径与检查清单

先从 README 和配置文件确认可选后端，再阅读 StorageManager 接口，最后沿 SimpleStorage 和 MooncakeStore 的实现追踪样本的读写路径。

- ① README：确认 StorageManager 的统一操作以及后端支持列表。
- ② config.yaml：定位 storage_backend、SimpleStorage、MooncakeStore 的配置项。
- ③ SimpleStorageUnit / AsyncSimpleStorageManager：追踪样本索引如何落到分布式单元。
- ④ MooncakeStoreClient：理解 KV 批量传输、shape/dtype 元信息与客户端缓冲区。
- ⑤ Mooncake GDR 指南：确认何时初始化 GPU、协议要求以及 staging buffer 生命周期。
- ⑥ 训练侧调用点：观察 Trainer 哪些阶段等待 get_data、哪些字段参与 Loss 计算。

## 九、阅读自测

- Rollout 完成后，还需要哪些字段就绪，Trainer 才能读取完整的训练样本？
- 为什么 SimpleStorage 也可以是多节点分布式系统？
- 如何区分样本元信息调度和实际 Tensor 数据传输？
- 为什么 MooncakeStore 不是默认就开启 RDMA？
- 为什么“RDMA”与“GPU Direct Zero-Copy”不能画等号？
- TMS 显存卸载与 TransferQueue 的存储后端各自管理什么数据？

## 参考资料与延伸阅读

TransferQueue README：[github.com/Ascend/TransferQueue](https://github.com/Ascend/TransferQueue)

TransferQueue 配置文件（默认 SimpleStorage、MooncakeStore 配置）：[github.com/Ascend/TransferQueue/blob/main/transfer_queue/config.yaml](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/config.yaml)

MooncakeStore Client 源码：[github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/clients/mooncake_client.py](https://github.com/Ascend/TransferQueue/blob/main/transfer_queue/storage/clients/mooncake_client.py)

Mooncake GPUDirect RDMA 使用说明：[github.com/Ascend/TransferQueue/blob/main/docs/storage_backends/mooncake_gdr.md](https://github.com/Ascend/TransferQueue/blob/main/docs/storage_backends/mooncake_gdr.md)

verl TransferQueue 文档：[github.com/verl-project/verl/blob/main/docs/data/transfer_queue.md](https://github.com/verl-project/verl/blob/main/docs/data/transfer_queue.md)

Relax Selective Offload PR #85：[github.com/redai-studio/Relax/pull/85](https://github.com/redai-studio/Relax/pull/85)

这些架构图用于解释数据流与组件边界，不代表运行时 Profiling 结果；关于吞吐、延迟和零拷贝的结论，应以具体设备与部署条件下的测试为准。


## 相关专题：控制面的 Rust 重构

当样本管理从功能正确进一步走向高并发，Controller 的 Metadata 处理与 SimpleStorage 的网络协议抽象会成为新的工程议题。可继续阅读：[从TransferQueue重构理解python、cpp、Rust](/posts/transferqueue-python-cpp-rust)，从高频 RPC、PyO3、Bitset 与 PayloadTransfer 的角度理解两条重构路径。
