---
title: Ray分布式计算框架
description: 从 Task、Actor、ObjectRef 到 Raylet、GCS、Object Store，再到 Ray + vLLM 的分布式推理与故障排查。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="ray分布式计算框架" tabindex="-1">Ray分布式计算框架</h1>
  <p class="asb-story-subtitle">从 Task、Actor、ObjectRef 到分布式推理资源编排</p>
  <p class="asb-story-deck">将一个 Python 函数放到多台机器上执行，除了远程调用，还需要决定任务在哪里运行、结果在哪里保存、依赖如何传递，以及 Worker 失败以后怎样恢复。沿一组真实代码追踪 Ray 的分布式执行机制。</p>
</div>

Ray 提供了一套跨进程与跨节点的执行抽象：**Task** 负责一次性远程计算，**Actor** 提供持久状态，**ObjectRef** 表示结果引用与数据依赖。理解这三者之后，才能解释 Raylet、GCS、Object Store 和 Ray 在 vLLM 分布式推理中的作用。

## 一、Ray Core：Task、Actor 与 ObjectRef

### Task：无状态远程任务

```python
import ray

ray.init()

@ray.remote
def square(x):
    return x * x

refs = [square.remote(i) for i in range(4)]
print(ray.get(refs))  # [0, 1, 4, 9]
```

`square.remote(i)` 返回的是一个 **ObjectRef**，而不是函数执行结果，也不是普通任务 ID。Driver 可以先提交多个独立的 Task，再用 `ray.get` 等待结果。这类任务适合数据预处理、独立评估或仿真。远程执行不保证一定与串行运行相比更快；细粒度任务还要考虑调度、序列化和对象传输成本。

### Actor：有状态的远程执行单元

```python
@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1
        return self.value

counter = Counter.remote()
print(ray.get(counter.inc.remote()))  # 1
print(ray.get(counter.inc.remote()))  # 2
```

Actor 通过远程方法操作自己的状态，适合模型 Worker、持续运行的服务或参数状态。**状态持续存在不等于每个方法都可以并发执行**：普通 Actor、异步 Actor 和并发组的执行语义不一样。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Ray Task Actor 生命周期图"><img src="/diagrams/ray-task-actor.svg" alt="Ray Task 一次执行与 Actor 有状态进程对比" /></button>
  <figcaption>Task 与 Actor：一次性远程执行与持续维护状态的远程进程。</figcaption>
</figure>

### ObjectRef 与任务依赖

```python
@ray.remote
def produce():
    return [1, 2, 3]

@ray.remote
def consume(values):
    return sum(values)

ref_a = produce.remote()
ref_b = consume.remote(ref_a)
print(ray.get(ref_b))  # 6
```

顶层传入的 `ObjectRef` 可以建立执行依赖，消费者无需先在 Driver 上获取生产者的数据。Ray 会在依赖就绪后安排执行。在需要逐步获取完成结果时，`ray.wait(refs, num_returns=1)` 返回就绪与未就绪的 **引用列表**，不是直接返回数据对象。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Ray ObjectRef 数据依赖图"><img src="/diagrams/ray-object-dag.svg" alt="Ray 的任务数据依赖和 Object Store" loading="lazy" /></button>
  <figcaption>传递 ObjectRef 描述依赖，真正的对象值可在其他 Worker 或节点上。</figcaption>
</figure>

## 二、Ray 集群架构与分布式任务调度

| 组件 | 主要负责 | 不应该误解成 |
| --- | --- | --- |
| Driver | 创建 Actor、提交任务、运行上层业务逻辑 | 执行所有远程计算的中央进程 |
| Worker | 执行 Task 或 Actor 方法 | 永远具有相同的生命周期 |
| Raylet | 本节点资源、Worker 管理和调度协作 | 只接受中央调度器指令的被动执行器 |
| GCS（Global Control Service） | 节点、Actor、Placement Group 等集群级元数据 | 存放所有对象和 Tensor 的数据存储 |
| Object Store | 缓存和共享部分 Ray 对象 | GCS 元数据服务 |

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Ray GCS Raylet Object Store 架构图"><img src="/diagrams/ray-cluster-architecture.svg" alt="Ray 集群控制面、Raylet、Worker 与对象存储的关系" loading="lazy" /></button>
  <figcaption>控制元数据、任务调度与实际对象传输属于不同层面。</figcaption>
</figure>

### 节点级调度与资源放置

Head 节点提供关键控制服务，但 Ray 的调度涉及节点级资源管理、对象依赖和放置约束，并不是每次 Task 都由一个中央服务逐一挑选最优机器。

实际阅读任务调度时，更值得依次考虑：**资源够不够、ObjectRef 依赖是否就绪、数据在哪个节点、Actor 是否有明确的放置约束**。CPU/GPU 资源规格、Placement Group 和节点可用性都可能影响调度结果。

Ray 对资源的分配和执行地点的决定，不等于它同时负责业务框架内部的 GPU Kernel 或 NCCL 数据传输。

## 三、对象生命周期与故障恢复

Ray 将返回对象的引用和对象值分开管理。结果可能存在 Object Store 中，需要它的 Worker 可能通过本地访问或跨节点传输取得对象。对象的生命周期还会受引用计数、内存使用和系统配置影响。

**失败后的恢复必须区分三类场景。**

- Task 出错：能否重试取决于配置、失败类型以及任务是否包含外部副作用；重试可能导致外部操作重复执行。
- Actor 进程故障：可配置重启与方法重试，但进程内业务状态不会在所有场景下自动恢复；需要显式持久化或重建。
- 对象或节点丢失：某些情况下可尝试重建对象，但并非一切对象都可以恢复，特别是有外部输入或不可重放副作用的计算。

GCS 主要管理集群级控制信息，而非保存全部对象数据或提供对所有 Actor 状态的自动回放。任务重试、对象重建与 Actor 状态恢复具有不同的机制和约束，实际行为还受任务所有权、系统配置与 Ray 版本影响。

## 四、Ray 与 vLLM 的分布式推理架构

在多 GPU 推理场景中，需要区分分布式资源管理与模型执行两类职责。

**Ray 管理 Worker 的分布式创建与资源放置；vLLM 负责模型加载、执行引擎、TP/PP 策略和所需 GPU 通信的初始化。** 不能把 NCCL AllReduce 简单说成由 Ray 的 GCS 来完成。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Ray vLLM 资源管理示意"><img src="/diagrams/ray-vllm-boundary.svg" alt="Ray 负责 GPU Worker 放置，vLLM 负责模型执行和 NCCL 通信" loading="lazy" /></button>
  <figcaption>Ray 解决 Worker 放在哪里，vLLM 解决 GPU Worker 如何协作计算。</figcaption>
</figure>

### 双节点张量并行部署示例

设 Node A 和 Node B 各有一张 GPU，先使其加入同一个 Ray 集群，再让 vLLM 采用 Ray Executor 与两卡 TP：

```bash
# Node A
ray start --head --port=6379

# Node B：用 Head 的真实地址替换占位符
ray start --address=<HEAD_IP>:6379

# 在能够连接 Ray 集群、访问模型的环境中
vllm serve <MODEL_PATH> \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray
```

这是为了说明各组件的**职责和参数关系**，不是可直接复制到任意集群的完整安装指南。分布式部署还要保证版本兼容、模型文件访问、GPU/NIC 设备可见性、网络与通信库配置。

启动流程可以拆成三段：

1. **Ray 集群资源检查**：节点是否加入、两张 GPU 是否能被调度，Placement Group 是否能满足资源预留。
2. **vLLM Worker 初始化**：Worker Actor 是否正确创建，模型权重分片与运行时是否准备好。
3. **GPU 通信与实际推理**：NCCL 等通信库能否完成初始化，TP Worker 是否能够执行真正的计算。

Ray Actor 启动成功，不等于 NCCL 一定可用。vLLM 也存在不使用 Ray 的其他执行后端，选择依赖部署规模和条件。

## 五、Ray Dashboard 与分布式故障排查

Ray Dashboard 提供 Cluster、Jobs、Actors、Metrics、Logs 等观察入口。排查分布式问题时，应先根据故障现象确定所在层级，再选择相应的日志、资源状态或性能指标。

| 故障现象 | 首先观察 | 后续排查 |
| --- | --- | --- |
| Worker 节点未加入 | Cluster / 节点状态 | 网络、Head 地址、Ray 版本与启动日志 |
| 请求 GPU 后始终 Pending | Cluster 资源、Placement Group | 资源总量及放置约束 |
| Actor 创建或运行失败 | Actors + Logs | Actor 异常栈、模型加载与环境 |
| 推理吞吐低，GPU 利用率低 | Metrics + 业务指标 | 负载、等待、并行策略及内核 |
| 跨节点 GPU 通信失败 | Logs、节点网络状态 | NCCL、网络设备和端口 |
| Ray 控制服务异常 | Ray 系统日志 | GCS、Dashboard / Agent 与节点服务 |

### vLLM Worker 启动异常的排查流程

假设 vLLM 使用 Ray Executor 启动后一直没有进入服务就绪状态。

先确认**Ray 是否能发现足够 GPU**。如果资源不够，检查节点、Placement Group、资源占用。如果 GPU 资源已经预留，再检查 Worker Actor 的生命周期；Actor 启动失败应重点查看其日志。如果 Actor 已运行，但停在模型初始化或集合通信阶段，再分析 vLLM 和 NCCL 的日志。

这样可以将资源调度、Actor 生命周期和模型初始化问题分别定位。Dashboard 的具体入口和可用指标会因版本以及监控部署而变化。

## 六、Ray 在 RL 基础设施中的职责边界

在 RL 后训练中，Rollout、Reward 和 Trainer 可以由 Ray Task 或 Actor 运行，但它们之间的样本何时就绪、Microbatch 如何组织、哪个版本的策略允许消费哪些数据，并不会仅靠 Ray 的远程调用自动解决。

类似地，Ray 的 Object Store 与专门的样本数据管理、KV/模型权重传输也不是可以互相替代的工具。

可以用四个问题划定职责：

1. **算力资源在哪里？** Ray 等运行时负责节点/Worker 管理与资源放置。
2. **计算怎么执行？** 训练或推理框架负责模型计算与具体的并行实现。
3. **数据存在哪里？** 对象存储、样本存储和模型参数存储各有所有权。
4. **业务什么时候能消费数据？** 这属于算法、数据就绪与上层调度语义。

如果明确了这几条边界，就更容易判断究竟是调度开销、对象传输、GPU 通信还是训练/推理算法造成的性能问题。

## 七、总结：Ray 的执行模型与系统边界

读懂 Ray 的一条主线是：

**Driver 提交 Task / 创建 Actor → ObjectRef 连接依赖 → Raylet 与 GCS 管理资源和控制信息 → Worker 执行计算 → Object Store 支撑对象访问。**

对应的工程问题也很具体：资源不足还是任务依赖没就绪？Actor 失败还是 GPU 通信失败？对象传输代价是否过大？故障恢复是否会重复外部副作用？

Ray 提供统一的分布式执行与资源管理抽象；模型计算、专用数据存储、GPU 集合通信和算法控制仍由相应的框架与组件承担。

## 参考资料与延伸阅读

[Ray Core 官方入门](https://docs.ray.io/en/latest/ray-core/walkthrough.html) · [Ray Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html) · [Ray Actors](https://docs.ray.io/en/latest/ray-core/actors.html) · [Ray Objects](https://docs.ray.io/en/latest/ray-core/objects.html)

[Ray Scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/index.html) · [Ray Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault-tolerance/index.html) · [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)

[vLLM Distributed Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving/) · [Ray GitHub](https://github.com/ray-project/ray) · [vLLM GitHub](https://github.com/vllm-project/vllm)

本文的 SVG 是教学图解，不代表性能 Profiling 或完整网络拓扑。部署命令、Actor 调度细节和故障恢复能力需结合实际的 Ray / vLLM 版本与环境检查。
