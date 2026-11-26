---
title: RL 训练状态卸载：从 TMS 到 Selective CPU Offload
description: 围绕 Megatron Flat Buffer、优化器状态、CPU View 与 Dispatcher，分析 TMS 和 Selective CPU Offload 的恢复语义及正确性验证。
sidebar: false
aside: right
outline: [2, 3]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="gpu-memory-management" tabindex="-1">RL 训练状态卸载：从 TMS 到 Selective CPU Offload</h1>
  <p class="asb-story-subtitle">参数要保存，梯度可重建：训练状态如何安全地离开 GPU？</p>
  <p class="asb-story-deck">从一次共置训练的 Sleep/Wake 出发，沿 Megatron 共享存储、优化器状态和权重同步路径，理解为什么显存卸载既需要底层机制，也需要训练语义。</p>
</div>

一次 Rollout 结束后，训练 Actor 需要恢复参数和优化器继续更新；下一次 Rollout 开始前，它又要把显存让给推理引擎。参数、Adam 历史统计、梯度和临时激活有不同的生命周期，不能用一次“把所有 Tensor 搬到 CPU”概括。

本文聚焦**训练 Actor 的状态管理**。Allocator、CUDA VMM、TMS 和 SGLang/vLLM 的完整调用链见[《GPU 显存管理全栈导读》](/posts/gpu-memory-full-stack)。

## 一、先确定卸载需要保留什么

| 问题 | TMS 的 VMM 路径 | Selective CPU Offload |
| --- | --- | --- |
| 如何释放设备资源 | 暂时解除受管分配的物理映射 | 按训练语义搬移状态，释放相应存储 |
| 如何处理地址 | 保留受管虚拟地址，恢复时重新映射 | 不承诺重新分配后获得原设备地址 |
| 如何保留内容 | 根据 region 的 CPU backup 等配置保存 | 显式保存参数和优化器状态，丢弃可重建内容 |
| 谁判断哪些状态必须保存 | 上层选择 region、tag 与备份策略 | 训练框架依据 Buffer 与优化器语义选择 |

TMS 是第三方库，CUDA VMM 是底层接口。两者提供的地址管理能力，不会自动判断一个梯度是否已被消费，也不会替训练循环选择正确的暂停时刻。

## 二、Sleep/Wake 必须位于安全边界

卸载前需要停止新的访问，并完成当前依赖这些存储的计算与通信；恢复后需要确保映射、拷贝与必要的状态重建完成，再允许下一阶段读取。

即使 TMS 维持了 `data_ptr()`，没有备份的数据仍需要重新生成；Selective 路径若改变了设备地址，则必须检查训练 CUDA Graph、通信注册和其他持有旧指针的组件。下面的 Storage 操作以 Relax 的 Megatron 集成为案例，依赖其 Buffer 所有权与阶段边界，不能直接套用到任意 Tensor。

## 三、训推共置为什么需要 Sleep / Wake？

在某些 RL Colocated 部署中，Trainer 与 Rollout 可能复用相同 GPU。训练更新需要模型参数、优化器状态和梯度缓冲，而推理采样希望获得足够空间存放 KV Cache 与推理工作区。反复销毁和重建完整训练进程会有较高成本，因此系统可能在训练与 Rollout 间执行 Sleep/Wake。

这里需要区分：

1. **必须持久保存的状态**：模型参数、FP32 Master Weights、Adam 的动量和二阶矩等。
2. **适当训练边界可丢弃的状态**：已经完成消费的 Gradient Buffer 与可重新计算的部分临时工作区。
3. **必须清除引用才能释放的状态**：某些持有 Autograd Graph 引用的 Cache、Dispatcher Buffer 或延迟释放对象。

并非所有这三类内存都应使用同一套备份策略。

## 四、Selective CPU Offload：由训练语义选择保留状态

Relax 的 [Selective CPU Offload PR #85](https://github.com/redai-studio/Relax/pull/85) 提供了理解该问题的工程案例（[代码快照](https://github.com/redai-studio/Relax/blob/2b11c80d57cb283f6e968f9d892bee420d54dc01/relax/backends/megatron/weight_update/train_offload.py)）：不只在底层追踪哪些字节被分配，还尝试根据 Megatron 训练状态的语义，选择需要备份和可以重建的 Buffer。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Selective CPU Offload 数据路径"><img width="960" height="705" src="/diagrams/gpu-selective-offload.svg" alt="Selective Offload 将必须保留的状态备份到 CPU，并释放可重建状态" loading="lazy" /></button>
  <figcaption>选择性卸载强调“保存什么、丢弃什么、什么时候能重新分配”，而不只是让显存占用下降。</figcaption>
</figure>

| 状态 | Sleep 阶段 | Wake 阶段 |
| --- | --- | --- |
| Param Flat Buffer | 复制到 CPU，释放/缩小对应 GPU Storage | 重建 GPU Storage 并拷回 |
| FP32 Master Parameters | 保存优化器需要的精度状态 | 搬回训练设备 |
| Adam Moments | 保存历史统计 | 搬回设备供下一次更新 |
| 已消费的 Grad Buffer | 可以释放旧值 | 重新分配并清零 |
| 持有旧 Graph 的临时 Cache | 清理引用 | 按需重建 |

该表表示案例中可采用的处理策略，而不是所有 Megatron 训练步骤都能在任意时刻安全卸载梯度。**梯度累积未结束时随意释放 Gradient Buffer 会破坏训练正确性。**

### 为什么要特别处理 Flat Buffer？

Megatron 的多个 Parameter 可能通过 View 共享一个连续 Param Buffer 的 Storage。如果逐个重绑 `param.data`，可能破坏原有的 Shard/Flat Buffer 假设。更稳定的处理策略是以共享底层存储为单位执行备份与恢复，并同时维护每个参数在连续缓冲区中的偏移。

从概念上说，卸载步骤为：

```python
# 结构性伪代码：不代表可直接用于通用 Tensor
cpu_backup.copy_(flat_param_buffer)
saved_size = flat_param_buffer.untyped_storage().nbytes()
# 在满足所有权和同步条件后释放/调整设备存储
```

恢复时也必须把值送回正确的存储位置，并确保任何异步 H2D 已完成。仅保存一个 CPU View 的指针，并不能让所有已被清空的设备 Tensor 自动恢复正常。

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：Flat Buffer 和 CPU 参数视图的对应关系"><img width="960" height="450" src="/diagrams/gpu-flat-buffer-views.svg" alt="多个 Parameter 共享 GPU Flat Buffer，CPU backup 中的对应切片提供卸载期间的参数视图" loading="lazy" /></button>
  <figcaption>备份按 Flat Buffer 复制，参数视图通过相同偏移关联 CPU 切片；恢复时保持参数与共享存储的关系。</figcaption>
</figure>

### Pinned Memory 与同步代价

Pinned Host Memory 可在适用的设备和数据路径上改善 H2D/D2H 传输行为，但它有主机内存占用与锁页成本；`non_blocking=True` 的调用语义也不等于数据已经抵达 GPU。必须通过事件或合适的同步机制确保后续计算不会使用未完成的拷贝。


### CPU View 如何参与权重同步

```python
param._relax_cpu_offload_data = cpu_data[start:end].view(param.shape)
 
# weight_update/common.py
if (getattr(x, "_relax_cpu_offload_data", None) is not None
        and x.storage().size() == 0):
    return x._relax_cpu_offload_data
```

参数对象可能仍被权重同步代码使用，但其 GPU Storage 已缩为零。额外的 CPU View 让同步逻辑能够在 Sleep 期间从备份读取有效参数。View 与 cpu_data 共享内存，下一轮 Offload 会覆盖这块 CPU 缓冲区，因此 Reload 时必须及时删除旧 View，避免跨周期读到过期数据。

### Reload 的参数与梯度恢复

```python
buffer.grad_data.storage().resize_(buffer.grad_data_size)
buffer.grad_data.zero_()
 
buffer.param_data.storage().resize_(buffer.param_data_size)
buffer.param_data.copy_(buffer.param_data.cpu_data,
                       non_blocking=_NON_BLOCKING)
# 删除临时 CPU 视图；如使用异步 H2D，返回前做必要同步
```

恢复参数时要把数据拷回设备。若已完成梯度消费且不处于梯度累积中，梯度缓冲区只需重新分配并清零。持久复用 CPU 备份可避免反复分配，但备份会继续占用主机内存。

### Dispatcher 引用与显存回收

DeepEP/HybridEP Dispatcher 缓存中的 token_probs、dispatched_probs 等 Tensor 可能携带 grad_fn，进而引用 Autograd Graph 和 Activation。若只是释放 Gradient Buffer，这些 Python 引用仍可能阻止显存回收。PR 遍历当前、保存的训练和推理 dispatcher，以及关联通信管理器，清理已知缓存字段。

### 优化器已有 CPU Offload 时，避免重复管理

案例中的 `_skip_optimizer` 检查 Megatron 的 `optimizer_cpu_offload`。若 HybridDeviceOptimizer 已经负责 CPU/GPU 放置，Selective 策略仍处理模型 Flat Buffer，但跳过手动搬移优化器状态，避免破坏优化器自身的设备与调度约束。

普通路径则遍历分布式优化器（包括 ChainedOptimizer 中的子优化器），保存 FP32 master parameters，以及已经创建的 Adam `exp_avg`、`exp_avg_sq`。这些状态可能延迟初始化，因此验证必须覆盖第一次更新前后，而不能只测试模型刚加载时的卸载。

### Actor、策略选择与调用链

```python
# relax/backends/megatron/actor.py
self._train_state_offloader = MegatronTrainStateOffloader(
    self.model, self.optimizer, args
)
 
# sleep()
self._train_state_offloader.offload()
# wake_up()
self._train_state_offloader.reload()
# update_weights()
with self._train_state_offloader.disable_during_update():
    self.weight_updater.update_weights()
```

Facade 根据 --offload-train、--selective-offload 和 TMS 的 LD_PRELOAD 状态选择策略。TMS 策略委托 pause/resume，并在权重更新期间临时 disable Hook；Selective 策略直接控制状态搬移，更新期间使用 nullcontext。

以下伪代码只展示启用训练卸载后的策略选择。

```python
if offload_train and not selective_offload and tms_hook_preloaded:
    strategy = TMS()
else:
    strategy = Selective()
```

实际代码还有 TMS 初始化失败时的 warning/fallback，以及 offload_train=False 时的 no-op。启动器在 --selective-offload 下不再为训练 Actor 配置 TMS 的 LD_PRELOAD。

## 五、TMS 与 Selective Offload 各有什么限制？

| 维度 | TMS 类方案 | Selective Training-State Offload |
| --- | --- | --- |
| 控制粒度 | 按受管分配与 Hook/Region 跟踪 | 按训练状态和 Buffer 语义选择 |
| 核心机制 | 分配拦截、VMM、Pause/Resume | CPU Backup、Storage 生命周期、训练状态恢复 |
| 保留地址 | 可以作为设计目标 | 不一定保留底层相同 VA |
| 旧值恢复 | 需对应备份/重建配置 | 显式选择和拷回必须保留的状态 |
| 风险 | Allocator 兼容性、映射状态、Hook 覆盖 | Flat Buffer 假设、优化器完整性、CPU RAM 峰值 |

特别需要注意 `expandable_segments`：它可能使用 CUDA 虚拟内存相关机制缓解动态形状下的碎片问题；某些 TMS 预加载 Hook 与这种分配路径的组合存在兼容限制。**这不是“VMM 与 expandable_segments 在原理上不兼容”**，而是特定实现对分配路径追踪与资源所有权的支持问题。

## 六、沿调用链验证正确性与性能

源码阅读顺序可以从 Actor 的 `sleep()` / `wake_up()` 开始，进入 TrainStateOffloader 的 `offload()` / `reload()`，再追到 Param/Grad Buffer、优化器状态和权重同步的 CPU Backup 分支。这样先看阶段边界，再看具体字节如何搬移。

### 正确性：单轮恢复不够，还要连续训练

| 验证点 | 对照方式 | 能发现的问题 |
| --- | --- | --- |
| 参数与优化器旧值 | offload 前保存小规模参考，reload 后逐项对比 | 参数恢复但 master weights 或 Adam 状态丢失 |
| Buffer 别名关系 | 检查参数 View 的偏移、形状与共享存储关系 | 数值暂时相同，但后续更新写入错误位置 |
| 梯度生命周期 | 在完成 optimizer step、允许清零的边界执行卸载 | 梯度累积尚未结束就丢弃数据 |
| Sleep 期间的权重读取 | 从 CPU View 读取并与参考参数比较 | 读到零 Storage 或上一轮备份 |
| 多轮恢复 | 固定输入和随机状态，对比不卸载与卸载后的连续更新 | 第二轮起 CPU View 过期、状态漂移或持续泄漏 |
| 异步执行与通信 | 覆盖实际 H2D/D2H、通信和 Graph 配置 | 拷贝未完成就读取，或组件仍持有旧地址 |

比较训练结果时采用适合精度与算子的误差标准；存在随机性或非确定性算子时，不应把位级一致作为所有配置的通用要求。

### 性能：观察完整周期和资源峰值

固定模型、并行度、batch、序列长度、生成数量和权重同步方式，分别记录显存释放量、Sleep/Wake 时间、CPU RAM/Pinned RAM 占用、权重更新峰值以及端到端 step 时间。

对于备份总量为 B 的状态，仅传输时间就受 D2H 与 H2D 有效带宽限制；此外还有同步、分配与缓存清理成本。Selective 的潜在收益来自少搬可重建数据，是否胜出取决于实际状态组成与主机设备链路，不能从“释放更多显存”直接推断。

Pinned backup 的持久复用可减少反复分配和锁页成本，也会长期占用主机内存。还应检查多轮循环的 RSS 是否稳定，以及权重更新和下一阶段加载是否造成瞬时峰值。单次 API 的返回时间、`memory_reserved()` 或 `nvidia-smi` 截图，都不足以单独证明方案更快。

## 七、把机制与训练语义接起来

PyTorch Caching Allocator 主要解决分配效率和缓存复用；CUDA VMM 使虚拟地址与物理资源的映射能够被更细粒度地控制；TMS 探索使用这一能力实现地址稳定的 Pause/Resume；Selective Offload 则在应用层明确哪些训练状态值得备份、哪些可以重建。

安全卸载需要同时管理数据内容、执行依赖、Buffer 所有权和恢复时序。需要回看底层映射与引擎接口时，可沿[全栈导读](/posts/gpu-memory-full-stack)中的分层调用链继续阅读。

## 参考资料

[NVIDIA CUDA Driver VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html) · [PyTorch CUDA Memory Management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) · [Torch Memory Saver](https://github.com/fzyzcjy/torch_memory_saver) · [Relax Selective CPU Offload PR #85](https://github.com/redai-studio/Relax/pull/85)

代码片段对应 Relax PR #85 的卸载路径；API 与后端策略按所使用的版本选择。
