---
title: GPU 显存管理：从 PyTorch Allocator、CUDA VMM 到 TMS 与 Selective Offload
description: 理解 Tensor、Storage、Caching Allocator 与虚拟内存映射，分析 TMS pause/resume 与 Relax Selective CPU Offload 的生命周期和正确性边界。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="gpu-memory-management" tabindex="-1">GPU 显存管理：从 PyTorch Allocator、CUDA VMM 到 TMS 与 Selective Offload</h1>
  <p class="asb-story-subtitle">当 Tensor 的地址保持不变，背后的物理显存还在吗？</p>
  <p class="asb-story-deck">CUDA Graph、训推共置和模型状态卸载都依赖正确区分地址、存储和内容。本文从一个 torch.empty 开始，沿 Caching Allocator、内存 Hook、VMM、Torch Memory Saver 与 Selective CPU Offload 的机制逐层展开。</p>
</div>

GPU 显存管理里有一个容易被忽略的事实：**Tensor 对象、虚拟地址、物理显存与数据内容不是同一个概念。** 只有把它们分开，才能解释为什么 `empty_cache()` 不能释放活跃 Tensor、为什么 CUDA VMM 可以保持地址不变，以及为什么 Pause/Resume 不一定能恢复旧值。

## 一、从 torch.empty 看 Tensor、Storage 和显存

```python
import torch

x = torch.empty((1024, 1024), device="cuda", dtype=torch.float32)
print(x.shape, x.dtype, x.device)
print(x.untyped_storage().nbytes())
```

一个 Tensor 包含 Shape、Stride、Dtype、Device 等元信息，其底层字节通过 Storage 关联的内存访问。普通 CUDA Tensor 在程序中通过 GPU 虚拟地址访问对应资源，而物理显存由设备驱动和分配器等层次管理。

注意 `torch.empty()` 的“empty”不是指不占 GPU 显存，而是**不初始化其中的数值**。它仍然需要为所请求的数据规模提供合适的存储。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：Tensor Storage 虚拟地址和物理显存的对应关系"><img width="960" height="560" src="/diagrams/gpu-tensor-storage.svg" alt="Tensor Storage 虚拟地址和物理显存的对应关系" loading="lazy" /></button>
  <figcaption>Tensor 元数据、共享 Storage、设备虚拟地址与物理分配应分层理解。</figcaption>
</figure>

## 二、PyTorch Caching Allocator：Allocated 与 Reserved

PyTorch CUDA Caching Allocator 会复用已释放给分配器但尚未归还驱动的缓存块，以减少频繁的底层分配成本。因此，删除 Python 变量或释放 Tensor 引用，不一定立刻使 `nvidia-smi` 显示的占用下降。

```python
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())

del x
torch.cuda.empty_cache()
```

`memory_allocated()` 近似反映 PyTorch 当前活跃 Tensor 的分配量，`memory_reserved()` 则包含分配器已经向底层申请、尚由其管理的显存。**`empty_cache()` 不能把仍被活跃 Tensor 引用的 Storage 当成空闲缓存释放。**

它也不等同于“把模型参数卸载到 CPU”，更不能作为训推共置场景中完整的 Sleep/Wake 机制。

## 三、CUDA VMM：地址预留和物理映射可以分离

CUDA 提供的 Virtual Memory Management API 允许应用显式管理虚拟地址区间、物理资源句柄、映射关系和访问权限。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 CUDA VMM 映射生命周期"><img width="960" height="650" src="/diagrams/gpu-vmm-mapping.svg" alt="CUDA VMM 从虚拟地址预留到物理映射与访问权限的关系" /></button>
  <figcaption>虚拟地址预留、物理资源创建、映射和访问授权是不同的步骤。</figcaption>
</figure>

| CUDA Driver API | 职责 | 关键边界 |
| --- | --- | --- |
| `cuMemAddressReserve` | 预留虚拟地址范围 | 此时不代表已分配承载数据的物理显存 |
| `cuMemCreate` | 创建物理资源分配句柄 | 句柄与虚拟地址是独立对象 |
| `cuMemMap` | 将物理资源映射到指定地址范围 | 仍需设置相应访问权限 |
| `cuMemSetAccess` | 配置设备访问权限 | 映射存在不等于所有设备都可访问 |
| `cuMemUnmap` | 解除映射关系 | 不会自动保存原来的 Tensor 数据 |
| `cuMemRelease` | 释放句柄引用 | 能否回收资源与映射及引用生命周期有关 |
| `cuMemAddressFree` | 释放虚拟地址预留 | 在地址范围不再需要时执行 |

这也回答了 CUDA Graph 常见问题：**Graph 记录的 Kernel 参数可能包含设备虚拟地址；地址稳定性有利于某些 Graph Replay 场景，但并不意味着原来的物理内存或数值内容仍然存在。**

如果在取消映射后仍访问原地址，而尚未建立有效的新映射与访问权限，程序会遇到非法访问或其他未定义的执行问题。进行 Unmap / Release 前还必须保证依赖该内存的 GPU 工作已经完成。

## 四、Torch Memory Saver 的 Pause / Resume 保留了什么？

[Torch Memory Saver（TMS）](https://github.com/fzyzcjy/torch_memory_saver) 是第三方项目，并不是 PyTorch 内置的标准接口。它通过受管理的分配路径和底层虚拟内存机制，支持在 Pause 时释放部分物理显存资源、在 Resume 时重新建立内存映射，并尽量维持地址稳定。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 TMS Pause Resume 资源路径"><img width="960" height="620" src="/diagrams/gpu-tms-pause.svg" alt="TMS 虚拟地址保留、暂停解除映射、恢复内容与物理显存的关系" loading="lazy" /></button>
  <figcaption>地址恢复与内容恢复必须分别讨论：Pause 不意味着原有 Tensor 数值已经被自动保存。</figcaption>
</figure>

### 重要区别：CPU Backup 并非默认语义

对于需要在 Resume 后继续使用的模型权重或优化器状态，需要确认是否开启 CPU Backup 或其他可恢复机制；某些可重建的 KV Cache/临时 Buffer 则不需要保留原始数值。

```python
# 概念性示例；API 和 hook 设置以安装的 TMS 版本为准
from torch_memory_saver import torch_memory_saver

with torch_memory_saver.region(enable_cpu_backup=True):
    weights = torch.full((1024,), 42, device="cuda")

torch_memory_saver.pause()
# 此时不要访问被 pause 的 GPU 数据
torch_memory_saver.resume()

assert weights[0].item() == 42
```

如果没有对应备份或其他重建机制，即使虚拟地址可以重新用于访问，也不应假定 Resume 后会看到 Pause 之前的值。

### Hook 不等于 GPU 内存类型

TMS 的部分接入方式使用 `LD_PRELOAD` 优先加载动态库，拦截受支持的分配/释放接口；一些版本也提供对 PyTorch 自定义 Allocator / Memory Pool 的接入。不同 Hook 方式的适用范围和兼容性并不一致。

`LD_PRELOAD` 是动态链接器机制，**不是 CUDA VMM 的替代品**；并不是任何 CUDA 分配请求都一定经过同一套可预加载符号。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：缓存分配器和内存分配 Hook 的接入边界"><img width="960" height="560" src="/diagrams/gpu-allocation-hook.svg" alt="缓存分配器和内存分配 Hook 的接入边界" loading="lazy" /></button>
  <figcaption>Preload 拦截支持的底层符号；自定义分配器在另一层接入，覆盖范围取决于模式。</figcaption>
</figure>

## 五、训推共置为什么需要 Sleep / Wake？

在某些 RL Colocated 部署中，Trainer 与 Rollout 可能复用相同 GPU。训练更新需要模型参数、优化器状态和梯度缓冲，而推理采样希望获得足够空间存放 KV Cache 与推理工作区。反复销毁和重建完整训练进程会有较高成本，因此系统可能在训练与 Rollout 间执行 Sleep/Wake。

这里需要区分：

1. **必须持久保存的状态**：模型参数、FP32 Master Weights、Adam 的动量和二阶矩等。
2. **适当训练边界可丢弃的状态**：已经完成消费的 Gradient Buffer 与可重新计算的部分临时工作区。
3. **必须清除引用才能释放的状态**：某些持有 Autograd Graph 引用的 Cache、Dispatcher Buffer 或延迟释放对象。

并非所有这三类内存都应使用同一套备份策略。

## 六、Selective CPU Offload：由训练语义选择保留状态

Relax 的 [Selective CPU Offload PR #85](https://github.com/redai-studio/Relax/pull/85) 提供了理解该问题的工程案例：不只在底层追踪哪些字节被分配，还尝试根据 Megatron 训练状态的语义，选择需要备份和可以重建的 Buffer。

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

Megatron 的多个 Parameter 可能通过 View 共享一个连续 Param Buffer 的 Storage。如果逐个重绑 `param.data`，可能破坏原有的 Shard/Flat Buffer 假设。更稳定的处理策略是以共享底层存储为单位执行备份与恢复。

从概念上说，卸载步骤为：

```python
# 结构性伪代码：不代表可直接用于通用 Tensor
cpu_backup.copy_(flat_param_buffer)
saved_size = flat_param_buffer.untyped_storage().nbytes()
# 在满足所有权和同步条件后释放/调整设备存储
```

恢复时也必须把值送回正确的存储位置，并确保任何异步 H2D 已完成。仅保存一个 CPU View 的指针，并不能让所有已被清空的设备 Tensor 自动恢复正常。

### Pinned Memory 与同步代价

Pinned Host Memory 可在适用的设备和数据路径上改善 H2D/D2H 传输行为，但它有主机内存占用与锁页成本；`non_blocking=True` 的调用语义也不等于数据已经抵达 GPU。必须通过事件或合适的同步机制确保后续计算不会使用未完成的拷贝。


### CPU View 如何参与权重同步

```python
param._relax_cpu_offload_data = cpu_data[start:end].view(param.shape)
 
# weight_update/common.py
if getattr(x, "_relax_cpu_offload_data", None) is not None         and x.storage().size() == 0:
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

## 七、TMS 与 Selective Offload 各有什么限制？

| 维度 | TMS 类方案 | Selective Training-State Offload |
| --- | --- | --- |
| 控制粒度 | 按受管分配与 Hook/Region 跟踪 | 按训练状态和 Buffer 语义选择 |
| 核心机制 | 分配拦截、VMM、Pause/Resume | CPU Backup、Storage 生命周期、训练状态恢复 |
| 保留地址 | 可以作为设计目标 | 不一定保留底层相同 VA |
| 旧值恢复 | 需对应备份/重建配置 | 显式选择和拷回必须保留的状态 |
| 风险 | Allocator 兼容性、映射状态、Hook 覆盖 | Flat Buffer 假设、优化器完整性、CPU RAM 峰值 |

特别需要注意 `expandable_segments`：它可能使用 CUDA 虚拟内存相关机制缓解动态形状下的碎片问题；某些 TMS 预加载 Hook 与这种分配路径的组合存在兼容限制。**这不是“VMM 与 expandable_segments 在原理上不兼容”**，而是特定实现对分配路径追踪与资源所有权的支持问题。

## 八、源码阅读路线与性能验证

建议依次阅读 PyTorch CUDA Memory 文档、TMS README，再沿 Relax PR #85 的入口追踪：Actor 的 `sleep()` / `wake_up()` → TrainStateOffloader 的 `offload()` / `reload()` → Megatron Param/Grad Buffer → 权重同步期间如何访问 CPU Backup。

需要分开测量 **显存释放量、Sleep 时间、Wake 时间、CPU RAM/Pinned RAM 占用、同步开销以及端到端 RL 吞吐**。只比较 GPU Memory Reserved 不足以证明卸载方案更优；即使单次 Sleep/Wake 更快，完整训练循环也可能仍受其他阶段限制。

## 九、总结：地址、内存和状态是三件不同的事

PyTorch Caching Allocator 主要解决分配效率和缓存复用；CUDA VMM 使虚拟地址与物理资源的映射能够被更细粒度地控制；TMS 探索使用这一能力实现地址稳定的 Pause/Resume；Selective Offload 则在应用层明确哪些训练状态值得备份、哪些可以重建。

它们共同说明：**安全地释放显存，不等于仅仅让某个指针继续存在。** 数据内容、执行依赖、Buffer 所有权和恢复时序都必须被正确管理。

## 参考资料

[NVIDIA CUDA Driver VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html) · [PyTorch CUDA Memory Management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) · [Torch Memory Saver](https://github.com/fzyzcjy/torch_memory_saver) · [Relax Selective CPU Offload PR #85](https://github.com/redai-studio/Relax/pull/85)

代码片段对应 Relax PR #85 的卸载路径；API 与后端策略按所使用的版本选择。
