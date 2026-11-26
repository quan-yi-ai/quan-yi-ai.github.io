---
title: GPU 显存管理全栈导读：从 PyTorch 到推理框架与 RL 训练
description: 从 PyTorch Caching Allocator、CUDA VMM、torch_memory_saver，到 SGLang、vLLM 与 slime、verl 的显存轮换调用链。
sidebar: false
aside: right
outline: [2, 3]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
</div>

# GPU 显存管理全栈导读：从 PyTorch 到推理框架与 RL 训练

在训练与推理共用 GPU 的 RL 系统里，Rollout 希望把显存留给 KV cache，训练又需要激活、梯度和优化器状态。如何让一个阶段暂时交出显存，同时避免下一阶段重建全部资源？

理解这件事，需要沿着分配器、虚拟内存、推理引擎和训练调度逐层往下看。本文围绕显存分配与 pause/resume，串起四个层次：

1. **PyTorch 显存管理（起点）**：Caching Allocator / MemPool / 可插拔分配器。解释缓存复用，以及释放后重新分配为何不能保证拿回原地址。
2. **torch_memory_saver**：借助 CUDA VMM 分离虚拟地址与物理显存，实现显存 pause/resume，作为底层能力被上层集成。
3. **推理框架（SGLang / vLLM）**：封装为 release/resume_memory_occupation、sleep/wake_up，是能力落地的场景。
4. **RL 训练框架（slime / verl）**：在 colocate 下编排训练与推理共享显存。

完整调用关系见下一节图示。训练侧的参数、优化器和梯度如何分别保存与恢复，见配套文章[《RL 训练状态卸载：从 TMS 到 Selective CPU Offload》](/posts/gpu-memory-vmm-tms-offload)。

涉及仓库：`pytorch`、`torch_memory_saver`、`sglang`、`vllm`、`slime`、`verl`。
下文代码路径均为仓库内相对路径（以仓库名开头），克隆对应仓库后即可对照定位。

几个高频术语：
- **KV cache**：自回归推理时缓存每个 token 的 Key/Value，避免重复计算，是推理期显存的大头。
- **CUDA Graph**：把一串 GPU kernel 录制成图、整体重放以降低启动开销；录制时会固化其中用到的显存地址。
- **colocate**：训练与推理部署在同一批 GPU 上、分时复用显存的方式。
- **rollout**：RL 中用当前策略模型批量生成样本的推理过程。

> 说明：文中代码为便于阅读的**简化摘录**（非逐行原文），完整实现以对应源文件为准；
> 下文聚焦 CUDA 路径，源码阅读基于 2026-11 的实现；TMS、vLLM 和 verl 的核对快照见文末。函数名、配置和状态边界比行号更适合用于定位。

---

## 零、核心结论

三种「释放显存」处在完全不同的层次，务必区分：

- `del tensor`：删除一个 Python 引用；只有最后的存储引用释放、且相关异步使用完成后，block 才能回到可复用状态。缓存池通常仍保留这块显存。
- `torch.cuda.empty_cache()`：把内存池中可释放的**空闲缓存段**归还驱动；不影响任何仍存活的 Tensor。
- `torch_memory_saver.pause()`：对纳入 TMS 管理的、**仍存活**的 Tensor 解除物理显存映射、但**保留虚拟地址**；`resume()` 后 `data_ptr()` 不变。

关键机制：CUDA VMM（虚拟内存管理）将「虚拟地址」与「物理显存」解耦。
地址保持不变，是复用已捕获 CUDA Graph 的关键条件；恢复映射与访问权限、准备好有效数据，并满足形状、布局和执行依赖约束之后，才能安全 replay。地址稳定本身并不保存旧值。

---

## 一、整体调用链

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：整体调用链：从 RL 框架到 CUDA VMM"><img width="900" height="650" src="/images/gpu-memory-full-stack/overall-stack.svg" alt="整体调用链：从 RL 框架到 CUDA VMM" loading="lazy" /></button>
</figure>

---

## 二、PyTorch 显存底层
> 这一层是整条链路的基础。核心是 **Caching Allocator**：PyTorch 在 `cudaMalloc/cudaFree` 之上实现的一层显存缓存池。

### 2.0 从一个 Tensor 开始

```python
import torch

x = torch.empty((1024, 1024), device="cuda", dtype=torch.float32)
y = x.view(-1)
```

`x` 与 `y` 是不同的 Tensor 对象，却共享 Storage。`torch.empty` 需要分配存储，只是不初始化数值；`del x` 后，`y` 仍可能让底层存储继续存活。

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：Tensor Storage 虚拟地址和物理显存的对应关系"><img width="960" height="560" src="/diagrams/gpu-tensor-storage.svg" alt="Tensor 元数据、共享 Storage、虚拟地址和物理显存的关系" loading="lazy" /></button>
  <figcaption>对象、存储、地址、物理资源与数值内容，应分别讨论。</figcaption>
</figure>

| 层次 | 描述什么 | 不能由它单独推出什么 |
| --- | --- | --- |
| Tensor | Shape、Stride、Dtype 与访问视图 | 一个 Tensor 对应一块独占物理显存 |
| Storage | 多个 Tensor 可以共享的底层字节存储 | 删除其中一个 View 就会释放存储 |
| 虚拟地址 | Kernel 用来访问设备内存的地址 | 该地址始终映射着物理显存 |
| 物理显存 | 实际承载字节的设备资源 | 重新映射后仍含有原来的数值 |
| 数据内容 | 权重、缓存或中间结果 | 保留地址就自动保留内容 |

### 为什么需要 Caching Allocator
频繁调用传统 `cudaMalloc` / `cudaFree` 可能引入昂贵的分配、释放与同步成本，打断 GPU 执行流水线。这里讨论 PyTorch 的 native caching allocator；`cudaMallocAsync` 等后端的行为有所不同。
深度学习中显存的申请与释放十分频繁（每次前向都会产生大量临时 tensor），直接调用驱动的开销不可接受。
因此 PyTorch 自行维护一个内存池：向驱动**大块申请**、内部**小块复用**，tensor 释放时不归还驱动，
而是退回内存池等待下次复用。由此引出贯穿全文的一点——**显存的「分配」与「占用驱动」是两件事**。

### 2.1 三层对象：Tensor → Storage → Block/Segment
文件：`pytorch/c10/cuda/CUDACachingAllocator.cpp`（`:100` 顶部的设计注释最具参考价值）

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：Tensor / Storage / Block 与 Segment 结构"><img width="760" height="300" src="/images/gpu-memory-full-stack/block-segment.svg" alt="Tensor / Storage / Block 与 Segment 结构" loading="lazy" /></button>
</figure>

- `struct Block`：一块显存的记账单元，含 `ptr / size / stream / allocated / prev / next / event_count`。
- 一次 `cudaMalloc` 得到一个 **segment**，可被切成多个相邻 Block（split）；释放时相邻 free block 合并（merge）。
- 大小不同走不同 pool（small ≤ 1MB / large），减少碎片。

### 2.2 分配路径：优先复用缓存，不足时才向驱动申请
```text
allocate → DeviceCachingAllocator::malloc
   → 在 pool 中以 best-fit 查找足够大的 free block
   → 命中则（必要时）split 后返回；未命中 →
   → alloc_block → cudaMallocMaybeCapturing → cudaMalloc  (真正向驱动申请)
   → 仍失败 → 释放缓存 block + 同步 event + 重试 → 再失败则 OOM
```
关键点：**绝大多数「分配」并不涉及驱动调用**，只是从内存池中取用一个 block。这解释了为什么
`memory_reserved`（内存池持有的总量）可能明显大于 `memory_allocated`（Tensor 实占）；两者差距取决于负载和缓存状态，并非总会很大。

### 2.3 释放与 empty_cache：两种「释放」
文件：`pytorch/torch/cuda/memory.py` · `pytorch/torch/csrc/cuda/Module.cpp`

```python
def empty_cache():                 # memory.py
    if is_initialized():
        torch._C._cuda_emptyCache()   # → THCPModule_emptyCache (Module.cpp:569)
                                      # → CUDACachingAllocator::emptyCache
```

- 最后的存储引用被释放后，分配器才开始回收对应 block；如仍有其他 stream 使用它，还需要等待相关事件。单独 `del tensor` 不保证触发这一过程。
- `empty_cache()`：把内存池中**完全空闲**的 segment 真正 `cudaFree` 归还驱动，`reserved` 才下降。
  调用链：`emptyCache → release_cached_blocks → release_blocks → cudaFree`。
- 它**不会**回收仍被 Tensor 引用的 block，其他 stream 仍在使用的 block 也必须先满足同步条件。

四个常用指标（`memory_stats` / `THCPModule_memoryStats`，Module.cpp:578）：
```text
allocated = 当前被 Tensor 占用        reserved = 内存池向驱动申请的总量
active    = 使用中及待释放的 block     inactive_split = split 出来的空闲碎片
```

### 2.4 stream 语义：地址复用为何需要同步
Block 记录创建它的 `stream`，默认只能在**同一 stream** 上复用。若要在其他 stream 上使用某 tensor，
可以调用 `tensor.record_stream(s)` 登记，或由调用者显式管理安全的 stream 同步与存储生命周期。登记后，分配器借助 CUDA event（`event_count`）确认该 stream
使用完毕后才会复用这块显存。否则可能出现「显存已被复用、旧 kernel 仍在读取」的数据竞争。`record_stream` 负责避免过早复用，不负责建立生产者与消费者的数据依赖；跨 stream 读取前还需适当的 `wait_stream` / event 同步。
这与 pause 前必须先 synchronize 是同一个道理：**异步执行下，回收显存前必须确认不再有人使用**。

### 2.5 释放后重新分配，为何不能依赖原地址

普通存活 Tensor 的地址不会因为缓存池 split/merge 就被随意移动。问题出在**销毁存储再重新分配**：原来的 block 可能被其他请求占用，segment 也可能被归还驱动，因此新分配不能保证得到原 `ptr`。

PyTorch 自身支持 CUDA Graph：捕获期间会使用 graph-private memory pool 管理相关分配，维持 replay 所需的地址生命周期。TMS 解决的是进一步的需求——在保留受管虚拟地址的同时，暂时释放背后的物理显存，而非替代 PyTorch 的 Graph 支持。

### 2.6 可插拔分配器：官方的三个接入组件（TMS / vLLM 的接入点）
```text
CUDAPluggableAllocator  以外部 malloc/free 函数指针替换分配实现
MemPool                 一个独立的 block 池，可绑定自定义分配器
use_mem_pool(pool)      上下文作用域内的分配都走这个 pool
```
`use_mem_pool` 让第三方能接管当前线程作用域内经过相应 PyTorch 分配路径的请求，而非全局替换所有 CUDA 分配。
torch_memory_saver 的 `region()`、vLLM 的 `use_memory_pool()` 都基于这组机制。

---

## 三、torch_memory_saver（pause/resume 本体）

> torch_memory_saver 是整条链路的核心，pause/resume 的具体实现都在这里。

### 3.0 为什么采用 CUDA VMM，而非 del + 重新 alloc

一种朴素做法是「pause 时 `del tensor`、resume 时重新 `torch.empty`」，但它存在一个致命问题：
**重新分配不能保证拿回原虚拟地址**。而启用 **CUDA Graph** 的推理引擎在 decode 阶段会复用捕获的执行图，
graph 捕获时会把 kernel 参数中的**显存地址固化**到 graph node 中，replay 时不再重新读取指针。
一旦地址变化，graph replay 便会访问到错误或非法地址。

CUDA VMM 的价值就在这里：它把一次分配拆成两种相互独立的资源——

- **虚拟地址**（VA range）：由 `cuMemAddressReserve` 保留，是进程地址空间里的一段编号。
- **物理显存**（physical handle）：由 `cuMemCreate` 创建，是 GPU 上真正的页。
- **映射**：`cuMemMap` 把前者绑到后者，`cuMemSetAccess` 赋读写权限。

| VMM 操作 | 资源变化 |
| --- | --- |
| `cuMemAddressReserve` | 保留一段虚拟地址，此时不代表已有物理 backing |
| `cuMemCreate` | 按设备与粒度要求创建物理资源句柄 |
| `cuMemMap` + `cuMemSetAccess` | 建立映射，并配置设备访问权限 |
| `cuMemUnmap` + `cuMemRelease` | 解除映射并释放句柄引用；实际回收还取决于其他映射与引用 |
| `cuMemAddressFree` | 在地址不再需要时释放预留范围 |

于是 pause 可以只回收**物理显存**、保留**虚拟地址**；resume 时物理页是全新的，
但重新映射回**同一段虚拟地址**。对上层（PyTorch allocator、tensor、CUDA Graph）来说，
`data_ptr()` 可以保持不变。但 pause 期间不得访问未映射地址，resume 后还需恢复或重建内容，才能继续计算。

### 3.1 状态机与数据结构

每块 TMS 管理的分配在 `core.h` 的 `AllocationMetadata` 里记账（相当于 PyTorch 的 `Block`）：

```cpp
struct AllocationMetadata {
    size_t raw_size;          // 用户请求的字节数
    CUdevice device;
    std::string tag;          // weights / kv_cache ... 供分组 pause/resume
    AllocationState state;    // ACTIVE 或 PAUSED
    bool enable_cpu_backup;   CpuBackupSlot cpu_backup;   // 可选 CPU 备份
    bool enable_disk_backup;  DiskBackupSlot disk;
    size_t allocation_size;   // 对齐到 VMM 粒度后的真实大小
    CUmemGenericAllocationHandle allocHandle;  // 当前物理句柄（resume 后会更新）
};
```

状态只有两个，转换由 pause/resume 驱动，非法转换直接 `exit(1)`（fail-fast，见 pause/resume 里的 state 检查）：

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：TMS 状态机：ACTIVE 与 PAUSED 之间切换"><img width="760" height="260" src="/images/gpu-memory-full-stack/tms-state-machine.svg" alt="TMS 状态机：ACTIVE 与 PAUSED 之间切换" loading="lazy" /></button>
</figure>

pause / resume 的时序如下（ptr 全程不变，物理句柄被整体替换）：

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：pause / resume 时序"><img width="840" height="630" src="/images/gpu-memory-full-stack/pause-resume-sequence.svg" alt="pause / resume 时序" loading="lazy" /></button>
</figure>

### 3.2 malloc：一次分配做了三件事
文件：`torch_memory_saver/csrc/core.cpp` · `TorchMemorySaver::malloc`

```cpp
// 1) 建物理显存句柄（按 VMM 粒度对齐后的 allocation_size）
cu_mem_create(&allocHandle, allocation_size, device);
// 2) 预留一段虚拟地址（返回给上层当 data_ptr 用）
cuMemAddressReserve((CUdeviceptr*)ptr, reserve_size, 0, 0, 0);
// 3) 绑定 + 赋权限
cuMemMap((CUdeviceptr)*ptr, allocation_size, 0, allocHandle, 0);
cu_mem_set_access(*ptr, allocation_size, device);
// 4) 记账：ptr → AllocationMetadata{..., ACTIVE, allocHandle}
```

注意 `set_memory_margin_bytes` 这个 OOM 护栏：若设置了 margin，malloc 前会先 `cudaMemGetInfo`，
当本次分配大小与 margin 之和超过空闲显存时主动返回 OOM。它限制初次分配挤占预留空间，但不能保证以后 resume 时一定有足够显存；其他阶段和进程仍可能消耗资源。

### 3.3 pause：释放物理，保留虚拟地址
`TorchMemorySaver::pause(tag)` 遍历 metadata，对 tag 匹配且处于 ACTIVE 的分配：

```cpp
if (metadata.enable_cpu_backup)  cpu_backuper::offload(ptr, raw_size, cpu_backup); // D2H 拷贝
else if (enable_disk_backup)     disk_backend_.offload(...);
cuMemUnmap((CUdeviceptr)ptr, allocation_size);  // 解除映射
cuMemRelease(allocHandle);                        // 归还物理显存 → 设备显存占用可下降
metadata.state = PAUSED;
```

关键点：
- **只 Unmap + Release 物理句柄，没动虚拟地址**。那段 VA 仍被本进程保留着，别的分配不会占用它——这正是 resume 能映射回原地址的前提。
- 物理显存**真的还给了驱动**，`nvidia-smi` 能看到下降（区别在于：`empty_cache()` 也能向驱动归还显存，但只能释放已经空闲且可回收的缓存；TMS 可暂时解除受管存活存储的物理映射）。
- 默认 `enable_cpu_backup=False` 时，pause **直接丢弃数据**。对 KV cache 是合理的（数据本就需要重算）；对权重则需开启 CPU backup，或在 resume 后通过 `update_weights` 重新写入。

### 3.4 resume：物理是全新的，地址是旧的
`TorchMemorySaver::resume(tag)` 对 PAUSED 的分配：

```cpp
CUmemGenericAllocationHandle newAllocHandle;
cu_mem_create(&newAllocHandle, allocation_size, device); // 全新物理句柄
cuMemMap((CUdeviceptr)ptr, allocation_size, 0, newAllocHandle, 0); // 映射回【原】ptr
cu_mem_set_access(ptr, allocation_size, device);
if (enable_cpu_backup) {
    cpu_backuper::onload(ptr, raw_size, device, cpu_backup);   // H2D 拷回
    if (!retain_cpu_backup) cpu_backuper::release(cpu_backup); // 默认释放 CPU 备份
}
metadata.state = ACTIVE;
metadata.allocHandle = newAllocHandle;  // 句柄换了，但 ptr 没换
```

这里有一个容易忽略的细节：**`allocHandle` 变了，`ptr` 没变**。物理页是重新申请的，
不要求是 pause 前的同一组物理页；映射回原虚拟地址后，Tensor 的地址可以保持不变，但数据与通信注册等资源仍需单独处理。
在没有 backup 的情况下，resume 后得到的是**未定义内容**（MUSA 平台会额外清零一次，CUDA 不会）。

### 3.5 pause 前为何必须先 synchronize
文件：`torch_memory_saver/torch_memory_saver/entrypoint.py` · `pause()`

Python 层在调 C++ `tms_pause` **之前**先 `synchronize` 受影响设备。原因：
`cuMemUnmap` 会立刻让那段地址失效，如果此时还有 kernel 在异步队列里排队、即将访问这块显存，
unmap 后 kernel 一跑就是**非法内存访问**。所以必须先把设备上 outstanding 的工作排空。
这条「先同步再 unmap」是 pause 正确性的硬约束，不是性能优化。

恢复时则有相反方向的依赖：先建立映射与访问权限，再恢复备份或写入新权重，最后才允许计算读取。对于无备份的 KV cache，应由引擎同时清除旧缓存索引和有效性元数据，避免把新物理页误当成已有 token 的缓存。

### 3.6 CPU backup：pinned vs mmap
文件：`torch_memory_saver/csrc/cpu_backup.h`

- `offload` = `cudaMemcpy(DeviceToHost)`，`onload` = `cudaMemcpy(HostToDevice)`。
- 两种后端：`PINNED`（`cudaMallocHost`，DMA 快但占不可换出的锁页内存）与
  `MMAP`（匿名 mmap 的 pageable 内存，驻留与换页受 OS 和 swap 配置影响；内容仍须保留，H2D 后需要相应同步）。
- `retain_cpu_backup=True` 时 resume 不释放 CPU 备份，省去下次 pause 的再分配——
  典型用于权重这种每轮都要 offload/onload 的固定缓冲。

启用 CPU backup 的最小用法如下；运行环境需先完成所选 Hook 模式的配置：

```python
from torch_memory_saver import torch_memory_saver

with torch_memory_saver.region(enable_cpu_backup=True):
    weights = torch.full((1024,), 42, device="cuda")

torch_memory_saver.pause()
# pause 期间不读取 weights，也不 replay 访问它的 Graph
torch_memory_saver.resume()
assert weights[0].item() == 42
```

### 3.7 free 与 pause 的区别
```text
pause: cuMemUnmap + cuMemRelease               ；保留 VA；metadata 保留（PAUSED）
free : cuMemUnmap + cuMemRelease + cuMemAddressFree；归还 VA；metadata 删除
```
free 还会 `cudaDeviceSynchronize` 并释放 CPU/disk backup。即 free 是「彻底销毁」，pause 是「挂起」。

### 3.8 两个常见陷阱
- **expandable_segments 冲突**：若设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
  `entrypoint.py` 的 `_sanity_checks()` 会直接报错并禁用 TMS。这是所核对实现的接入和资源所有权限制；expandable segments 也使用 VMM，并不代表两种 VMM 用法在原理上必然不兼容。
- **NCCL CUDA Graph 注册失效**：TMS 在 pause/resume 时更换了物理 backing，若该 buffer 曾被 NCCL
  以 graph user-buffer 形式注册，相关注册可能需要重新处理，具体取决于 NCCL 版本与注册路径。排查时可检查 `NCCL_GRAPH_REGISTER=0` 是否适用，或避免把相关通信 buffer 纳入 TMS region；它不是所有通信内存问题的通用开关。

### 3.9 torch 模式如何挂接到 PyTorch
文件：`torch_memory_saver/torch_memory_saver/hooks/mode_torch.py`

```python
self.allocator = CUDAPluggableAllocator(
    self.get_path_binary(), "tms_torch_malloc", "tms_torch_free")
```
把 PyTorch 的可插拔 allocator 绑定到 TMS 的 C++ `tms_torch_malloc/free`。
再配合 `entrypoint.py` 里 `region()` 用 `use_mem_pool` 包住分配，
**只有 region 作用域内的 tensor** 才走进 TMS 的 VMM 分配、从而变得可 pause；
region 外的 tensor 仍走 PyTorch 默认分配器。这也解释了为何需要 MemPool 这一层：
本场景选择作用域接管，以免把不希望暂停的分配纳入管理；全局替换是另一种覆盖范围不同的设计。


### 3.10 Preload 与 torch 模式：接入范围不同

TMS 是第三方库，CUDA VMM 才是 NVIDIA 提供的底层接口。`LD_PRELOAD` 是动态链接器的符号拦截机制，不是一种 GPU 内存类型。

| 接入方式 | 进入受管路径的方式 | 阅读源码时要确认 |
| --- | --- | --- |
| Preload Hook | 预加载动态库，拦截受支持的分配/释放符号 | 实际调用是否经过这些符号、区域标记及后端兼容性 |
| torch 模式 | `CUDAPluggableAllocator` + `MemPool` | 分配是否发生在相应线程和 pool 的作用域内 |

两种方式最终都要正确跟踪分配的所有权。看到 `region()` 不能直接推断进程内所有显存都受它控制，也不能把某一种模式的兼容限制推广到所有接入方式。

---

## 四、推理框架如何使用它（SGLang / vLLM）

> 这一层是「显存 pause/resume」真正落地的环节：推理引擎将其封装为对外的
> `release/resume_memory_occupation`（SGLang）或 `sleep/wake_up`（vLLM）。
> 两条路线机制同源（都是 CUDA VMM + MemPool），上层封装、备份策略和兼容性处理分别实现，不能把其中一个库的行为直接当作另一个库的契约。

### 4.0 两条路线概览

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：SGLang 与 vLLM 两条路线汇流到 CUDA VMM"><img width="820" height="300" src="/images/gpu-memory-full-stack/engine-two-routes.svg" alt="SGLang 与 vLLM 两条路线汇流到 CUDA VMM" loading="lazy" /></button>
</figure>

### 4.1 SGLang：adapter 一对一转调 TMS
文件：`sglang/python/sglang/srt/utils/torch_memory_saver_adapter.py`

```python
# 工厂：enable_memory_saver=True → Real，否则 Noop（所有方法空转）
adapter = TorchMemorySaverAdapter.create(enable=server_args.enable_memory_saver)

class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    def region(self, tag, enable_cpu_backup=False):
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)
    def pause(self, tag):   return _memory_saver.pause(tag=tag)
    def resume(self, tag):  return _memory_saver.resume(tag=tag)
```

要理解的三件事：
- **怎么让 KV cache 变得可 pause**：引擎在分配 KV cache / 权重时用 `adapter.region(tag=...)`
  包住（tag 区分 weights / kv_cache），于是这些显存走进 TMS 的 VMM 分配。
- **Noop 的意义**：未安装或未启用 TMS 时使用空实现，业务代码无需编写 `if enable:` 分支。
- **对外入口**：HTTP `release_memory_occupation` / `resume_memory_occupation`（见 `entrypoints/http_server.py`）
  最终落到 `adapter.pause/resume(tag)` → `torch_memory_saver.pause/resume`。这正是 RL 框架调用的接口。

### 4.2 vLLM：CuMemAllocator（自带的同源实现）
文件：`vllm/vllm/device_allocator/cumem.py`

vLLM 未使用 torch_memory_saver，而是自行实现了一套等价机制，原理完全相同：

```python
with allocator.use_memory_pool(tag="weights"):   # 相当于 TMS 的 region
    load_weights(...)                              # 这些分配被记入 pointer_to_data[ptr]

def sleep(self, offload_tags=..., discardable=()):   # 相当于对所有 region 执行 pause
    for ptr, data in self.pointer_to_data.items():
        offload = data.tag in offload_tags          # 该 tag 是否需要备份
        self._put_to_sleep(ptr, data, offload)       # offload 则 D2H 拷贝，否则直接丢弃
        # _put_to_sleep 内部最终调用 _unmap_and_release(handle) → cuMemUnmap + release

def wake_up(self, tags=None):                        # 相当于 resume
    for ptr, data in self.pointer_to_data.items():
        if data.is_asleep and (tags is None or data.tag in tags ...):
            self._create_and_map(handle)             # 将物理页重新映射回原 ptr
            if data.cpu_backup_tensor: cudaMemcpy(ptr, cpu_ptr, ...)  # H2D 拷回
```

以下几处与 TMS 一一对应，也印证了这是一种通用机制、而非某个库的特例：
- `_python_free_callback` 在 `cuMemUnmap` 前调用 `torch.accelerator.synchronize(...)`——
  与 pause 前必须先同步是**同一个硬约束**，其注释写明否则会 `CUDA_ERROR_ILLEGAL_ADDRESS`。
- `use_memory_pool` 会临时关闭 `expandable_segments`（pytorch#147851）——
  与前述 expandable_segments 冲突完全同源。
- `sleep` 分为 **offload（CPU 备份）** 与 **discard（直接丢弃）**，对应 TMS 的 `enable_cpu_backup` 开关。

### 4.3 TMS vs vLLM cumem 对照

- `region(tag)` ←→ `use_memory_pool(tag)`：划定可释放区域并打 tag
- `pause(tag)` ←→ `sleep(offload_tags)`：释放物理，保留虚拟地址
- `resume(tag)` ←→ `wake_up(tags)`：重建物理，映射回原地址
- `enable_cpu_backup` ←→ offload vs discard：数据是否备份到 CPU
- `AllocationMetadata`（C++ map）←→ `pointer_to_data`（Python dict）：记账结构
- 底层都是 `cuMemCreate / cuMemMap / cuMemUnmap / cuMemRelease`

概括地说，两者实现同构、底层机制一致：`cumem.py` 与 `core.cpp` 几乎可以逐函数对应。

---

## 五、RL 训练框架的编排（slime / verl）

> 这是整条链路的「需求源头」。这一场景把按 tag 管理、内容备份和 CUDA Graph 兼容的需求串在一起。

### 5.0 为什么 RL 需要 pause/resume：colocate 共享 GPU
在 RLHF / colocate 架构中，**训练**（actor 的 fwd/bwd/optim）与**推理**（rollout 生成样本）
运行在**同一批 GPU** 上。两者显存需求都很大，在显存紧张的共置配置中，需要按阶段让出部分资源：
- rollout 阶段：推理引擎使用权重、KV cache 与工作区；训练侧是否仍驻留参数和优化器状态，取决于卸载策略。
- train 阶段：需要将显存让给训练的激活、梯度与优化器状态。

若采用「关闭推理引擎再重启」，开销过大（需重建 CUDA Graph、重新加载权重），
因此改用 pause/resume：rollout 结束 → 暂停推理显存 → 训练 → 恢复 → 更新权重 → 再次 rollout。
保留地址有助于复用已有 CUDA Graph，减少重建成本。TMS 是一种实现；vLLM 的 CuMemAllocator 则提供了另一条同类路线。

### 5.1 colocate 一轮的时序

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：colocate 一轮的显存轮换时序"><img width="900" height="550" src="/images/gpu-memory-full-stack/colocate-loop.svg" alt="colocate 一轮的显存轮换时序" loading="lazy" /></button>
</figure>

### 5.2 slime
```python
# slime/slime/ray/rollout.py —— rollout 管理器级
def offload(self):            # 训练前：暂停健康检查 + 让每个 server 释放显存
    self.health_monitoring_pause()
    for srv in self.servers.values(): srv.offload()
def onload(self, tags=None):  # 训练后：恢复

# slime/slime/ray/actor_group.py —— actor 级（Ray remote）
def onload(self):  return ray.get([a.wake_up.remote() for a in self._actor_handlers])
def offload(self): return ray.get([a.sleep.remote()  for a in self._actor_handlers])

# slime/slime/backends/sglang_utils/sglang_engine.py —— 落到 SGLang HTTP 接口
def release_memory_occupation(self):            # 先 flush_cache 再请求释放
    self.flush_cache(); return self._make_request("release_memory_occupation")
def resume_memory_occupation(self, tags=None):  # tags: weights / kv_cache
    return self._make_request("resume_memory_occupation", {"tags": tags})
```

### 5.3 verl（用 sleep_level 决定 tag）
```python
# verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py
async def release(self):
    tags = ["kv_cache"] if self.sleep_level == 1 else ["kv_cache", "weights"]
    await self._engine.release_memory_occupation(tags=tags)

# verl/verl/trainer/ppo/v1/trainer_colocate_async.py —— 生命周期钩子即编排点
def on_sample_end(self):  self.checkpoint_manager.abort_replicas(); ... sleep_replicas()
def on_step_end(self):    self.checkpoint_manager.update_weights(...); resume_generation_replicas()
```

`engine_workers.py` 中有一处**必须知道的跨引擎差异**（直接影响 resume 逻辑）：
- **SGLang** 的 `sleep_level=1`：在本文 verl 的 LoRA adapter 保留 base weights 路径中，并未释放 weights（仅释放 KV cache），resume 时**不应**再 resume weights。
- **vLLM** 的 level-1 sleep：经由 `CuMemAllocator.sleep(offload_tags=("weights",))`，weights 已被 unmap，**必须** resume。

### 5.4 tag 相似，sleep level 不应跨引擎套用

`weights` 与 `kv_cache` 的生命周期不同：权重需要恢复旧值或接收训练后的新值；KV cache 在本轮生成结束并失效后通常可以重建。tag 让引擎按语义选择释放与恢复的对象，level 则是各引擎或适配器自己的策略约定。

| 本文讨论的路径 | 权重如何处理 | KV cache 如何处理 | 恢复时的重点 |
| --- | --- | --- | --- |
| verl 的 SGLang 适配路径，`sleep_level=1` | 不通过此 release 调用释放 | release `kv_cache` | 不应对未暂停的 weights 重复执行状态转换 |
| verl 的 SGLang 适配路径，`sleep_level=2` | release `weights` | release `kv_cache` | 先恢复权重存储、更新权重，再恢复 KV cache |
| vLLM 原生 `sleep(level=1)` | 备份到 CPU 并释放受管 GPU 存储 | 丢弃 | wake 恢复权重内容，KV 内容重建 |
| vLLM 原生 `sleep(level=2)` | 常规参数丢弃，需重新加载/更新 | 丢弃 | 先唤醒 weights、写入权重，再唤醒 KV cache |

vLLM level 2 还会处理部分非参数模型 buffer，并提供特定参数保留选项，详见[官方 Sleep Mode 文档](https://docs.vllm.ai/en/latest/features/sleep_mode/)。因此，“level 1 只释放 KV cache”只能用于上表对应的适配路径，不能作为全局定义。

### 5.5 训练侧：释放哪些状态，是另一个决策层

推理引擎交出显存，并不等于训练状态也自动消失。Trainer 仍可能持有参数、FP32 master weights、Adam moments 和梯度缓冲。

- 必须跨阶段保存的状态，需备份或由其他持久副本恢复。
- 已完成消费的梯度和可重算工作区，可在安全边界丢弃并重建。
- 若存在共享 Flat Buffer 或携带 Autograd Graph 的缓存引用，必须按存储所有权与依赖关系处理。

TMS 提供受管分配的暂停能力，Selective Offload 则在训练层选择保存什么。两者并非同一层的互斥概念。具体实现见[训练状态卸载专题](/posts/gpu-memory-vmm-tms-offload)。

---

## 六、端到端：一次 release / resume 走完全程

以 verl + SGLang、`sleep_level=2` 为例，把各层串起来（箭头表示「调用」）：

释放（训练开始前）：
```text
verl on_sample_end → abort + release(tags=["kv_cache","weights"])
  → SGLang HTTP release_memory_occupation
    → adapter.pause("kv_cache") / adapter.pause("weights")
      → torch_memory_saver.pause → TorchMemorySaver::pause
        → 每块 ACTIVE 分配：cuMemUnmap + cuMemRelease（nvidia-smi 显存下降）
```
恢复（训练结束后）：
```text
verl on_step_end → resume(tags=["weights"]) → update_weights → resume(tags=["kv_cache"])
  → SGLang HTTP resume_memory_occupation
    → adapter.resume → TorchMemorySaver::resume
      → cuMemCreate + cuMemMap 回原 ptr + cuMemSetAccess
      → 写入有效权重、重建缓存状态，满足 Graph 依赖后才恢复生成
```
若换成 vLLM 路线：`llm.sleep(level=2)` / `llm.wake_up()` → `CuMemAllocator.sleep/wake_up`
→ `_unmap_and_release` / `_create_and_map`，底层同样是 `cuMem*` 系列调用。

---

## 七、分层锚点（仓库内相对路径）

PyTorch
- `pytorch/torch/cuda/memory.py` — empty_cache / memory_stats / memory_snapshot
- `pytorch/c10/cuda/CUDACachingAllocator.h` — CUDAAllocator 接口
- `pytorch/c10/cuda/CUDACachingAllocator.cpp` — Block / alloc_block / emptyCache / release_blocks
- `pytorch/torch/csrc/cuda/Module.cpp` — Python↔C++ 绑定

torch_memory_saver
- `torch_memory_saver/torch_memory_saver/entrypoint.py` — region / pause / resume / disable
- `torch_memory_saver/torch_memory_saver/hooks/mode_torch.py` — 绑定 pluggable allocator
- `torch_memory_saver/csrc/core.cpp` — malloc / pause / resume / free
- `torch_memory_saver/csrc/core.h` — AllocationMetadata

推理框架
- `sglang/python/sglang/srt/utils/torch_memory_saver_adapter.py`
- `vllm/vllm/device_allocator/cumem.py`

RL 训练框架
- `slime/slime/ray/rollout.py` · `slime/slime/ray/actor_group.py` · `slime/slime/backends/sglang_utils/sglang_engine.py`
- `verl/verl/trainer/ppo/v1/trainer_base.py` · `verl/verl/trainer/ppo/v1/trainer_colocate_async.py`
- `verl/verl/workers/engine_workers.py` · `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`

> 注：文中行号仅供参考，可能随版本变化，建议以函数名定位。



## 八、怎样验证显存轮换有效

先确认正确性，再比较性能。固定模型、并行配置、batch/序列长度、Graph 设置与权重同步方式，对比不卸载和目标卸载策略。

| 检查项 | 观察点 |
| --- | --- |
| 地址与内容 | 受管 Tensor 的地址是否稳定；备份区是否恢复旧值；无备份区是否先写入再读取 |
| 设备显存 | 分别观察 allocated、reserved 与设备级占用，记录 rollout、训练和权重更新的峰值 |
| 主机内存 | CPU backup 的 RSS、Pinned RAM，以及多轮循环后是否持续增长 |
| 切换时间 | 分开测量 Sleep、Wake、同步和权重更新；异步 API 返回时间不等于 GPU 工作完成时间 |
| 端到端结果 | 在相同训练工作量下比较 step 时间、生成吞吐和训练结果；不能只凭显存下降推断吞吐提高 |

尤其应观察**权重更新期间**的瞬时峰值。先唤醒权重、完成更新，再申请 KV cache，可以避免过早恢复全部推理资源；但收益仍要由具体负载验证。

## 参考资料

- [PyTorch CUDA 显存管理、Stream 与 CUDA Graph](https://docs.pytorch.org/docs/stable/notes/cuda.html)：缓存池、`record_stream`、graph-private pool 与 replay 约束。
- [NVIDIA CUDA Driver VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html)：地址预留、物理分配、映射与访问权限。
- [TMS entrypoint.py](https://github.com/fzyzcjy/torch_memory_saver/blob/f67956841b39ffc1ab3103f23a717d7fcdc36222/torch_memory_saver/entrypoint.py) 与 [core.cpp](https://github.com/fzyzcjy/torch_memory_saver/blob/f67956841b39ffc1ab3103f23a717d7fcdc36222/csrc/core.cpp)：本文 CUDA pause/resume 与接入边界的实现依据。
- [vLLM CuMemAllocator](https://github.com/vllm-project/vllm/blob/955f85be04b6250d26e6915e26bdf6720fbc8258/vllm/device_allocator/cumem.py) 与 [Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)：另一条 VMM 路线及原生 sleep level 语义。
- [verl SGLang rollout](https://github.com/verl-project/verl/blob/8612d0292c38dbe7b104a8c783594f5c57119fa7/verl/workers/rollout/sglang_rollout/sglang_rollout.py)、[engine workers](https://github.com/verl-project/verl/blob/8612d0292c38dbe7b104a8c783594f5c57119fa7/verl/workers/engine_workers.py) 与 [colocate async trainer](https://github.com/verl-project/verl/blob/8612d0292c38dbe7b104a8c783594f5c57119fa7/verl/trainer/ppo/v1/trainer_colocate_async.py)：核对 tag、LoRA base weights 保留分支和阶段钩子。
- [SGLang adapter](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/torch_memory_saver_adapter.py) 与 [slime](https://github.com/THUDM/slime)：结合对应路径追踪引擎接入与编排。
