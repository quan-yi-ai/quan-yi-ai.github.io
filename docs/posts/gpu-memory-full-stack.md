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

围绕 GPU 显存的分配与 pause/resume，涉及从底层驱动到上层应用的四个层次，各层职责如下：

1. **PyTorch 显存管理（起点）**：Caching Allocator / MemPool / 可插拔分配器。原生分配器地址不稳定，由此引出下一环。
2. **torch_memory_saver**：借助 CUDA VMM 分离虚拟地址与物理显存，实现显存 pause/resume，作为底层能力被上层集成。
3. **推理框架（SGLang / vLLM）**：封装为 release/resume_memory_occupation、sleep/wake_up，是能力落地的场景。
4. **RL 训练框架（slime / verl）**：在 colocate 下编排训练与推理共享显存。

完整调用链见下一节图示。

涉及仓库：`pytorch`、`torch_memory_saver`、`sglang`、`vllm`、`slime`、`verl`。
下文代码路径均为仓库内相对路径（以仓库名开头），克隆对应仓库后即可对照定位。

几个高频术语：
- **KV cache**：自回归推理时缓存每个 token 的 Key/Value，避免重复计算，是推理期显存的大头。
- **CUDA Graph**：把一串 GPU kernel 录制成图、整体重放以降低启动开销；录制时会固化其中用到的显存地址。
- **colocate**：训练与推理部署在同一批 GPU 上、分时复用显存的方式。
- **rollout**：RL 中用当前策略模型批量生成样本的推理过程。

> 说明：文中代码为便于阅读的**简化摘录**（非逐行原文），完整实现以对应源文件为准；
> 内容基于各仓库 2026-11 前后的主分支，源码可能随版本演进，建议以函数名而非行号定位。

---

## 零、核心结论

三种「释放显存」处在完全不同的层次，务必区分：

- `del tensor`：Tensor 析构，block 退回 PyTorch 内存池；`reserved` 不变，驱动层面毫无变化。
- `torch.cuda.empty_cache()`：把内存池中**完全空闲**的 block 归还驱动；不影响任何仍存活的 Tensor。
- `torch_memory_saver.pause()`：对**仍存活**的 Tensor 解除物理显存映射、但**保留虚拟地址**；`resume()` 后 `data_ptr()` 不变。

关键机制：CUDA VMM（虚拟内存管理）将「虚拟地址」与「物理显存」解耦。
这正是 CUDA Graph 在 pause/resume 后仍然有效的前提——地址保持不变。

---

## 一、整体调用链

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：整体调用链：从 RL 框架到 CUDA VMM"><img width="780" height="560" src="/images/gpu-memory-full-stack/overall-stack.svg" alt="整体调用链：从 RL 框架到 CUDA VMM" loading="lazy" /></button>
</figure>

---

## 二、PyTorch 显存底层
> 这一层是整条链路的基础。核心是 **Caching Allocator**：PyTorch 在 `cudaMalloc/cudaFree` 之上实现的一层显存缓存池。

### 2.0 为什么需要 Caching Allocator
`cudaMalloc` / `cudaFree` 开销很大：二者是**同步**调用，会阻塞 stream、打断执行流水线。
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
关键点：**绝大多数「分配」并不涉及驱动调用**，只是从内存池中取用一个 block。这正是
`memory_allocated`（Tensor 实占）远小于 `memory_reserved`（内存池向驱动申请的总量）的原因。

### 2.3 释放与 empty_cache：两种「释放」
文件：`pytorch/torch/cuda/memory.py` · `pytorch/torch/csrc/cuda/Module.cpp`

```python
def empty_cache():                 # memory.py
    if is_initialized():
        torch._C._cuda_emptyCache()   # → THCPModule_emptyCache (Module.cpp:569)
                                      # → CUDACachingAllocator::emptyCache
```

- `del tensor`：将 block 标记为 `allocated=false`，**退回内存池**，`reserved` 不变（驱动层面未动）。
- `empty_cache()`：把内存池中**完全空闲**的 segment 真正 `cudaFree` 归还驱动，`reserved` 才下降。
  调用链：`emptyCache → release_cached_blocks → release_blocks → cudaFree`。
- 它**不会**回收仍被 Tensor 引用的 block，也无法回收其他 stream 仍在使用的 block。

四个常用指标（`memory_stats` / `THCPModule_memoryStats`，Module.cpp:578）：
```text
allocated = 当前被 Tensor 占用        reserved = 内存池向驱动申请的总量
active    = 使用中及待释放的 block     inactive_split = split 出来的空闲碎片
```

### 2.4 stream 语义：地址复用为何需要同步
Block 记录创建它的 `stream`，默认只能在**同一 stream** 上复用。若要在其他 stream 上使用某 tensor，
必须调用 `tensor.record_stream(s)` 登记，分配器借助 CUDA event（`event_count`）确认该 stream
使用完毕后才会复用这块显存。否则会出现「显存已被复用、旧 kernel 仍在读取」的数据竞争。
这与 pause 前必须先 synchronize 是同一个道理：**异步执行下，回收显存前必须确认不再有人使用**。

### 2.5 为什么地址不稳定
默认分配器中，block 会 split / merge，segment 会被 `empty_cache` 整段 `cudaFree`，
同一个逻辑 tensor 下次拿到的 `ptr` 完全可能变化。这对常规训练影响不大，但对
**CUDA Graph、需要跨阶段保留地址**的推理场景无法接受——这正是 torch_memory_saver 要解决的问题。

### 2.6 可插拔分配器：官方的三个接入组件（TMS / vLLM 的接入点）
```text
CUDAPluggableAllocator  以外部 malloc/free 函数指针替换分配实现
MemPool                 一个独立的 block 池，可绑定自定义分配器
use_mem_pool(pool)      上下文作用域内的分配都走这个 pool
```
`use_mem_pool` 让第三方能**只接管一段作用域**内的分配，而非全局替换分配器。
torch_memory_saver 的 `region()`、vLLM 的 `use_memory_pool()` 都基于这组机制。

---

## 三、torch_memory_saver（pause/resume 本体）

> torch_memory_saver 是整条链路的核心，pause/resume 的具体实现都在这里。

### 3.0 为什么采用 CUDA VMM，而非 del + 重新 alloc

一种朴素做法是「pause 时 `del tensor`、resume 时重新 `torch.empty`」，但它存在一个致命问题：
**重新分配得到的虚拟地址几乎必然改变**。而推理引擎 decode 阶段依赖 **CUDA Graph**，
graph 捕获时会把 kernel 参数中的**显存地址固化**到 graph node 中，replay 时不再重新读取指针。
一旦地址变化，graph replay 便会访问到错误或非法地址。

CUDA VMM 的价值就在这里：它把一次分配拆成两种相互独立的资源——

- **虚拟地址**（VA range）：由 `cuMemAddressReserve` 保留，是进程地址空间里的一段编号。
- **物理显存**（physical handle）：由 `cuMemCreate` 创建，是 GPU 上真正的页。
- **映射**：`cuMemMap` 把前者绑到后者，`cuMemSetAccess` 赋读写权限。

于是 pause 可以只回收**物理显存**、保留**虚拟地址**；resume 时物理页是全新的，
但重新映射回**同一段虚拟地址**。对上层（PyTorch allocator、tensor、CUDA Graph）来说，
`data_ptr()` 从头到尾没变，一切透明。这是 TMS 存在的根本理由。

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
  <button class="asb-zoom-trigger" type="button" aria-label="放大：pause / resume 时序"><img width="760" height="430" src="/images/gpu-memory-full-stack/pause-resume-sequence.svg" alt="pause / resume 时序" loading="lazy" /></button>
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
当预留的物理显存与 margin 之和超过空闲显存时主动返回 OOM——避免 resume 时耗尽显存、拖垮其他进程。

### 3.3 pause：释放物理，保留虚拟地址
`TorchMemorySaver::pause(tag)` 遍历 metadata，对 tag 匹配且处于 ACTIVE 的分配：

```cpp
if (metadata.enable_cpu_backup)  cpu_backuper::offload(ptr, raw_size, cpu_backup); // D2H 拷贝
else if (enable_disk_backup)     disk_backend_.offload(...);
cuMemUnmap((CUdeviceptr)ptr, allocation_size);  // 解除映射
cuMemRelease(allocHandle);                        // 归还物理显存 → nvidia-smi 立刻下降
metadata.state = PAUSED;
```

关键点：
- **只 Unmap + Release 物理句柄，没动虚拟地址**。那段 VA 仍被本进程保留着，别的分配不会占用它——这正是 resume 能映射回原地址的前提。
- 物理显存**真的还给了驱动**，`nvidia-smi` 能看到下降（这与 `empty_cache` 只在进程内缓存层面回收不同）。
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
和 pause 前并非同一块显存，但因为映射回了同一段虚拟地址，上层对此毫无感知。
在没有 backup 的情况下，resume 后得到的是**未定义内容**（MUSA 平台会额外清零一次，CUDA 不会）。

### 3.5 pause 前为何必须先 synchronize
文件：`torch_memory_saver/torch_memory_saver/entrypoint.py` · `pause()`

Python 层在调 C++ `tms_pause` **之前**先 `synchronize` 受影响设备。原因：
`cuMemUnmap` 会立刻让那段地址失效，如果此时还有 kernel 在异步队列里排队、即将访问这块显存，
unmap 后 kernel 一跑就是**非法内存访问**。所以必须先把设备上 outstanding 的工作排空。
这条「先同步再 unmap」是 pause 正确性的硬约束，不是性能优化。

### 3.6 CPU backup：pinned vs mmap
文件：`torch_memory_saver/csrc/cpu_backup.h`

- `offload` = `cudaMemcpy(DeviceToHost)`，`onload` = `cudaMemcpy(HostToDevice)`。
- 两种后端：`PINNED`（`cudaMallocHost`，DMA 快但占不可换出的锁页内存）与
  `MMAP`（匿名 mmap，可被 OS 回收 RSS，但 H2D 是 pageable、onload 后要 `cudaDeviceSynchronize`）。
- `retain_cpu_backup=True` 时 resume 不释放 CPU 备份，省去下次 pause 的再分配——
  典型用于权重这种每轮都要 offload/onload 的固定缓冲。

### 3.7 free 与 pause 的区别
```text
pause: cuMemUnmap + cuMemRelease               ；保留 VA；metadata 保留（PAUSED）
free : cuMemUnmap + cuMemRelease + cuMemAddressFree；归还 VA；metadata 删除
```
free 还会 `cudaDeviceSynchronize` 并释放 CPU/disk backup。即 free 是「彻底销毁」，pause 是「挂起」。

### 3.8 两个常见陷阱
- **expandable_segments 冲突**：若设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
  `entrypoint.py` 的 `_sanity_checks()` 会直接报错并禁用 TMS。因为 expandable segments 自身
  也用 VMM 管理 segment，与 TMS 的 VMM 接管相互冲突。
- **NCCL CUDA Graph 注册失效**：TMS 在 pause/resume 时更换了物理 backing，若该 buffer 曾被 NCCL
  以 graph user-buffer 形式注册，注册信息会失效并导致 replay 挂起。规避方式：`export NCCL_GRAPH_REGISTER=0`，
  或不要将 NCCL 使用的 buffer 纳入 TMS region。

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
不能全局替换分配器（否则会将所有分配一并纳入），只能按作用域精确接管。

---

## 四、推理框架如何使用它（SGLang / vLLM）

> 这一层是「显存 pause/resume」真正落地的环节：推理引擎将其封装为对外的
> `release/resume_memory_occupation`（SGLang）或 `sleep/wake_up`（vLLM）。
> 两条路线机制同源（都是 CUDA VMM + MemPool），区别只在封装与策略。

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

> 这是整条链路的「需求源头」。TMS 的 tag、CPU backup、CUDA Graph 兼容等设计，都源自这一场景的需求。

### 5.0 为什么 RL 需要 pause/resume：colocate 共享 GPU
在 RLHF / colocate 架构中，**训练**（actor 的 fwd/bwd/optim）与**推理**（rollout 生成样本）
运行在**同一批 GPU** 上。两者显存需求都很大，无法同时常驻：
- rollout 阶段：推理引擎占用 权重 + KV cache，训练几乎不占。
- train 阶段：需要将显存让给训练的激活、梯度与优化器状态。

若采用「关闭推理引擎再重启」，开销过大（需重建 CUDA Graph、重新加载权重），
因此改用 pause/resume：rollout 结束 → 暂停推理显存 → 训练 → 恢复 → 更新权重 → 再次 rollout。
地址稳定使 CUDA Graph 无需重新捕获，这正是 TMS 在此不可替代的原因。

### 5.1 colocate 一轮的时序

<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：colocate 一轮的显存轮换时序"><img width="720" height="440" src="/images/gpu-memory-full-stack/colocate-loop.svg" alt="colocate 一轮的显存轮换时序" loading="lazy" /></button>
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
- **SGLang** 的 `sleep_level=1`：并未释放 weights（仅释放 KV cache），resume 时**不应**再 resume weights。
- **vLLM** 的 level-1 sleep：经由 `CuMemAllocator.sleep(offload_tags=("weights",))`，weights 已被 unmap，**必须** resume。

### 5.4 回扣：tag / sleep_level 的实际意义
至此，「为什么需要多 tag」的问题便有了答案：
- `weights` 与 `kv_cache` 的生命周期不同——权重在训练后由 `update_weights` 覆盖，KV cache 则作废后重算。
- 因此需要**分别** pause/resume：`sleep_level=1` 只处理 KV cache（权重原地保留，省去一次 offload/onload），
  `sleep_level=2` 则连权重一并释放（显存压力较大时）。
- 这正是 TMS 的 `AllocationMetadata.tag` 与 vLLM 的 `offload_tags` 存在的理由——**按业务语义分组管理显存**。

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
      → cuMemCreate 新物理页 + cuMemMap 回原 ptr（地址不变，CUDA Graph 可直接复用）
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

