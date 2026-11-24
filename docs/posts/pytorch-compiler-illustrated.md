---
title: 图解 PyTorch 编译器：从 Dynamo 到 Inductor 与 Triton
description: 通过算子融合、FX Graph、Guards、AOTAutograd、Inductor 调度与 CUDA Graphs，理解 torch.compile 的整个编译与执行过程。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="pytorch-compiler" tabindex="-1">图解 PyTorch 编译器：从 Dynamo 到 Inductor 与 Triton</h1>
  <p class="asb-story-subtitle">torch.compile 如何从 Python 程序生成更少、更高效的 GPU Kernel？</p>
  <p class="asb-story-deck">在 Eager Mode 下，模型里的逐元素操作往往变成很多独立 Kernel。torch.compile 尝试捕获可编译区域，把算子关系转成图，再针对具体 Shape 与硬件生成代码。它既可能减少内存访问和 Launch 开销，也可能引入冷启动、Guard 和重新编译成本。</p>
</div>

`torch.compile()` 看似是一次函数包装，但背后至少有两类工作：**捕获程序中可优化的计算关系**，以及**为给定硬件选择合适的算子融合、调度和代码生成路径**。理解它时，应该把前端的 Dynamo 与后端的 Inductor 分开，也要区分训练时的 AOTAutograd 与另一个含义不同的离线 AOT Compile。

学习材料包括 Cishoon 的《图解 PyTorch 编译器》系列，以及 [PyTorch Compiler 官方文档](https://docs.pytorch.org/docs/stable/torch.compiler.html)。下文围绕图捕获、Guard、反向传播、融合与执行展开。

## 一、Eager Mode 的开销从哪里产生？

考虑一个简单的逐元素表达式：


$
Y=\operatorname{relu}(X\cdot 2+1)
$


普通 Eager 执行可能在 GPU 上分别发起乘法、加法和 ReLU 计算，并为中间结果分配或写回 Tensor。对于计算量很小的 Pointwise 操作，Kernel Launch 与中间数据的 HBM 读写可能比实际算术更值得优化。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 PyTorch 算子融合示意"><img src="/diagrams/compiler-fusion.svg" alt="Eager 多算子与融合 Kernel 的内存访问路径" loading="lazy" /></button>
  <figcaption>融合允许中间计算在一个 Kernel 中完成，避免部分中间 Tensor 的物化和额外 Launch。</figcaption>
</figure>

但并不是所有操作都能无条件融合。涉及不同数据依赖、特殊 Reduction、外部副作用、同步或资源约束的算子可能需要独立执行。编译器必须保证计算语义与可接受的数值误差边界。

## 二、torch.compile 的整体流水线

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 PyTorch 编译器全流程"><img src="/diagrams/compiler-stack.svg" alt="TorchDynamo、AOTAutograd、Inductor 和 Triton 之间的关系" loading="lazy" /></button>
  <figcaption>Dynamo 负责捕获可编译区域；AOTAutograd 处理训练所需计算图；Inductor 执行优化与目标代码生成。</figcaption>
</figure>

一个简化的入口如下：

```python
import torch

@torch.compile
def fn(x):
    return torch.relu(x * 2 + 1)

x = torch.randn(1024, device="cuda")
y = fn(x)
```

第一次调用可能需要捕获与编译，后续满足相应 Guard 的调用才可能复用已经生成的结果。编译得到的代码可以来自 Inductor/Triton 或其他 Backend，不应将“调用 torch.compile”理解成每次都重新生成一个 GPU Kernel。

## 三、TorchDynamo：在 Python 执行边界捕获图

[TorchDynamo](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html) 使用 CPython Frame Evaluation 机制及字节码分析，识别能够转换为 FX Graph 的 PyTorch 操作。它尝试在保持 Python 运行语义的前提下抽取图，而不是把任意 Python 程序直接翻译为 GPU 机器码。

### 为什么 Dynamo 可以“不执行真正的大矩阵乘”也获得计算图？

图捕获过程中会使用表示 Shape、Dtype 等元信息的代理或 Fake Tensor 机制来推理计算关系；在合适的路径上，无需先对真实大型 Tensor 完成全部 GPU 运算即可描述图结构。

但是**并不是所有 Python 分支、外部 I/O 和自定义副作用都能安全捕获**。遇到不支持的操作时，默认策略可能发生 Graph Break，先编译已捕获区域，再回到 Eager 执行相应 Python 代码。

### Graph Break 并不等于整个程序编译失败

Graph Break 会打断融合范围，并使多个编译图之间存在 Python 调用开销。因此它可能损害性能，但不必然影响结果正确性。

`fullgraph=True` 可以用于更严格地检查是否存在未捕获路径；生产代码是否应该强制 Fullgraph，需要结合模型动态控制流和编译收益评估。

## 四、Guards 与 Recompilation：同一个函数为何编译多次？

一个编译版本通常依赖某些假设：输入 Tensor 的 Shape / Dtype、设备、模块属性或相关 Python 状态。**Guard** 会检查当前调用是否还满足这些假设。

假设一个函数第一次接收 `[32, 128]`，后续接收 `[64, 128]`。如果当前编译图是针对固定尺寸生成的，而动态 Shape 处理不能覆盖新输入，就可能触发另一轮捕获和编译。

这意味着动态请求较多的推理服务，需要同时关心：

- Guard 检查本身的开销。
- 因 Shape / Dtype / Python 状态变化导致的 Recompilation。
- 动态 Shape 支持范围与生成代码的泛化程度。
- 缓存是否被正确复用，以及是否触及编译结果数量或资源限制。

排查时可使用 `TORCH_LOGS=recompiles`、相关 Guard 日志和官方建议的分析工具，而不是只统计第一次调用的运行时间。

## 五、AOTAutograd：为什么训练图比推理图复杂？

训练不仅执行 Forward，还需要为 Backward 保存或重新计算某些中间状态。AOTAutograd 可以在编译过程中追踪并处理前向与反向计算图，配合 Decomposition、Functionalization 等机制使复杂算子形式更适合后端优化。

例如某些高层操作可以分解为一组更基础的算子，而原地修改也需要被转换为具有清晰副作用语义的形式。最终哪些中间结果被保存、哪些计算重算，以及是否存在额外内存开销，会影响训练性能。

**AOTAutograd 与“把编译产物打包部署”的 AOT Compile 不是同一个概念**。后者属于额外的提前编译与部署工作流，具体 API 和稳定性取决于 PyTorch 版本。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：矩阵乘加法 ReLU 与逐行求和的前反向图"><img src="/images/figures/pytorch-compiler-illustrated-01.jpg" alt="矩阵乘加法 ReLU 与逐行求和的前反向图" loading="lazy" /></button>
  <figcaption>矩阵乘、ReLU 和逐行求和的前反向对应关系：反向需要前向保存的输入与 ReLU 掩码。来源：<a href="https://www.bilibili.com/video/BV149h16QE7p">B 站 Cishoon《图解 PyTorch 编译器》</a>视频画面。</figcaption>
</figure>

### 联合图如何决定保存与重算

前向输出之外，反向还需要部分中间结果。AOTAutograd 可在联合图上使用分区策略，将计算切成前向与反向，并选择保存哪些值、重算哪些操作。最小割重计算分区器是其中一种实现，代价模型决定了内存和重计算之间的取舍。

Decomposition 将适合分解的高层操作表达为基础算子组合；Functionalization 则处理原地修改及别名语义，使图更便于分析。未被后端直接生成的操作也可以通过受支持的外部算子路径执行。

## 六、TorchInductor：IR、融合与代码生成

Dynamo 提供的图进入 Inductor 等后端后，需要进行调度和代码生成。Inductor 的主要工作包括识别算子读写关系、决定 Fusion Group、安排 Loop/Block 结构，以及为目标设备选择 Kernel 实现。

### Pointwise / Reduction 与 GEMM 不一定采用同一策略

对逐元素操作，Inductor 可以通过组合计算表达式减少 HBM 中间结果；对 GEMM 等高计算密度算子，则可能使用针对硬件优化的模板或库实现，或执行 Autotune 搜索参数。

因此不能将“torch.compile 优化性能”简化为“所有算子都改写成 Triton”。实际生成路径可能涉及 Triton、C++/CPU、模板 Kernel 或外部数学库。

### Triton 的 Block 与 GPU 程序实例

Triton 允许开发者以 Block 为粒度描述索引和张量运算，编译器把这些程序实例映射到 GPU 执行资源上。有效的 Block Size 取决于并行度、寄存器、Shared Memory、访存模式和计算强度。

所谓“融合更多算子就一定更快”并不成立：更大的 Fusion Group 可能导致寄存器压力、Occupancy 下降或重复计算，最终必须依靠 Benchmark 验证。

### 编译缓存与 max-autotune

[编译缓存](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)可以复用图、生成代码或调优结果等产物。进程内 Guard 命中和重启后的持久缓存命中是不同层次；跨进程复用还受软件版本、设备及缓存键约束。

`mode="max-autotune"` 会在适用路径上搜索并测量候选实现，例如矩阵乘模板配置，以增加编译成本换取稳态性能的机会。缓存可减少重复搜索，但具体收益仍由形状和硬件决定。

## 七、CUDA Graphs 与编译器的关系

CUDA Graph 可以降低重复 Kernel Launch 的 CPU 调度成本，但它和 Dynamo 捕获的 FX Graph 是**两个不同层次的 Graph**：

- FX Graph 记录可优化的 PyTorch 算子关系。
- CUDA Graph Capture / Replay 记录特定 CUDA 工作路径及相关设备地址和执行条件。

`torch.compile(mode="reduce-overhead")` 等配置可以在适用场景利用 CUDA Graph 降低 Overhead，但这不是普遍保证。动态 Shape、输入变异、控制流、Buffer 地址和部分不兼容操作都可能影响 Capture / Replay 的可行性。

像 vLLM 这类请求长度和批量大小持续变化的系统，会通过 Bucket、形状专门化或特定的 Graph 管理策略减少 Overhead，而不是假定任意请求都能复用同一张 Graph。

## 八、什么时候值得使用 torch.compile？

评估一个编译优化时，应分别记录：

| 指标 | 重要原因 |
| --- | --- |
| Cold Compile Time | 首次编译可能远高于正常推理时间 |
| Steady-State Latency | 决定是否有真实运行时收益 |
| Recompile / Guard Frequency | 动态输入时影响吞吐与尾延迟 |
| GPU Kernel 数和 Launch 开销 | 判断是否减少碎片化的小算子 |
| Peak Memory / Workspace | 编译模式和图捕获可能增加显存开销 |
| Correctness / Numeric Drift | 融合与算子重排可能影响浮点数值 |

简单说，**编译器的目标不是让一个模型“被编译过”，而是让相同工作负载在满足正确性约束下更高效地执行**。

## 九、总结

理解 `torch.compile` 可以沿一条主线展开：**Dynamo 捕获图 → Guard 决定复用条件 → AOTAutograd 处理训练图 → Inductor 决定融合与调度 → Triton/其他 Backend 执行 GPU 或 CPU Kernel**。

性能问题则需要反向排查：是 Graph Break 太多、Shape 触发反复编译、Kernel Launch 过于分散、Fusion 增大寄存器压力，还是 CUDA Graph 不适用？

## 参考资料

[PyTorch torch.compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html) · [Dynamo Overview](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html) · [Compiler Programming Model](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.html) · [Graph Breaks](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html) · [Reducing Compile Time](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html)


