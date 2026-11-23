---
title: SGLang 服务启动日志详解：从权重加载到 CUDA Graph 与 KV Cache
description: 以 Qwen3-0.6B 在 RTX 5090 上的一次 SGLang 启动记录为例，分析模型加载、KV Cache 分配、Graph 捕获、调度器初始化与离线加载故障。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="sglang-startup-logs" tabindex="-1">SGLang 服务启动日志详解：从权重加载到 CUDA Graph 与 KV Cache</h1>
  <p class="asb-story-subtitle">用一份真实启动日志理解推理服务的资源与执行初始化</p>
  <p class="asb-story-deck">模型加载为什么只花很短时间，CUDA Graph 却要逐个捕获不同形状？KV Cache 的预留容量如何影响可接纳请求？当 Hugging Face 连接失败时，哪些错误应当阻止启动，哪些只影响可选配置？沿同一份启动记录逐段分析。</p>
</div>

本文分析一份具体的启动记录：2026 年 9 月 8 日，单张 **RTX 5090（32 GB）** 上启动 **Qwen/Qwen3-0.6B**，使用的命令为：

```bash
sglang serve --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30000
```

日志数值属于这一台设备、这一模型和当时的软件配置，**不是所有 SGLang 版本的固定默认值**。以下会区分日志中明确出现的事实与基于推理框架机制的解释，避免将一次运行记录误当成普遍结论。

## 一、从启动过程看 SGLang 的主要工作

启动不仅是加载一份模型权重，还包括解析配置、创建 Worker、初始化分布式上下文、分配 KV Cache、执行 Graph Capture、初始化调度和 HTTP 接口。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 SGLang 启动阶段图"><img src="/diagrams/sglang-startup-stages.svg" alt="SGLang 配置、Distributed、权重、KV Cache、CUDA Graph 和 HTTP 服务初始化" /></button>
  <figcaption>启动阶段之间存在依赖，但细节和顺序受版本、执行后端与功能开关影响。</figcaption>
</figure>

可以粗略将运行时职责分为三部分：Tokenizer Manager 处理请求进入和 Token 编解码等工作；Scheduler 负责请求调度、Batch 与相关模型执行管理；Detokenizer Manager 负责生成结果的解码与输出。并不是每个日志行都能直接对应一个独立的 OS 进程或一个固定组件。

## 二、参数解析：实际运行配置比命令行更重要

日志一开始显示：

```text
[20:51:24] Attention backend not specified.
             Use flashinfer backend by default.
[20:51:25] server_args={'model_path': 'Qwen/Qwen3-0.6B', ...}
```

这里有两个重要信息。首先，未指定 Attention Backend 时，当时版本和环境选择了 FlashInfer；**这不是所有 GPU、模型或新版 SGLang 都必然选择 FlashInfer 的保证**。其次，`server_args` 记录最终生效的配置，应该比“我以为命令行参数是什么”更值得信任。

本次日志中有以下参数：

| 参数 | 日志中的值 | 影响 |
| --- | --- | --- |
| `mem_fraction_static` | 0.874 | 控制部分静态显存预留的资源预算 |
| `context_len` | 40960 | 模型配置与运行时约束下的上下文长度 |
| `chunked_prefill_size` | 2048 | Chunked Prefill 切分规模 |
| `max_running_requests` | 3001 | 调度层的配置上限，不等于 GPU 真能同时容纳这么多长请求 |
| `schedule_policy` | fcfs | 采用的请求调度策略 |

启动时还检测到了模型对应的 Reasoning / Tool Parser 配置，使后续服务能够处理适当的生成与解析语义。

## 三、Distributed 初始化和权重加载

日志中的分布式初始化片段：

```text
[20:51:33] Init torch distributed begin.
[20:51:33] Init torch distributed ends.
             elapsed=0.11 s, mem usage=0.08 GB
```

即使单卡部署，运行时也可能沿统一的分布式初始化路径准备相关通信与执行组件。这并不意味着单卡必须发生跨 GPU AllReduce，而是说明软件将单卡视为可扩展分布式运行时的一种配置。

模型权重阶段的日志显示：

```text
[20:51:34] Load weight begin. avail mem=30.71 GB
[20:51:34] Found local HF snapshot ... skipping download.
[20:51:34] Load weight end. elapsed=0.34 s,
             type=Qwen3ForCausalLM,
             avail mem=29.53 GB, mem usage=1.18 GB.
```

在该次运行中，权重加载阶段约 **0.34 秒**，记录的显存变化与 0.6B 参数规模和 BF16 存储量级相符。但不能把它直接解释成“任意 0.6B 模型都能在 0.34 秒内加载”：本地 Snapshot、文件格式、权重切片、磁盘缓存和初始化逻辑都会影响耗时。

本次还有 `Parameter lm_head.weight not found in params_dict` 相关日志，服务仍继续运行。在共享输出层/Embedding 权重的模型上，这可能是预期行为；**遇到相同文本应当结合模型结构和后续权重校验判断**，不能把这一告警一概视为永远可忽略。

## 四、KV Cache 分配：为什么它比模型权重更占显存？

这次启动中最醒目的两个数值来自 KV Cache：

```text
[20:51:35] KV Cache is allocated.
             dtype: torch.bfloat16, #tokens: 240120,
             K size: 12.82 GB, V size: 12.82 GB
[20:51:35] Memory pool end. avail mem=3.36 GB
```

也就是说，日志记录的 K 与 V Cache 各约 **12.82 GB**，合计约 **25.64 GB**；与权重的 1.18 GB 相比，KV Cache 占了明显更大的显存比例。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 SGLang 显存账本"><img src="/diagrams/sglang-gpu-budget.svg" alt="日志中模型权重、KV Cache K/V 和其他显存资源的占用关系" loading="lazy" /></button>
  <figcaption>数字来自这次 RTX 5090 启动记录，不是 SGLang 通用的显存划分比例。</figcaption>
</figure>

### 240120 个 Token 等于 240120 个并发请求吗？

**不是。** `#tokens: 240120` 表示这份 KV Cache 的 Token 容量量级；一个请求可能占用上千 Token，多个请求还有不同长度的历史上下文、前缀缓存共享及临时工作区开销。能够同时运行多少请求，受每请求 KV 占用、模型结构、调度策略和剩余显存共同制约。

同理，`max_running_requests=3001` 是调度配置上限，不等于 3001 条请求都能在内存和算力条件下同时执行。

`mem_fraction_static` 也不是一个能把静态显存精确拆成全部项目的万能公式。想核算真实显存预算，需要同时测量权重、KV Pool、CUDA Graph Workspace、运行时其他 Buffer 与剩余显存。

## 五、CUDA Graph Capture：为什么启动时要运行很多遍模型？

日志中的 Graph Capture 片段：

```text
[20:51:36] Capture target prefill CUDA graph begin.
             backend=breakable, num_tokens=[4, 8, ..., 2048]
[20:51:39] Capture target prefill CUDA graph end.
             elapsed=3.81 s

[20:51:39] Capture target decode CUDA graph begin.
             backend=full, num_tokens_per_req=1,
             bs=[1, 2, 4, 8, 12, 16, 24]
[20:51:40] Capture target decode CUDA graph end.
             elapsed=0.82 s
```

CUDA Graph 允许将适用的 GPU 调用序列捕获后重复 Replay，从而减少重复启动大量 Kernel 的 CPU 提交开销。特别是在短序列 Decode 或小 Batch 下，这类开销有可能占据较大比例。

本次配置对 Prefill 捕获了多档 Token Shape，对 Decode 捕获了多档 Batch Size。**具体形状列表是该运行的配置结果**；不同版本、模型、显存预算与执行模式可能采用不同的 Graph 策略。

CUDA Graph 的重要限制包括设备地址稳定、可捕获操作集合、Buffer 生命周期和某些动态形状问题。Graph Capture 成功也不能直接推出推理吞吐提升了固定倍数，需要在相同真实负载上比较 Graph On/Off。

## 六、Scheduler、Radix Cache 与启动就绪

完成缓存和 Graph 初始化后，日志包含：

```text
max_total_num_tokens=240120,
chunked_prefill_size=2048,
max_prefill_tokens=16384,
max_running_requests=3001
Init Unified Radix Cache...
```

`max_total_num_tokens` 与上文 KV Cache 容量对应，是调度资源预算的重要依据。Radix / Prefix Cache 管理可复用的历史 KV 前缀，让具有相同前缀的后续请求有机会避免重复计算；但命中率取决于真实请求分布，不能仅凭打开 Cache 就保证加速。

HTTP 服务随后开始监听并完成启动检查：

```text
[20:51:41] Uvicorn running on http://0.0.0.0:30000
[20:51:42] GET /model_info ... 200 OK
[20:51:42] POST /generate ... 200 OK
[20:51:42] The server is fired up and ready to roll!
```

这里最有价值的是把**端口已经监听**与**模型请求真的能够被服务处理**区分开。前者只说明 Web Server 启动，后者还需要模型执行和必要的调度初始化成功。部分启动脚本会进行内部预热或探活请求，具体做法依赖版本。

## 七、Hugging Face 网络错误为何会拖慢冷启动？

这份材料记录了一个典型的环境问题：模型权重已经位于本地缓存，但 Model ID 仍触发远端配置探测或缓存元数据请求。在网络不可达时，某些可选请求失败可以被跳过，某些带重试的 HEAD 请求则可能明显延长启动时间。

本次排查采用了：

```bash
export HF_HUB_OFFLINE=1
```

在缓存文件完整的情况下，Hugging Face 离线模式可以阻止部分远程 Hub 查询，使加载逻辑直接使用本地资源。但**如果权重、Tokenizer 或配置尚未缓存完整，离线模式也可能导致启动失败**。因此它适用于已有完整本地模型资源、网络访问不稳定的场景，不是所有启动故障的通用解决方案。

另一种方式是使用真实存在的本地模型目录作为 `--model-path`，并确认相关子文件完整。

## 八、看到这些日志时，应该怎么判断？

| 日志现象 | 优先检查 | 不应立刻下的结论 |
| --- | --- | --- |
| Attention Backend 自动选择 | 实际设备与后端支持 | 所有版本都使用相同 Backend |
| `lm_head.weight` 缺失提示 | 是否 Shared / Tied Weights | 任何缺失权重都可忽略 |
| KV Cache 容量较大 | KV Token 数、剩余显存 | 请求并发数必然等于 Cache Token 数 |
| CUDA Graph Capture 较慢 | Captured Shapes / Warmup 配置 | 所有模型启动都会同样慢 |
| HF 网络请求超时 | 本地缓存完整性和离线配置 | 模型本体一定损坏 |
| HTTP Server 已监听 | 模型执行健康检查 | 服务已经能正确处理所有推理请求 |

## 九、总结：把日志当作资源与状态转换的证据

一份启动日志可以揭示推理服务的配置、硬件资源预算、模型权重、KV Cache、Graph Capture、调度与 HTTP 服务的不同生命周期。但每个字段都需要放在**具体版本、模型、硬件和时间范围**内解释。

这个案例最有代表性的观察是：**0.6B 模型的权重不是占用最多显存的部分，KV Cache 才是主要资源；Graph Capture 则是一项可能显著影响冷启动时间的准备工作。** 这也解释了为什么排查“启动慢”和“并发吞吐低”往往需要分别观察。

## 参考资料

[SGLang 官方项目](https://github.com/sgl-project/sglang) · [SGLang 文档](https://docs.sglang.ai/) · [Hugging Face Hub 环境变量](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) · [CUDA Graph](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

日志数值与启动耗时来自一次指定环境的实际记录，属于示例而非跨硬件的性能承诺。不同 SGLang 版本的模块名、CUDA Graph 策略和启动阶段顺序可能发生变化。
