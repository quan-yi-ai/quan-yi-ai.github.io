---
title: Python 并发模型：从 asyncio、GIL 到多进程
description: 从 CPython 的执行模型出发，深入理解 Event Loop、Coroutine、Task、Future、TaskGroup、GIL、线程池和多进程的正确边界。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="python并发模型" tabindex="-1">Python 并发模型：从 asyncio、GIL 到多进程</h1>
  <p class="asb-story-subtitle">协程、线程和进程，究竟分别在解决什么瓶颈？</p>
  <p class="asb-story-deck">并发不是把所有任务都变快。一次 await 为什么能让其他任务继续运行？GIL 限制了什么？线程池与进程池在 CPU 计算和 I/O 等待上有什么本质差异？从执行权、状态与资源所有权三个角度理解 Python 的并发模型。</p>
</div>

一个服务同时处理一百个请求时，瓶颈可能是网络等待；一个 Python 函数在单核上运算很久时，瓶颈可能是 CPU；几个任务共享内存又需要避免数据竞争时，问题变成了同步。**这些不是同一个问题，也就不应该寄希望于同一种并发工具。**

本文围绕 CPython 的常见执行环境，依次讨论事件循环、协程和 Task、结构化并发、GIL 与线程，以及多进程的独立地址空间。文中的代码以现代 Python 3 为背景；特别注明 Python 3.13+ 可选的 free-threaded 构建与传统启用 GIL 的解释器存在差异。

## 一、并发与并行首先是两种不同的概念

**并发（Concurrency）** 关注多个任务在一段时间内交替推进；**并行（Parallelism）** 关注多个计算在同一时刻真正运行。一个事件循环可以在单线程中并发处理多个网络请求，但并不会因此让一段纯 Python CPU 密集型循环占用多个 CPU Core。

| 场景 | 核心开销 | 常见策略 |
| --- | --- | --- |
| 同时等待多个 HTTP / Socket 请求 | I/O 等待 | `asyncio`、异步网络库 |
| 调用可能释放 GIL 的 C 扩展或执行阻塞 I/O | 执行与 I/O 调度 | Thread / ThreadPool |
| 大量纯 Python CPU 运算（传统带 GIL 构建） | CPU 执行 | 多进程、原生代码、可并行后端 |
| 长期维护模型与业务状态的服务 | 状态与生命周期 | 进程、线程、Actor 或服务组件 |

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Python 并发模型对比"><img src="/diagrams/python-concurrency-comparison.svg" alt="asyncio、线程和多进程的执行与隔离方式" loading="lazy" /></button>
  <figcaption>不同工具所管理的是不同层次：事件循环、操作系统线程与独立进程。</figcaption>
</figure>

## 二、asyncio：单线程如何处理多个等待中的任务？

### Coroutine 不是立即执行的函数结果

通过 `async def` 定义的是协程函数。调用它会得到一个 Coroutine Object，只有被 `await`、包装成 Task 或以其他方式驱动时才会执行。

```python
import asyncio

async def fetch(name, delay):
    print(f"start {name}")
    await asyncio.sleep(delay)
    print(f"done {name}")
    return name

async def main():
    result = await fetch("A", 1)
    print(result)

asyncio.run(main())
```

`asyncio.run()` 通常作为顶层异步程序入口：创建并管理事件循环，执行主协程，最后清理相关资源。一个已经运行事件循环的线程中，不应再直接嵌套调用 `asyncio.run()`。

### await 的关键作用：让出控制权

协程执行到 `await` 时，如果被等待的对象尚未完成，它可以挂起并把执行权交回事件循环。事件循环随后调度其他已经就绪的回调或任务。当 I/O 完成，相应 Task 又变为可运行。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Event Loop 调度图"><img src="/diagrams/python-event-loop-flow.svg" alt="Event Loop 在 I/O 等待期间调度其他协程执行" loading="lazy" /></button>
  <figcaption>这里的并发来自等待期间切换任务，而不是在一个线程中同时执行两段 Python 字节码。</figcaption>
</figure>

一个重要的细节是：**`async def` 里的代码并不会自动非阻塞**。如果在异步函数里调用 `time.sleep(2)` 或执行耗时的同步计算，就会堵住运行它的事件循环线程。应将其替换为异步 I/O，或适当移交线程池/进程池。

```python
async def good():
    await asyncio.sleep(1)  # 允许事件循环运行其他任务

async def bad():
    import time
    time.sleep(1)           # 阻塞当前线程的事件循环
```

## 三、Task、Future 和并发结果的组织

Coroutine 代表可被执行的协程；`asyncio.Task` 则负责把协程安排到事件循环中推进。一个 Task 同时也是 Future 风格的可等待对象：结果尚未完成时代表未来的值，完成后保存结果或异常。

```python
async def main():
    a = asyncio.create_task(fetch("A", 2))
    b = asyncio.create_task(fetch("B", 1))
    values = await asyncio.gather(a, b)
    print(values)  # 按传入顺序返回 ["A", "B"]

asyncio.run(main())
```

两个请求可以重叠等待，但 `gather` 的结果顺序仍与入参一致，并不取决于任务完成先后。

### gather、wait_for 与取消语义

- `asyncio.gather()`：并发等待多个可等待对象，并组织它们的结果。默认情况下一个任务抛异常后，其他任务**不一定自动取消**，需要理解其实际生命周期。
- `asyncio.wait_for(awaitable, timeout)`：达到超时时间通常会请求取消目标任务并抛出 `TimeoutError`；实际结束可能还涉及任务的取消清理时间。
- `asyncio.sleep()`：暂停当前 Task，让事件循环有机会推进其他任务，并不阻塞线程。
- `asyncio.wait()`：按就绪与未就绪集合组织任务，可以用来实现分批处理；与 `gather()` 的结果组织方式不同。

这也是为什么异步系统不能只讨论“快不快”，还要讨论**谁持有 Task、谁负责取消、异常由谁处理**。

### TaskGroup：结构化并发的生命周期边界

Python 3.11 引入的 `asyncio.TaskGroup` 让相关子任务属于同一个代码范围。一个子任务出现非取消类异常时，其他尚在运行的子任务通常会被取消；退出 `async with` 时等待任务收尾，并将相关异常组织成 `ExceptionGroup` 等形式。

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(fetch("A", 1))
        t2 = tg.create_task(fetch("B", 2))
    print(t1.result(), t2.result())

asyncio.run(main())
```

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Python TaskGroup 结构化并发图"><img src="/diagrams/python-structured-concurrency.svg" alt="TaskGroup 子任务生命周期、错误传播和取消机制" loading="lazy" /></button>
  <figcaption>结构化并发关注的是任务所属范围与异常处理，不只是让几个函数同时启动。</figcaption>
</figure>

## 四、GIL 限制了什么，又没有限制什么？

在传统带 GIL 的 CPython 构建中，全局解释器锁（Global Interpreter Lock）约束同一解释器内多个线程并行执行 Python 字节码的能力。因此，单纯创建更多线程，通常无法让**纯 Python CPU 密集型循环**利用更多 CPU Core 获得线性加速。

但 GIL 并不是“线程完全没有用”的意思。线程等待 Socket、磁盘 I/O 时可以让其他线程运行；某些原生扩展进行计算时也会释放 GIL，使多个线程有机会真正并行。PyTorch、NumPy 的部分操作还可能使用自身的原生线程池或设备端并行执行。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Python GIL 与进程并行图"><img src="/diagrams/python-gil-execution.svg" alt="传统带 GIL 的 CPython 线程与多进程并行的区别" loading="lazy" /></button>
  <figcaption>GIL 的作用范围与具体 Python 构建有关；多进程拥有独立解释器与内存空间。</figcaption>
</figure>

**版本边界：** Python 3.13 起已有可选的 free-threaded CPython 构建，它与传统带 GIL 的构建在并行能力与兼容性方面不同。因此，“所有 Python 线程都不可能并行运行 Python 代码”不再是无需限定的普遍结论。

### GIL 也不是对象与数据竞争的万能保护

在多线程程序中，即使使用传统带 GIL 的 CPython，也不应靠 GIL 替代应用级锁。跨越多个字节码或原生库调用的“读取→修改→写回”操作未必具有业务意义上的原子性；涉及共享资源时，仍需要合理的同步策略。

## 五、线程池、进程池与事件循环的协作

### 把阻塞 I/O 移交线程池

如果要从 `asyncio` 调用一个没有异步 API 的同步 I/O 函数，可考虑 `asyncio.to_thread()`：

```python
import asyncio
import time

def blocking_io():
    time.sleep(1)
    return "ok"

async def main():
    result = await asyncio.to_thread(blocking_io)
    print(result)

asyncio.run(main())
```

这让阻塞操作在其他线程上进行，避免直接堵塞事件循环；但 `to_thread` 并不自动解除传统 GIL 对纯 Python CPU 计算的限制。

### 多进程为什么能实现 CPU 并行？

`multiprocessing` 创建的不同进程通常各自拥有独立的 Python 解释器和虚拟地址空间。它们不共享同一把传统 GIL，因此适合将可分解的 CPU 工作分配给多个 Core。

```python
from concurrent.futures import ProcessPoolExecutor

def square(x):
    return x * x

def main():
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(square, range(8)))
    print(results)

if __name__ == "__main__":
    main()
```

但多进程会引入进程启动、数据序列化、IPC、对象复制与内存占用等成本；小任务可能因为这些开销反而更慢。不同操作系统和 Python 版本的默认进程启动方式也有区别，因此示例采用 `if __name__ == "__main__":` 保护入口。

| 维度 | 线程池 | 进程池 |
| --- | --- | --- |
| 地址空间 | 同一进程内共享 | 默认相互隔离 |
| 数据传递 | 可共享对象，需同步 | 通常经 IPC、序列化或显式共享内存 |
| 传统 CPython 纯 Python CPU 任务 | 易受 GIL 限制 | 可以在多核并行执行 |
| 启动与管理开销 | 通常较低 | 通常较高 |
| 适合的工作负载 | 阻塞 I/O、原生扩展等 | 可分割的 CPU 密集任务 |

## 六、CPython 为什么能执行我们写的 Python 文件？

理解 GIL 和执行过程时，有必要把 CPython 的几个阶段分清：

**源码 → Token / Parser → AST → Code Object / Bytecode → 解释器执行。**

```python
import dis

def add_one(x):
    return x + 1

dis.dis(add_one)
```

`dis` 可以查看字节码；它是理解执行模型的观察工具，而不是承诺不同 Python 版本有固定且完全相同的字节码序列。现代 CPython 包含解释器优化、专门化指令等机制，不能机械地把所有版本描述为“每个 Python 表达式只按一个完全静态的逐条执行路径运行”。

在常见的带 GIL 解释器中，线程执行 Python 代码时受 GIL 约束；执行某些阻塞系统调用或释放 GIL 的扩展逻辑时，情况则不同。这与上文线程池的适用边界互相对应。

## 七、实践中的选择准则

判断 Python 并发方案时，可以按以下顺序做决策：

1. **首先明确瓶颈。** 是 CPU 运算、网络/磁盘等待，还是 GPU / 数据库 / 远端服务的吞吐限制？
2. **识别阻塞的位置。** 异步函数内部是否使用同步 I/O 或长时间执行的 Python CPU 循环？
3. **确定状态共享需求。** Task 在同一个事件循环中共享线程上下文，线程共享进程地址空间，多进程则需要明确 IPC/共享内存边界。
4. **落实取消和异常处理。** 长时间运行的服务应明确超时、资源释放和任务归属，而不是让“后台 Task”无限存活。
5. **最后才进行性能测量。** 比较端到端吞吐和尾延迟，不能只看一个协程或一次进程启动耗时。

对分布式系统而言，Python `asyncio` 与多进程还可以组合使用：例如一个进程中的事件循环负责高并发元数据请求，不同进程承担相互隔离的执行工作，而真正的 GPU 计算交给专门的原生后端。

## 参考资料与验证范围

- [Python asyncio 官方文档](https://docs.python.org/3/library/asyncio.html)
- [Python asyncio Task 与 TaskGroup](https://docs.python.org/3/library/asyncio-task.html)
- [Python threading 与 GIL](https://docs.python.org/3/library/threading.html)
- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [Python free-threaded CPython](https://docs.python.org/3/howto/free-threading-python.html)
- [Python dis 模块](https://docs.python.org/3/library/dis.html)

文中的程序均为展示 API 行为的最小示例，不包含系统性 Benchmark。具体调度开销、并行收益与异常恢复路径需要使用目标 Python 版本和实际负载测量。
