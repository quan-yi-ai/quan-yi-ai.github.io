---
title: Python 函数对象与资源协议：闭包、装饰器、生成器和 with
description: 从函数是一等对象出发，理解装饰器与闭包的环境绑定、生成器的暂停恢复，以及 with/async with 的资源生命周期。
sidebar: false
aside: right
outline: [2, 2]
pageClass: asb-reading
editLink: false
lastUpdated: false
---

<div class="asb-story-header">
  <div class="asb-story-breadcrumb"><a href="/posts/">← 返回文章列表</a></div>
  <h1 id="python-function-protocols" tabindex="-1">Python 函数对象与资源协议：闭包、装饰器、生成器和 with</h1>
  <p class="asb-story-subtitle">从“函数也是对象”到暂停、恢复和资源收尾</p>
  <p class="asb-story-deck">装饰器、闭包、生成器与上下文管理器看起来属于不同语法，但它们都围绕一个核心展开：Python 如何保存环境、延后执行，以及在明确的边界内管理状态与资源。</p>
</div>

要理解 Python 的装饰器和生成器，不必先背一长串 `@`、`yield` 和 `with` 的用法。先回答一个更根本的问题：**一个函数在执行前、暂停中以及调用结束后，哪些状态仍然存活？**

## 一、函数是一等对象，装饰器才有意义

在 Python 中，函数可以被赋值给变量、传给其他函数，也可以作为返回值。

```python
def greet(name):
    return f"Hello, {name}"

alias = greet
print(alias("Python"))
```

装饰器利用了这一点：**接收一个函数对象，返回一个可调用对象，并把原来的函数名重新绑定到返回值**。

```python
from functools import wraps
import time

def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            print(f"{fn.__name__}: {time.perf_counter()-start:.4f}s")
    return wrapper

@timer
def work(x):
    return x * x
```

`@timer` 可以理解为定义时执行 `work = timer(work)`。装饰器本身不是在每次调用 `work()` 时才被构造；执行的是返回后的 `wrapper`。这里使用 `functools.wraps` 保留常用的函数元数据，但它也不会自动使同步装饰器适用于异步函数。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：装饰器定义时包装与闭包持有原函数"><img width="960" height="585" src="/diagrams/python-decorator-closure.svg" alt="装饰器定义时包装与闭包持有原函数" loading="lazy" /></button>
  <figcaption>定义时建立 wrapper 并重新绑定函数名；调用时才执行包装逻辑和原函数。</figcaption>
</figure>

### 带参数装饰器为什么多一层？

```python
def retry(times=3):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise
        return wrapper
    return decorate

@retry(times=3)
def request():
    ...
```

对应关系是 `request = retry(times=3)(request)`。这里省略了幂等性检查和退避策略，所以只是展示语言机制，不是可以直接用于生产网络请求的重试组件。

## 二、闭包：为什么外部函数返回后变量还能存在？

```python
def make_counter():
    count = 0

    def inc():
        nonlocal count
        count += 1
        return count

    return inc

counter = make_counter()
print(counter())  # 1
print(counter())  # 2
```

`make_counter()` 已经返回，但 `inc` 仍引用了外部作用域的 `count`。闭包就是**函数对象与其词法环境绑定**的机制；并非所有外层局部变量都会永远保留，关键在于被内部函数捕获及其引用生命周期。

`nonlocal` 的作用是在嵌套作用域中修改已存在的外层绑定。如果只是读取自由变量，则不一定需要 `nonlocal`。

闭包可以表达小型状态机，但它并不自动保证线程安全；多个线程同时修改捕获的可变状态仍需考虑同步。

## 三、生成器：函数如何暂停并恢复？

普通函数调用会沿执行路径运行直到返回或抛出异常。生成器函数在函数体中使用 `yield` 后，**调用时通常先创建生成器对象，而不是立刻执行函数体**。

```python
def numbers():
    for i in range(3):
        print("produce", i)
        yield i

g = numbers()
print(next(g))  # produce 0, 然后得到 0
print(next(g))  # produce 1, 然后得到 1
```

每次 `next(g)` 推进到下一个 `yield`，执行帧中的局部变量与控制位置被保留，以便下次继续。生成器是迭代器，但不是所有迭代器都由生成器函数创建。

如果生成器执行到 `return`，下一次请求值会触发 `StopIteration`；`return value` 的值可通过相关机制获得，但它不是一次普通的 `yield value`。

### Generator 与 Coroutine 不等价

生成器的 `yield` 支持向调用方逐步产出数据；`async def` 协程的 `await` 则通过 Awaitable 与事件循环协作。在机制历史上二者有关联，但现代 Python 的业务语义不能直接混用。

对于数据处理中的流式迭代，生成器常用于避免一次性构造完整列表；但是否节约内存取决于上游输入、缓存和下游是否立即消费。

<figure class="asb-figure asb-figure--cover">
  <button class="asb-zoom-trigger" type="button" aria-label="放大 Python 函数与资源协议图"><img width="960" height="550" src="/diagrams/python-runtime-protocols.svg" alt="闭包、生成器、上下文管理器与 asyncio 的不同状态边界" loading="lazy" /></button>
  <figcaption>闭包保存词法环境，生成器保存暂停位置，上下文管理器定义资源退出协议。</figcaption>
</figure>

## 四、with：资源退出如何从 try/finally 变成协议

打开文件时，需要确保即使中途发生异常也能执行关闭逻辑：

```python
f = open("example.txt", "r", encoding="utf-8")
try:
    content = f.read()
finally:
    f.close()
```

`with` 提供更紧凑的表达：

```python
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

上下文管理器遵循 `__enter__()`、`__exit__(exc_type, exc, traceback)` 协议。进入作用域时建立或获取资源，退出时执行清理；`__exit__` 的返回值还关系到异常是否被抑制。

```python
class Managed:
    def __enter__(self):
        print("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        print("exit")
        return False  # 不抑制异常

with Managed():
    print("using resource")
```

这里的“退出”不意味着一定在函数调用之后立即释放所有对象，真正发生的操作由上下文管理器的实现决定；例如文件关闭与数据库事务提交/回滚是不同语义。


<figure class="asb-figure">
  <button class="asb-zoom-trigger" type="button" aria-label="放大：with 进入使用退出与异常处理"><img width="960" height="590" src="/diagrams/python-context-lifecycle.svg" alt="with 进入使用退出与异常处理" loading="lazy" /></button>
  <figcaption>__enter__ 成功后进入代码块，__exit__ 处理正常收尾或异常；返回真值可抑制代码块异常。</figcaption>
</figure>

## 五、async with：清理自身也可能需要等待

有些资源的获取和释放需要执行 I/O，例如建立网络连接、提交异步数据库事务或关闭远程 Session。此时使用实现了 `__aenter__`、`__aexit__` 的异步上下文管理器：

```python
class AsyncResource:
    async def __aenter__(self):
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await asyncio.sleep(0)
        return False
```

与普通 `with` 不同，异步上下文管理器的进入与退出都可以挂起。但这也引入取消处理问题：当任务被取消时，释放资源的异步逻辑可能需要额外保护，以免留下半关闭连接。

## 六、总结：语法背后是不同的生命周期

| 机制 | 保存了什么 | 何时恢复/退出 | 常见用途 |
| --- | --- | --- | --- |
| Decorator | 返回后的 Wrapper 与闭包引用 | 调用被装饰函数时 | 日志、计时、权限检查 |
| Closure | 被引用的词法环境 | 再次调用内部函数 | 配置化函数、小型状态 |
| Generator | 暂停的执行帧与局部变量 | `next()` / `send()` | 流式数据与惰性迭代 |
| `with` | 资源作用域 | 作用域结束或异常 | 文件、锁、事务 |
| `async with` | 可异步进入/退出的资源作用域 | `await` 式收尾 | 网络、异步连接 |

这些语法不是为了让代码看起来更 Pythonic，而是帮助程序明确**状态由谁持有、操作什么时候执行，以及异常发生时谁负责收尾**。

## 参考资料

[Python Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) · [functools.wraps](https://docs.python.org/3/library/functools.html#functools.wraps) · [Python Data Model](https://docs.python.org/3/reference/datamodel.html) · [Iterator / Generator](https://docs.python.org/3/howto/functional.html#generators) · [contextlib](https://docs.python.org/3/library/contextlib.html)
