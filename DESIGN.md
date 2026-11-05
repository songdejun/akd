# akd — Auto Kernel Dispatcher 设计文档

> 一句话定位：为手写 Triton kernel 变体提供「收集 + 实测 + 缓存分发」：
> `akd.cand(...).cand(...).dispatch()` 首调用实测各候选性能，之后按 key 直接命中，
> 用可移植的实测决策替代跨平台会失效的硬编码 dispatch 逻辑。kernel 保持裸 `@triton.jit`，akd 不包装任何东西。

## 修订记录

- 2026-11-05 v1：初版，guard / dict 参数形态 / kernel_params。
- 2026-11-05 v2：去掉 guard；更名 `akd.dispatch`；launch 表达式惰性化形态。
- 2026-11-05 v3：固定 dispatch 路径不设机制；key 来源改为外层 wrapper 参数。
- 2026-11-05 v4：key 改为 dispatch 时刻的 kernel 参数；site 身份加 `f_lasti`。
- 2026-11-06 v5（定稿，当前实现）：**彻底放弃惰性化包装**——不装饰、不代理、不拦截；
  API 改为**收集式链** `akd.cand(...).cand(...).dispatch(...)`；`ignore=[...]` 取代 `key=` 白名单；
  返回 `AKDRet` 结构体；删除自动包装（wrap.py）、`AKD_EAGER`、LSP 提示目标（triton 无 `py.typed`，
  类型链脆弱且用户裁决不值得）。

## 1. 问题定义

手写 Triton kernel 的现状流程：

1. 一个逻辑功能往往有多个 kernel 变体（不同向量化宽度 / 归约策略 / 是否用 `tl.dot` / 是否 split-K / persistent）；
2. 每个 kernel 内部还要 `@triton.autotune` 选 config；
3. wrapper 里用硬编码 `if/else` 按规模、dtype、架构分发。

痛点：这些 `if/else` 是在某个平台上手调出来的，换 GPU（H100 → A100 → 4090 → MIG 切片 → 其他厂商）之后：

- 阈值不再最优，甚至完全反向（SM 数、L2 大小、显存带宽比都变了）；
- 换平台就要重新人工调一遍分发逻辑，且这段逻辑没有测试保障。

现有机制为什么不够：

| 机制 | 解决的问题 | 缺口 |
|---|---|---|
| `@triton.autotune` | 单个 kernel 内的 config 选择 | 不解决变体之间的选择 |
| vLLM Kernel Dispatcher RFC | 跨平台**功能**兼容（静态注册 override） | 静态、人工维护注册表，不做性能寻优 |
| inductor `max-autotune` | 代码生成层面 benchmark 选实现 | 不面向手写 kernel 集合 |

akd 补的位置：**手写变体集合 + 实测性能 + 按 key 缓存决策**，一个轻量的、用户可控的运行时组件。

## 2. 分工原则

| 层 | 内容 | 写法 |
|---|---|---|
| 硬逻辑 + 固定 dispatch 路径 | 早退、shape/dtype 检查、「小输入必走某 kernel」这类固定路径 | 写在 dispatch 链**之前**的普通 Python：裸 Triton 调用然后 `return` |
| 性能软决策 | 要比较了才知道谁快的候选 | 交给 `akd.cand(...).dispatch(...)` |

原则：**akd 内部没有任何「固定/优先路径」机制**。有自信、或某输入必须走某 kernel——就手写直接 launch
（裸 `@triton.jit` 调用，语义与无 akd 时完全一致），这根本不会触发 dispatch；只有「要比较性能才能定」的候选才进 akd。

akd 仅保留一处与正确性相关的鲁棒性处理（不是分发机制）：候选在 bench 首轮 launch 失败 → 剔除并记录（§4.3）。

## 3. API 设计

### 3.1 候选链

```python
import akd

# kernel 是普通 @triton.jit（或 @triton.autotune 产物），akd 零接触
akd.cand(
    rowsum_oneshot, (rows,), x, out, n, BLOCK=triton.next_power_of_2(n),
  ).cand(
    rowsum_chunked, (rows,), x, out, n, BLOCK=1024,
  ).dispatch(ignore=["eps"])
```

- `akd.cand(fn, grid, /, *args, **kwargs)`：开始链，展开形态与 Triton launch 语句逐字对应（把 `[grid]` 挪进 cand）；
  `grid` 钉成位置限定参数，kernel 自带名为 `grid` 的参数时互不冲突。
- `akd.cands(*specs)`：分组形态，spec 为 `(fn, grid)` / `(fn, grid, args)` / `(fn, grid, args, kwargs)`
  （args 接受 tuple/list，kwargs 必须是 dict），用于参数已打包的转发 wrapper 和表驱动的动态构建；
  `.cands` 同时是 builder 方法，可与 `.cand` 混链。
- builder **原地变异**（`.cand`/`.cands` 返回 self）——中间对象不存在，泄漏警告判定干净：
  被回收却从未 `dispatch()` 的链会发 `AkdWarning`（防止「写了候选却什么都没跑」的静默 bug）。
- 链在 **wrapper 内每次调用重建**（候选记录的是当次 tensor 引用；重建就是一次列表 append，微秒级）。
- 设计上**不设**「重复 dispatch 守卫」：同一 builder 二次 dispatch 是用户自己的事（2026-11-06 拍板）。

### 3.2 `dispatch(...)` 语义与选项

**首次调用** = 编译各候选（warmup，不计时，失败者剔除）→ 乱序轮次 bench（§7）→ 胜者向真实 buffer
重新 launch → 缓存决策；**后续调用** = 按 key 命中（§5），只 replay 胜者。

| 选项 | 说明 |
|---|---|
| `ignore=["eps", ...]` | 从决策 key 中剔除的**kernel 参数名**（按 arg_names 对各候选匹配）。默认全量参数入 key；ignore 的失败模式是丢性能、不丢正确性（§5） |
| `outputs=["out"]` | 声明被**原地累积**修改的输出参数名；bench 时对这些参数做 clone（§4.1） |
| `verify` | 首次 bench 时对候选输出 allclose 交叉验证（默认关，§4.2） |
| `min_improvement` | 择优阈值，默认 0.03：声明顺序中第一个「不比最优慢 3%」的候选胜出 |
| `site` | 分发点名字；默认取调用方 frame 的 `co_qualname + f_lasti`（§5.2），一般不用填 |
| `skip_bench` | `True` 时永远选第一个候选（测试与调试用） |
| `bench_budget_ms` / `rounds` / `flush_l2` | 计时控制（§7），多数场景留在配置文件 |

**返回 `AKDRet`**：

| 字段 | 含义 |
|---|---|
| `winner` | 胜者声明索引（int，无歧义） |
| `name` / `kernel` | 胜者 kernel 名 / fn 对象（同 kernel 双 BLOCK 会撞名，与 winner 并存） |
| `return_value` | **胜者最终那次 launch 的返回值**（Triton 通常返回编译句柄，无则 None）；bench 中间 launch 的返回值不暴露 |
| `benched` | 本次是否真的实测（缓存命中 / 单候选 / skip_bench 均为 False） |
| `timings` / `pruned` | 各存活候选 median `[(索引, ms)]` / 被剔除候选 `[(索引, 原因)]` |

### 3.3 显式 warmup（可选）

首调用 = benchmark，有一次性的秒级开销（多候选 JIT 编译 + 计时）。延迟敏感场景启动期预热：
`akd.warmup(lambda: rowsum(x_bench))`（或直接正常调用一次）。配合持久化缓存（§5.3），每台机器只付一次。

### 3.4 观测与调试

- `AKD_DEBUG=1` / 配置 `log.debug`：bench 过程、key 构成、被跳过的参数、命中/未命中打到 stderr；
- `akd.report()`：所有 site 的决策表；`akd.explain()`：逐候选耗时、剔除原因、编译开销；
- `akd.reset(site=None)`：清内存决策缓存（强制重 bench）。

### 3.5 配置文件

查找顺序（先找到先用）：项目 `akd.toml`（或 `pyproject.toml` 的 `[tool.akd]`）→
`~/.config/akd/config.toml` → 内置默认。环境变量最高：`AKD_DEBUG` / `AKD_CACHE_DIR`。

```toml
[dispatch]
min_improvement = 0.03

[bench]
budget_ms = 30.0
rounds = 3
flush_l2 = true

[store]
enabled = true
dir = "~/.cache/akd"

[log]
debug = false
```

## 4. 正确性契约

benchmark 的本质是**真的把这些 kernel launch 出去**，所以有一串必须显式处理的正确性问题。

### 4.1 bench 会污染输出 buffer

候选们往同一个 `out` 写，bench 顺序跑完后 `out` 里是最后一个候选的（可能错误的）结果。

- **bench 结束后永远把胜者重新 launch 一次**到真实 buffer。代价是首次调用多一次 launch。
- 隐含契约：候选对输出参数必须是**覆盖语义**（幂等）；`out += ...` 这类**累积型** kernel、
  以及任何被原地修改的参数，必须通过 `outputs=` 声明，akd 对声明的输出做 clone，
  bench 各候选写各自的 clone。契约文案：**除 `outputs` 声明的参数外，所有 tensor 参数视为只读**。

### 4.2 数值不等价

不同变体的浮点归约顺序不同，结果不 bit 级一致是正常的。契约：**语义等价由用户保证**。
`verify=True` 在首次 bench 时把各候选输出 clone 出来交叉比对（并集-同意分组），
防止「错误的变体因为最快而胜出」——benchmark 择优会放大静默 bug。两候选意见相分歧时保留更快者：
akd 没有 oracle，只有一致性。默认关（有开销），开发期强烈建议开。

### 4.3 候选 launch 失败（bench 鲁棒性，非分发机制）

warmup/首轮 launch 抛错（如某卡编译不了某变体）→ 捕获、剔除、记录、继续；全部候选失败则汇总报错。
剔除后就少了候选，决策在「活下来的候选」里做。固定路径约束不靠它表达（§2）。

### 4.4 其他陷阱

- **CUDA graph capture 中禁止 bench**：capture 中直接用缓存决策；无缓存则取第一个候选且不缓存该决策。
- **多线程**：bench 持进程级锁（并发 bench 互相污染计时）。同 key 并发调用等锁后命中缓存。
- **多设备**：key 含设备指纹（§5），不同卡决策天然分开；launch 在参数 tensor 所在设备执行，
  所有 tensor 必须同设备否则报错；CPU tensor 需 `TRITON_INTERPRET=1`（解释器模式下无 stub bench 时自动选第一个候选）。
- **单候选优化**：只剩 1 个候选时跳过 bench 直接 launch。
- **`torch.compile`**：dispatch 调用点是 graph break；key 不依赖栈帧，仅 site 命名退化。v1 接受。

## 5. key 设计：dispatch 时刻的 kernel 参数

### 5.1 triton 实际怎么做（v3.8 源码核对）

**autotuner 的决策 key**（`triton/runtime/autotuner.py:221-227`）：

```python
key = [_args[name] for name in self.keys if name in _args]   # key= 命名的 kernel 参数，按值
for _, arg in _args.items():
    if hasattr(arg, "dtype"):
        key.append(str(arg.dtype))                            # 所有带 dtype 的参数自动追加 str(dtype)
key = tuple(key)
```

即：**命名参数按值 + 所有 tensor 参数的 dtype 字符串**。不含 shape、对齐、stride、设备。

**jit 层的二进制特化边界**（`triton/runtime/jit.py`，`do_not_specialize` 可关）：
非 constexpr 参数按「int ==1 / 整除 16、指针 data_ptr 整除 16（对齐）」特化编译——
不同对齐/整除性输入下是**不同的二进制**，性能特征可以完全不同。

**autotuner 的磁盘缓存**（`check_disk_cache`，`cache_results` 默认 False）：`sha256(triton版本 + fn.cache_key(源码) + str(tuning_key))`。

### 5.2 akd 的 key

原则与 triton 一致：**key 状态是调用时从所有实参计算出来的**。候选链在 dispatch 语句执行那一刻
记录参数，所以 wrapper 内此前的输入替换、新建 tensor、切片 view 捕捉到的都是最新值；
`out`、scratch 这类 wrapper 内创建的 tensor 本来就是 kernel 参数，自动进 key。

```python
dkey = (
    site(co_qualname + f_lasti),                       # 调用点身份：同 wrapper 多个 dispatch 点不撞名
    候选集身份(各候选源码 hash + grid callable hash),    # 改代码/改候选构成 → 决策自动失效
    设备指纹(name, capability, sm_count, 总显存),        # 进程级按设备缓存
    Σ 按候选顺序 × 每个参数 (按 arg_names 命名):
        tensor → (str(dtype), shape, stride, data_ptr % 16)   # dtype→str 结果缓存
        标量(int/float/bool/str/None) → 按值
        其他对象 → 跳过，AKD_DEBUG 下列出
    减去 ignore=[...] 命中的参数项,
)
```

- **默认全量入 key，宁细勿粗**：key ⊇ 二进制特化边界（标量按值覆盖 int ==1/%16；ptr%16 覆盖指针特化），
  决策缓存永远不和 triton 二进制缓存错位；hit-rate 靠 `ignore=` 收窄和 M4 分桶。
- **`ignore=` 是黑名单**（对齐 triton `do_not_specialize` 的精神，层次不同）：被忽略参数的变化不再触发
  重 bench。失败模式：该参数其实影响性能 → 决策偏旧，**丢性能不丢正确性**。
- **指纹只含对齐不含指针值**：`torch.empty_like` 每次新分配不影响命中。
- **候选行 constexpr / launch 选项应静态**：动态值（`BLOCK=min(n, 8192)`）让参数指纹逐调用变化
  → 每次 miss 重 bench（AKD_DEBUG 可见）。要动态 tile 就提为多行静态候选。
- grid tuple 的值不入候选集身份（它们由 keyed 参数派生）；grid **callable** 的源码 hash 入身份。

### 5.3 持久化

`$AKD_CACHE_DIR` / 配置 `store.dir`（默认 `~/.cache/akd`），条目 = sha256(site + 候选集身份 + key)，
值 = {胜者, 各候选耗时, bench 时间戳, akd/torch/triton 版本}；版本不匹配即失效。
内核编译产物仍归 triton 自己的缓存管，akd 只存「选谁」。

## 6. 核心数据结构与热路径

- `Launch`：一条记录 `(fn, grid, args, kwargs)`；`CandidateChain`：记录列表 + 已 dispatch 标记。
- `Problem`（key 事实来源）：site + 候选集身份 + 设备指纹 + 参数指纹（§5.2）。
- 决策状态机（每个 dkey 一个条目）：

```
UNSEEN ──候选数==1──► 直接launch并缓存
   │      └─候选数>1──► BENCHING(持锁) ──► CACHED(winner, timings, confidence)
   └──持久化缓存命中──► CACHED(跳过bench)
```

**热路径**（缓存命中）= 参数指纹 → key 构造 → dict 命中 → replay 胜者。全部微秒级（远小于一次
kernel launch 的 ~5–10µs+），无 GPU 同步：device 指纹按设备缓存、dtype→str 缓存、元数据按 `id(fn)`
弱缓存（`registry.meta_for`）、同参数多候选共享指纹。

## 7. benchmark 方法论

目标不是绝对耗时精确，而是**排名稳定**。

1. **warmup（不计时）**：每个候选 launch 一次（失败即剔除并记录，编译耗时单独记录），再各补两次热身
   ——顺带触发 Triton JIT 编译和内层 `@triton.autotune` 的首轮选型。
2. **计时结构：R 轮 × 每轮乱序测全部候选**。每轮前 shuffle 候选顺序，候选间用 CUDA event 计时。
   交替而非顺序的原因：DVFS/温漂会系统性偏袒后测者；乱序交替把漂移变成可均摊的噪声。
3. **聚合**：每候选取各轮 **median**（不是 min——min 对偶发短路过于乐观且不稳定）。
4. **L2 flush**：默认开（对齐 `triton.testing.do_bench` 惯例），`flush_l2=False` 可关。
5. **自适应重复次数**：pilot 计时估计后按 `bench_budget_ms`（默认每候选 ~30ms）自适应 rep 数（1–256）；
   单次 launch 太慢遇下限即止。
6. **置信度**：任一候选轮间 CV > 15% → 决策标记 `unstable`，report 显示。
7. **择优**：`min_improvement` 边际，声明顺序即偏好顺序（同等速度取更简单变体）。

## 8. 与 `@triton.autotune` 的关系

- **组合（当前）**：内层 autotune 的 kernel 直接作为候选（`fn[grid](...)` 协议天然兼容）。
  warmup 触发它，计时测的是稳态。缺点：首调用成本是两层 benchmark 的乘积。
- **收编（后续）**：把 `(变体 × config)` 展平成扁平候选集联合搜索——`cands()` 下展开是平凡的
  （同一 kernel 不同 `BLOCK=` 各一条 spec）。两层贪心会错过「变体 B 用非其默认最优 config 时全局更快」
  的组合，联合搜索严格更优。

## 9. 路线图

| 里程碑 | 内容 | 验收 |
|---|---|---|
| ~~M0–M2~~ | 链式 API、key、bench 方法论、持久化（已随 v5 实现） | 55 测试全过 + GPU 端到端 |
| M3 | unstable 触发重 bench、`akd.warmup` 文档化实践、LRU、跨 site bench 去重（相同 (候选集身份， key) 共享决策） | 噪声下决策不抖动 |
| M4 | key 分桶/近邻复用（serving 形状漂移）、后台 re-bench | 连续 shape 不再逐个 bench |
| M5 | 多进程共享 store（torchrun 同 FS file lock）、收编 autotune 展平联合搜索、非 Triton 候选（任意 `fn[grid](...)` 协议对象） | 多卡训练吞吐无回退 |

## 10. 代码结构

```
src/akd/
  chain.py      # CandidateChain / Launch / AKDRet / cand / cands / 泄漏警告
  api.py        # dispatch 主流程：key → 缓存 → capture/单候选守卫 → bench → 重放
  registry.py   # 源码 hash、grid callable hash、meta_for 弱缓存
  problem.py    # site 身份 / 候选集身份 / 参数指纹 / ignore 过滤 / 设备指纹
  selector.py   # 决策缓存与择优策略（无 torch/triton 依赖）
  bench.py      # 计时引擎：warmup 剔除、pilot、乱序轮次、median、L2 flush
  contract.py   # 正确性检查：设备一致性、capture 检测、outputs clone、verify
  store.py      # 持久化缓存（json, AKD_CACHE_DIR）
  config.py     # akd.toml / pyproject [tool.akd] / ~/.config/akd / 环境变量
  report.py     # report / explain / reset
  errors.py
tests/          # stub + fake bench 的单元测试；TRITON_INTERPRET 子进程 e2e；GPU 集成
examples/rowsum.py
```

关键实现约束：**akd 核心不 import triton**（bench 后端可注入，CI 无 GPU 可测 selector 全部逻辑）；
triton 只出现在真实 kernel 与集成测试里。akd 对 kernel 文件零侵入——`import akd` 只出现在 wrapper 层。

## 11. 测试策略

- **单元**（无 GPU，CI 主力）：key 指纹/ignore 过滤、状态机、择优边际、spec 归一化、泄漏警告、
  持久化读写、并发锁——stub kernel + 注入式 fake bench（`akd.api._bench` 测试缝）；
- **key 行为矩阵**：dispatch 前替换输入 / 新建 out / 切片 view → 决策键正确变化；新分配同指纹 → 命中；
  多调用点 `f_lasti` 区分；
- **功能集成**：`TRITON_INTERPRET=1` 子进程跑真实 kernel 端到端（决策缓存、胜者重放、固定路径）；
- **性能集成**（`@skipif(no_gpu)`）：rmsnorm/rowsum 双变体——决策随规模翻转、缓存命中、固定路径绕过 dispatch、report 冒烟；
- **黄金文件**：report/explain 输出格式（待补）。

## 12. 相关工作与定位

- [vLLM RFC: Triton Kernel Dispatcher](https://github.com/vllm-project/vllm/issues/45133)：
  痛点同源，方案是**静态注册**，解决异构平台**功能**兼容。akd 是**动态实测**，互补。
- `@triton.autotune`：变体内 config 选择；akd 在其上一层（§8），`ignore=` 的精神沿其 `do_not_specialize` / `key=`。
- inductor `max-autotune`：生成式实现的 benchmark 择优，不针对手写变体集合。
- [FlagGems](https://dev.to/tamizuddin/triton-at-the-core-building-portable-llm-operator-libraries-with-flaggems-and-benchmarking-against-4k2p)：
  库级跨平台 Triton 算子集；akd 可视为把这类库的 dispatch 经验自动化成通用件。

## 13. 开放问题（后续迭代）

1. **verify 默认值**：当前默认关 + 文档建议开发期开。
2. **`min_improvement` 默认 3%**：需更多 GPU 实测噪声数据（笔记本 GPU 上 unstable 标记频繁触发）。
3. **AKDRet 字段扩展**：compile 开销、site、决策时间戳是否入结构体。
4. **M4 分桶**的 key 语义（黑名单 ignore 与桶化的交互）。
5. **非 Triton 候选**（任意 `fn[grid](...)` 协议对象，如 `torch.matmul` 包装）——M5。
