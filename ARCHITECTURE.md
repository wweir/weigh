# 架构

## 1. 这个项目是什么

`weigh` 从生成式模型中读出一个决策，方式是询问*具名答案*的概率，而不是问模型"你想说什么"。

本服务**不是生成器**。对一次请求，它做四件事：

1. 校验单字段的决策 schema，并把其中的取值绑定到答案槽位；
2. 为 criterion/evidence 这一对渲染 `direct-options-v1` 提示词；
3. 向后端索取每个已声明槽位在那唯一评分位置上的精确 logprob；
4. 把 argmax 作为一次 chat-completion 返回，其 content 是本地构造的，而不是模型生成的。

以下后果属于契约的一部分，而不是实现细节：

- 模型从不输出 JSON。`choices[0].message.content` 由 schema 枚举值构造，因此"响应符合
  schema"由构造保证，不需要 guided decoding。
- 采样参数被接受、被忽略，并在 `semif.ignored_parameters` 中回列。
- 每个响应都携带足够的 provenance，足以说明是哪个提示词、哪些槽位、哪条读出路和哪套服务
  配置产生了这个数字。

## 2. 系统边界

```
 client ──POST /v1/chat/completions──▶ weighd ──POST /tokenize──────────▶ backend (vLLM/SGLang)
                                        │      ──POST /v1/completions──▶
                                        │      ──POST /generate────────▶
                                        └──POST /v1/chat/completions──▶ (media only)
```

- **本进程拥有**：SemIf 的契约（schema 的各类拒绝、提示词内容、criterion/evidence 的折叠、
  响应面）、答案槽位模型，以及 provenance。
- **后端拥有**：聊天模板的分词、模型、那些数字。它是 token id 的唯一来源（见决策 D2）。
- **本进程不拥有**：采样、批调度、分词、聊天模板渲染。它是一个同步的、纯 HTTP 的读取器，
  由网关置于前面。

## 3. 分层

| 层 | Go 包 | 职责 |
|---|---|---|
| 库 API | 根包 `weigh` | 对下面这些包的策展式再导出：`New`/`Client.Decide`、组装原语、枚举与 provenance 类型 |
| 入口 | `cmd/weighd` | argv、配置加载、logger 初始化、退出码 |
| HTTP 契约 | `internal/serve` | 路由、请求校验、错误信封、SSE、`/health`、优雅退出、指标 |
| 决策契约 | `internal/schema`、`internal/prompt`、`internal/template`、`internal/slots` | 单字段单类型的拒绝集、`{evidence,criterion,options}` 载荷、7 族聊天模板注册表、`A..P` 答案槽位 |
| 读出 | `internal/readout`、`internal/tokenize`、`internal/media` | 后端探测、exact-slot 请求、softmax、provenance、图像上限 |
| 共享值 | `internal/version` | 由 ldflags 注入的版本与构建日期 |

依赖方向是单向的：根包 `weigh` 与 `serve` → 决策 + 读出 → tokenize/media。读出层永远看不
到 "criterion"、"evidence" 字符串或 `direct-options-v1` 提示词；决策层永远看不到 HTTP、
tokenizer 或 softmax。这个切分是本实现所取代的那个两 crate 边界的延续，以包的形式保留下来，
也正是它让读出层可以被独立测试。

根包 `weigh` 把这条边界重新变成一个**可被第三方 import 的库**：`weigh.Client.Decide` 接受
调用方渲染好的成品 prompt 字符串，库自己不渲染聊天模板；`weigh` 里的组装原语让想复现服务端
`direct-options-v1` 契约的调用方可以逐字节复现它，而不是重新实现。它是别名式的再导出而非
实现，所以同一份代码同时是库和服务的引擎。`Config.Source` 在库模式下可留空：那时不解析模板、
不运行 checkpoint 守卫；显式命名的族仍被记录并可渲染，`TemplateAuto` 因无处可检测而被拒绝，
未指定的模板则让 `Metadata.PromptTemplate` 如实留空。HTTP 面仍是 SemIf 的
`direct-options-v1` 契约，外部组装是库模式的能力，不是新的入口。

## 4. 关键数据流

```
request JSON
  │ locate the decision schema (response_format.json_schema.schema | schema | guided_json)
  ▼
DecisionSchema{field, values}          values[i] is answer slot LETTERS[i]
  │ render: system = DIRECT_SYSTEM, user = {"evidence","criterion","options":[{"letter","description"}]}
  ▼
rendered prompt string
  │ POST {backend}/tokenize  (add_special_tokens=false)          ← replaces the local tokenizer
  ├─ ids      = tokenize(prompt)
  └─ slots[i] = last id of tokenize(prompt + LETTERS[i])         ← the answer-boundary proof
  ▼
scoring request: prompt pinned as token ids, logprob_token_ids = slots, max_tokens = 1
  ▼
option_logprobs[slot] → softmax over the declared subset → probabilities[i]
  ▼
choice_index = argmax(probabilities) ; content = {"<field>": values[choice_index]}
  ▼
chat.completion with choices[].semif + top-level semif provenance
```

## 5. 设计决策

**D1 —— 一个字段、一种类型、一次 softmax。** 读出器只在一个有限的槽位集上计算一个分布。多
字段 schema、联合类型，或不生效的关键字（`pattern`、`minimum`）都在门口被拒。见
`docs/decision-contract.md`。

**D2 —— token id 来自后端，永不来自本地实现。** 客户端把提示词**以 token id 形式**提交，并
**按 token id** 钉住答案槽位。本地 tokenizer 会是第二个实现，它可能与真正执行评分的引擎不
一致；而这种不一致是静默的（得到看似合理但错误的概率）。因此 top-N 路由也按 token id 匹配，
而不是把槽位 token 在本地解码后去查 top-N 的键；以解码文本作答的构建会被拒绝，而不是靠猜去
匹配。后端被要求通过 `POST /tokenize`
分词，而该端点的缺失是启动失败，不是回退。见 `docs/tokenize-contract.md`。

**D3 —— 保留答案边界证明。** 对每个字母 `L`，客户端要求
`tokenize(prompt + L) == ids + [one_token]`，并要求得到的 token 两两不同。正是这一点让"模型
被要求只输出一个 token"成为一件被检查过的事实。该证明由后端自己的 tokenizer 执行，因此它证
明的是与引擎的一致，而不是与某份本地拷贝的一致。

**D4 —— 聊天模板是注册表，不是 Jinja 引擎。** 被服务的形状始终是一个朴素的 system 轮加一
个朴素的 user 轮，因此每个族可达的子图都只是一个固定前缀加一个 assistant 轮的开头。某个
`tokenizer_config.json` 的 `chat_template` 若不匹配任何已知族，就是错误，绝不是静默默认。

**D5 —— 宁可大声拒绝，也不降级。** exact-slot 响应里缺少某个槽位是错误；某端点在被探测证明
会回显钉住键之后又不再回显，是错误；无法识别的模板、tokenizer 与 checkpoint 不匹配、在媒体
未被探测的情况下送来媒体载荷——全都是带可区分错误码的拒绝。这里不存在"改了被测量的东西却不
记录"的回退。

**D6 —— provenance 不是可选项。** `prompt_sha256` / `input_ids_sha256` / `readout` /
`serving_config` / `fallback_used` / `server_tokenized` 都被记录，因为两种不同的读出结果永
远不能合池。见 `docs/provenance.md`。

## 6. 状态

这是代码树中唯一的实现。每一层都已实现并测试——配置、决策核心、tokenize 边界证明、文本与媒
体读出、HTTP 面、优雅退出、指标——其 hermetic 测试套件见 `docs/testing.md`。

它取代了一套更早的、两个 crate 的 Rust 实现，那套实现已被**移除**。那套实现的两 crate 边界
（库 + 服务）已作为 Go 的根包 `weigh` 恢复：它是别名式再导出，不是第二份实现，所以同一个
`readout` 同时服务库调用方与 HTTP 面。它原本只作为本次重写的
oracle 保留，所以当本实现通过自身的测试后，它就不再具有权威性；它仍可在 git 历史中阅读。移
除它使验证计划的一半作废——即与 Rust 二进制做行为 diff 的那一半。与它刻意不同的几处行为，其
规则本身就写在权威文档里：`docs/media-contract.md` §5 的 chat 路由响应形状，以及
`docs/tokenize-contract.md` §5 的 `server_tokenized` 语义；不要照着 git 历史把它们"改回来"。

本实现把那个无法忠实移植的依赖（`tokenizers`）替换为一次后端调用，并去掉了 Python 字节对齐
层（`pyjson`、`ryu`、CPython 浮点 repr）——那是 Python 客户端的需求，不是正确性的需求。所
有属于正确性不变量的一切都被保留。

**未闭合的缺口：本仓库没有 CI，也没有发布流水线。** 被删除的那些 workflow 构建的是 Rust
crate，对 Go 一无所知，所以它们是删掉而不是移植；`make lint test package` 就是今天的全部门
禁。编写 Go 流水线是另一件独立工作，不属于本次重写。

**库 API 有三处未决取舍**（在此记录，因为 `docs/plans/` 不是权威来源）：`readout.Prepare`/`Fetch`
没有拆开（拆开需把选中的端点钉进 `Prepared`，否则 ids 与分数可能来自不同引擎）；根包以别名
再导出 `internal/*`，公共类型若要独立演进须把那些包提升出 `internal/`；库模式的拒绝信息仍
带 CLI 口吻，改它要连同 `docs/decision-contract.md` §4 一起改。

## 7. 文档索引

设计文档（位于 `docs/`）：

- `docs/tokenize-contract.md` —— 后端必须暴露什么、边界证明，以及去掉本地 tokenizer 带来的
  provenance 后果。
- `docs/media-contract.md` —— 多模态请求：接受的形状、拒绝集、两条媒体路由，以及一张图像的
  provenance。
- `docs/decision-contract.md` —— 路由、守卫顺序、校验顺序、错误信封与请求体上限。
- `docs/provenance.md` —— 每个被记录的字段及其存在的原因。
- `docs/testing.md` —— hermetic 套件、假后端，以及实弹检查额外覆盖了什么。
