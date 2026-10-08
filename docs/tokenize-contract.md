# Tokenize 契约

状态：对 Go 实现具有权威性。取代参考实现中基于本地 `tokenizers` 的方案（该实现已移除，见
`ARCHITECTURE.md` §6）。

## 1. 为什么由后端分词

读出把提示词**以 token id 形式**提交，并**按 token id** 指名答案槽位（`logprob_token_ids` /
`token_ids_logprob`）。这些 id 必须就是评分引擎所用的那些 id。第二份本地实现的 tokenizer 可能
与引擎不一致，而且这种不一致是静默的：请求被接受，返回看似合理但错误的概率。

因此 token id 从后端取得。这去掉了 Go 移植无法忠实复现的唯一依赖（`tokenizers` Rust crate，
它正是 Python 包所封装的那份代码），代价是每个不同的提示词多一次 HTTP 往返。

## 2. 必需的端点

| 后端 | 端点 | 请求 | 响应 |
|---|---|---|---|
| vLLM | `POST /tokenize` | `{"model": <str>, "prompt": <str>, "add_special_tokens": false}` | `{"tokens": [int], "count": int, "max_model_len": int}` |
| SGLang | `POST /v1/tokenize` | `{"prompt": <str>, "add_special_tokens": false}` | `{"tokens": [int], "count": int, "max_model_len": int}` |

两者都已对照项目自身源码核实：vLLM
`vllm/entrypoints/serve/tokenize/protocol.py`（`TokenizeCompletionRequest`），SGLang
`python/sglang/srt/entrypoints/openai/serving_tokenize.py`（合入于 PR #9545）。

承重的几条注意事项：

- `add_special_tokens` **必须发 false**。vLLM 在 completion 模式下的默认值是 `true`，那会前
  置一个 BOS，使评分请求所钉住的 id 里包含两个。
- 客户端在本地渲染聊天模板，并发送渲染出的**字符串**。它不使用 chat/`messages` 那种 tokenize
  形式，因为那种形式不返回渲染后的字符串，而没有那个字符串，答案边界证明就无法执行。
- 该响应中的 `max_model_len` 就是客户端执行的上下文界限，因为分词的那个引擎正是必须接受它的
  引擎。评分端点报告的该值只是给"此处省略它的后端"用的回退。

## 3. 启动前置条件

该端点在启动时被探测一次，在监听端口绑定之前（本进程本就已拒绝在方言、读出档位或 checkpoint
身份不明时为请求服务）。这次探测会对**每一个**端点执行，无论 `--readout` 选了哪一项、也无论
是否允许媒体：构建任何评分请求都需要分词，所以它是服务的前置条件，而不是某一条读出的属性。
启动失败会点名那条路由与那次操作；实际观察到的原因（路由缺失、tokenizer 不可用、答案边界被打
破）由被包装的错误携带，而不是由外层消息断言。不存在到本地 tokenizer 的每请求回退，因为没有
本地 tokenizer。

这成为一条明确的部署前置条件：**后端必须向本服务暴露它的 tokenize 端点。** 没有 PR #9545 的
较旧 SGLang 无法被服务。

## 4. 答案边界证明

对渲染出的提示词 `P`，其槽位 `S[0..N)` 绑定到 `LETTERS`：

1. `ids = tokenize(P)`，要求结果非空且 `len(ids) <= context limit`。
2. 对下标 `i` 处的每个字母 `L`：`w = tokenize(P + L)`；要求 `w == ids + [S[i]]`——该字母**恰好**
   追加一个 token。
3. 要求所有 `S[i]` **两两不同**，这样一个槽位 id 就标识一个选项。

在一个方面，这比本地实现给出的证据更强：它由引擎将用来评分的同一个 tokenizer 执行。把
`" A"` 合并成一个 token 的 tokenizer 会在第 2 步失败，请求被拒绝，而不是在错误的位置上被评
分。

该证明按 `(last BOUNDARY_WINDOW ids, slots)` 做了备忘。一次命中跳过全部 `N + 1` 次调用，因此
稳定流入的同一形状提示词只需付一次证明的开销。

## 5. Provenance 后果

字段级的归属（哪个字段在哪条路径上存在）见 `docs/provenance.md`，那里是每个被记录字段的权威
位置。去掉本地 tokenizer 造成的后果是两件事：

- `server_tokenized` 现在在两条路径上都表示“id 来自后端”，因此消费者不再能用它区分文本与媒
  体；那是 `modality` 的职责。
- 媒体路径上本服务既不渲染也不分词，因此 `prompt_sha256` 与 `input_ids_sha256` 缺省，取而代
  之的是 `request_sha256`（实际发出的请求体）。

`prompt_version` 保持 `direct-options-v1`：提示词内容未变。哈希基线会移动，因为发出哈希的那个
进程变了——这正是需要有这份文档、而不是一句代码注释的原因。
