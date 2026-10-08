# Provenance

状态：对 Go 实现具有权威性。

一个没有 provenance 的数字不是证据：两种不同的读出结果永远不能合池。因此每个响应都记录下是
哪个提示词、哪些槽位、哪条读出路和哪套服务配置产生了它。

## 1. 每个决策：`choices[].semif`

| 字段 | 含义 |
|---|---|
| `field` | schema 声明的那唯一一个属性名 |
| `values` | schema 允许的取值，按槽位顺序 |
| `labels` | 每个取值面向模型的文本 |
| `choice` | 胜出的取值，保留其 JSON 类型 |
| `choice_index` | 它的槽位下标；并列时取最小下标 |
| `probabilities` | 在已声明槽位上的子集 softmax |
| `option_logprobs` | 后端为这些槽位返回的值 |
| `answer_token_ids` | 每个槽位的 token id |
| `option_ids` | 调用方的标签，按槽位顺序 |
| `readout` | 配方描述，点名实际走的那条路由 |
| `fallback_used` | 是否有槽位落在 top-N 之外、由 `/generative_scoring` 作答 |
| `endpoint` | 机群中哪一台后端服务了这一行 |
| `input_tokens` | 提示词长度；在媒体路径上是后端自己的计数 |
| `input_ids_sha256` | 评分请求所钉住的那些 id 的摘要（仅文本） |
| `prompt_sha256` | 渲染出的提示词的摘要（仅文本） |
| `request_sha256` | 实际发出的请求体的摘要（仅媒体） |
| `modality` | `"text"` 或 `"text+image"` |
| `server_tokenized` | 每条路径上都是 `true`：id 来自后端 |
| `media` | 每张图像一个 provenance 对象，顺序与给出的一致 |
| `probability_status` | 长期保留的告诫：这是条件性的选项分数，不是经校准的置信度 |
| `engine_seconds` | 那次评分调用 |
| `fallback_seconds` | `/generative_scoring` 循环；未运行时为 `0` |
| `total_seconds` | 整次读出，含分词 |

有两个字段相互作用、且容易被误读：

- **`server_tokenized` 不再区分文本与媒体。** 自从本地 tokenizer 被后端的取代之后，它在所有
  地方都是 `true`；区分两条路径的是 `modality`。见 `docs/tokenize-contract.md`。
- **那些摘要按路径互斥。** `prompt_sha256` 与 `input_ids_sha256` 只在本服务既渲染又分词了提
  示词的地方存在。在媒体路径上是后端分词的，因此诚实的承诺更弱：`request_sha256` 标识实际发
  出的那个请求体。

## 2. 每个响应：顶层的 `semif`

| 字段 | 含义 |
|---|---|
| `backend` | 探测所判定的方言（`vllm`、`sglang`） |
| `serving_config` | 读出配方：`vllm-openai-logprob-token-ids-v1`、`vllm-openai-completions-subset-softmax-v1`、`sglang-native-generate-token-ids-v1`，或某个媒体配方 |
| `media_support` | 探测就图像证明了什么（`none`、`chat-top-n`、`chat-exact-slot`） |
| `prompt_version` | 标识提示词**内容**的版本（`direct-options-v1`） |
| `prompt_template`、`prompt_template_source` | 标识**渲染**的版本，以及族是如何选定的（`explicit`、`detected:<path>`、`default:no-chat-template`） |
| `revision` | 运维为这个 checkpoint 打的标签 |
| `prompt_contract` | 一句话说明这个提示词是什么意思 |
| `evidence_renderer` | 跑了哪种折叠：`system+user`、`transcript`、`system+user+media`、`transcript+media` |
| `ignored_parameters` | 被接受、但没有产生任何影响的采样参数 |
| `client` | 本实现 |
| `tokenize_endpoint` | 本部署经由哪条后端路由分词 |

`prompt_version` 与 `prompt_template` 合起来才标识出服务器实际收到的那串字符；单靠其中任何一个
都不行。记录 `evidence_renderer` 是因为折叠会改变提示词，所以在媒体折叠存在之前服务的决策，
无法与之后服务的决策相互比较。

## 3. 机群

`/health` 报告每一个端点 URL，以及启动探测对它们施加的检查：相同方言、相同的首个被服务模型
id、相同的 checkpoint 目录名、相同的读出档位、相同的媒体支持、相同的服务器版本与相同的
`max_model_len`。任一端点上缺字段都是拒绝，而不是一次 `None == None` 的通过，因为"这些端点完
全相同"必须被验证，而不是被假定。
