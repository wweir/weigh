# 媒体契约

状态：对 Go 实现具有权威性。

## 1. 带媒体的请求是一次降级的读出，而不是"多了一种 content 类型"

文本契约是狭窄的：聊天模板被渲染并分词，答案边界被**证明**（`tokenize(P + L) == ids +
[slot]`）。一张图像破坏了其中最后一条：后端的 processor 把它展开成一系列本服务永远看不到的
占位 token，因此关于评分位置不存在任何本地证明。

因此，携带媒体的请求会切换读出，并且每个响应都会说明这一点：

| 字段 | 文本 | 媒体 |
|---|---|---|
| `choices[].semif.modality` | `"text"` | `"text+image"` |
| `prompt_sha256` | 渲染出的提示词 | 缺省 |
| `input_ids_sha256` | 被钉住的那些 id | 缺省 |
| `request_sha256` | 缺省 | 实际发出的请求体 |
| `semif.serving_config` | `vllm-openai-logprob-token-ids-v1`，… | `vllm-openai-chat-media-token-ids-v1` 或 `…-topn-v1` |
| `semif.evidence_renderer` | `system+user`、`transcript` | `system+user+media`、`transcript+media` |

两种读出结果永远不能合池，这正是附加一张图像时 renderer 和配方两者都会改变的原因。

## 2. 透传是铁律

`Media.Reference` 是调用方 `image_url.url` 的**逐字节原样**，转发给后端的正是这个确切的字符
串。不存在任何重新编码或改写成规范形式：记录 sha256 的意义就在于，被记录的值描述的是实际发
出的东西。

解码只是旁路，仅用于一个尺寸上限和一份字节指纹，它**永不**回喂被转发的引用。MIME 按调用方声
明的那样记录——某个类型是否为可用图像是后端 processor 的决定，而第二份白名单会拒绝掉更新的
checkpoint 能处理的图像，同时不增加任何后端尚未强制的东西。

## 3. 接受的形状

一条消息的 `content` 要么是字符串，要么是部件数组：

- `{"type": "text", "text": "<string>"}` —— 各部件以换行拼接。
- `{"type": "image_url", "image_url": {"url": "<reference>"}}` —— `url` 是**唯一**会被转发的
  键。

OpenAI 在 `image_url` 上定义的 `detail` 及其它键决定后端**如何**渲染图像，即模型看到多少
image token。丢掉其中一个就会以默认分辨率作答，而响应看起来与明确要求的那个请求一模一样，
因此它们的存在会被拒绝。

## 4. 拒绝集

一张图像可能不被接受的每一种情形都被点名，因为替代方案——丢掉图像、只给文本打分——回答的是
调用方没有问的问题。每条消息都携带 `messages[<index>]`。

| 码 | 何时 |
|---|---|
| `media_not_enabled` | 出现 `image_url` 部件却没有 `--allow-media` |
| `media_remote_not_allowed` | 出现 `http(s)://` 引用却没有 `--allow-remote-media` |
| `invalid_media` | 没有 `image_url.url`；引用既不是 `data:` URI 也不是 http(s)（文件系统路径会在后端主机上成为一个文件读取原语）；`data:` URI 没有逗号；非 base64 的 `data:` URI（其解码后大小无法被界定，上限将不可执行）；base64 非法；解码后为零字节 |
| `media_too_large` | 编码后的载荷超出推导出的上限，或解码后字节超出 `--max-media-bytes` |
| `too_many_images` | 单条消息超出 `--max-images` |
| `unsupported_content_part` | 部件没有 `type`；`text` 不是字符串；部件类型未知；`image_url` 携带了除 `url` 以外的任何键 |
| `messages_not_semif_contract` | 完全没有 `content` |

每请求图像上限与 `media_in_criterion`（`system`/`developer` 消息里出现图像，那会改变**被问的
是什么**）由 `messages` 折叠一并施加，与文本折叠在一起。

## 5. 为图像评分

**chat 路由的响应形状不是 completions 路由的形状。** 在 vLLM 0.27.1 上实测：

- 设置了 `return_tokens_as_token_ids` 时，`POST /v1/completions` 的
  `choices[0].logprobs.top_logprobs[0]` 是一个**以** `token_id:<id>` **为键的对象**。
- `POST /v1/chat/completions` 把评分位置嵌在 `choices[0].logprobs.content[0]`，它的
  `top_logprobs` 是一个**条目列表**，每个条目带一个 `token` 字段（`"token_id:236776"`），
  外加 `logprob` 与 `bytes`。

用其中一种形状的解析器去读另一种，会把一个完全正常的答案报成不可读——这正是本服务第一次遇到
真实多模态部署时发生的事。此处使用的解析器（`chatLogprobEntry`、`chatTokenIDs`、
`parseChatSlots`）读的是 chat 形状。

`logprob_token_ids` 保证被请求的那些 id 无论 top-N 取多少都在那个列表里：实测即使在
`top_logprobs: 1` 时，两个被钉住的 id 也都在。因此缺一个属于契约违约，而不是排名上的偶然。

两条媒体路由都向后端索取 token id 形式的键（`return_tokens_as_token_ids`）。以解码后的 token
文本作答的构建会被拒绝：没有本地 tokenizer，本服务无法按 id 寻址那些条目，而靠猜去匹配正是
让一份看似合理却错误的分布被记录下来的一条路。

| 路由 | 请求 | 槽位落在返回的 top-N 之外时 |
|---|---|---|
| exact-slot（`vllm-openai-chat-media-token-ids-v1`） | `logprob_token_ids` + `return_tokens_as_token_ids` | 拒绝：该端点已被探测证明会回显每一个已声明的槽位 |
| top-N（`vllm-openai-chat-media-topn-v1`） | `return_tokens_as_token_ids`、`logprobs: 20` | 拒绝：**图像提示词没有 `/generative_scoring` 回退**，因为那条路由只接受 token id，无法携带图像 |

启动探测靠**哪一个请求成功了**来决定存在哪条路由，而不只靠响应形状：钉住形式成功即证明
exact-slot 路由存在，而钉住形式被拒（400/422）、未钉住形式却能应答，则证明只有 best-effort
路由。此处单看响应形状无法区分二者，因为两种形式都在索取 token id 形式的键。

`--allow-media` 只接受 exact-slot 路由。只提供 best-effort 路由的构建会被禁用媒体，而不是被
静默升级；`--allow-media-topn` 才是那道允许放行的开关。以解码 token 文本为键作答的构建，在此
无论策略如何都完全没有媒体路由。

## 6. Provenance

每张图像记录一个对象，顺序与给出的图像顺序一致：

```json
{"kind": "image", "mime": "image/png", "bytes": 67, "sha256": "<hex>", "url": null}
```

`url` 恰在没有本地指纹时逐字重述那个引用——远程 URL 是唯一一种什么都不哈希的情形，因此
`sha256` 存在与 `url` 存在不可能互相矛盾。
