# 决策契约

状态：对 Go 实现具有权威性。

## 1. 路由

每个响应都带有 `Content-Type`，包括 204 和每一个错误响应体。

| 方法 | 路径 | 成功时 |
|---|---|---|
| POST | `/v1/chat/completions` | 一个决策；当 `stream: true` 时为一条 SSE 流 |
| POST | `/v1/semif/batch` | 一个信封里最多 32 个相互独立的决策 |
| GET | `/health` | 存活状态与确切的当前服务配置 |
| GET | `/v1/models` | OpenAI 形状的模型列表 |
| GET | `/openapi.json` | 机器可读契约（`api/openapi.json`，已内嵌） |
| GET | `/metrics` | Prometheus 计数器与读出耗时直方图 |
| OPTIONS | 任意 | 204，无响应体 |

当方法本身是本服务会路由的方法（GET 或 POST）而路径无对应路由时，是 **404 `not_found`**；
对任何其它方法则是 **405 `method_not_allowed`**。因此 `POST /health` 是 404 而不是 405，
`GET /v1/chat/completions` 是 404。

## 2. 守卫，按顺序

1. **OPTIONS** 在最前面以 204 应答：浏览器不会给预检请求带上 bearer token。
2. **CORS**：设置了 `--cors-origin` 时，携带的 `Origin` 与配置值不同（或配置值为 `*` 时的任
   何来源）的请求，在做任何工作之前就是 **403 `origin_not_allowed`**。不带 `Origin` 的请求
   不是浏览器请求，不受 CORS 约束。
3. **鉴权**：设置了 `--api-key` 时，除 `GET /health` 外的每条路由都要求
   `Authorization: Bearer <key>`，比较采用常数时间。`/health` 保持开放，以便监管进程无需
   密钥即可探测存活。

## 3. 一个决策内部的校验顺序

只报告第一个失败，因此顺序是契约的一部分：

1. 请求体可解析且是 JSON 对象 → `invalid_json`
2. `model`、`n`、`tools`/`functions`/`tool_choice`、`max_tokens`/`max_completion_tokens` →
   `model_not_found` / `unsupported_parameter`
3. `stream` 与 `stream_options` → `invalid_parameter`
4. `logprobs` 与 `top_logprobs` → `invalid_parameter`
5. 决策 schema → `schema_*` 系列码，`param` 指出承载它的字段
6. `messages` → `messages_*`、`media_*` 与 content-part 系列码，`param: "messages"`
7. 读出，然后是响应

第 3、4 步跑在第 5 步**之前**，也在任何后端调用之前：`logprobs` 写错是调用方的错误，而在花掉
一次读出之后才应答它，既浪费了那次读出，又会把它计入 `semif_readouts_total`。

## 4. 错误信封

```json
{"error": {"message": "...", "type": "invalid_request_error", "code": "..."}}
```

`type` 在 5xx 时为 `server_error`，否则为 `invalid_request_error`。`param` 键当且仅当失败指
名了某个请求字段时出现。

| 状态 | 码 | 何时 |
|---|---|---|
| 401 | `invalid_api_key` | bearer token 缺失或错误 |
| 403 | `origin_not_allowed` | 浏览器来源被 `--cors-origin` 拒绝 |
| 404 | `model_not_found` | `model` 指名的不是当前服务的 checkpoint |
| 404 | `not_found` | 该路径无路由 |
| 405 | `method_not_allowed` | 本服务从不服务的方法 |
| 413 | `payload_too_large` | 请求体超出推导出的上限 |
| 400 | `invalid_json` | 请求体不是 JSON，或者不是对象 |
| 400 | `invalid_parameter` | `stream`、`stream_options`、`logprobs`、`top_logprobs`、批量的 `requests` |
| 400 | `unsupported_parameter` | `n`、`tools`、`functions`、`tool_choice`、非正的 `max_tokens` |
| 400 | `schema_*` | 15 种 schema 拒绝，见 `internal/schema` |
| 400 | `messages_*`、`media_*`、`*_content_part`、`*_media` | 见 `docs/media-contract.md` |
| 400 | `row_contract`、`prompt_contract` | 组装出的行或渲染出的提示词不可用 |
| 502 | `backend_error` | 后端失败，且未耗尽预算 |
| 503 | `client_gone` | 请求在取得并发槽位前已被取消（`/v1/semif/batch` 的该状态出现在单个条目里，信封本身仍是 200） |
| 504 | `backend_timeout` | 读出耗尽了它的全部预算 |
| 500 | `internal_error` | 本服务内部的矛盾 |

**错误文本保留自参考实现，包括首字母大写**（`"Cannot reach <url>: ..."`、
`"Each option needs string id and description fields"`）。该文本是可观测的，因此按它做匹配
的客户端并不算错；Go 的 `ST1005` 要求首词小写，此处刻意不遵循该 lint。调用方应当据以分支的
是 `code` 字段，而不是 message。

一处刻意的偏差：被点名报错的结构化值（`found [...]`、`enum contains ...`）按**紧凑 JSON**
渲染，而不是参考实现的 Rust `Debug` 表示；逐字保留的是其中的自然语言部分。以
`internal/schema` 的包注释为准。

## 5. 请求体上限

由媒体上限推导而来，这样 `--max-media-bytes` 才能真正生效，而不是被一个固定的请求体上限提前
截断：

```
media off            →  1 MiB
media on             →  max(1 MiB, (max_media_bytes/3*4 + 4) * max(max_images, 1) + 64 KiB)
```

当前生效的值在 `/health` 中以 `max_body_bytes` 报告。

## 6. 采样参数

该端点在一个答案槽位集上评出一个分布；它不采样。采样参数被接受、被忽略，并**回列**在
`semif.ignored_parameters` 中——从不静默。`logprobs` 与 `top_logprobs` 是例外：它们被真正采
纳，因此不在那个列表里。
