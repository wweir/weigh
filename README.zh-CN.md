# weigh

[English](README.md) | [简体中文](README.zh-CN.md)

从任意 OpenAI 兼容的推理服务上取回**逐槽位**的精确 logprob，并附带可为每个数字归因的
provenance —— 以 SemIf 的 `direct-options-v1` 决策 API 对外服务。

`weighd` 从生成式模型中读出一个决策：它询问**具名答案**的概率，而不是问模型"你想说什么"。对
一次请求，它校验单字段的决策 schema、渲染 `direct-options-v1` 提示词、向后端索取每个已声明槽位
在那唯一评分位置上的精确 logprob，然后把 argmax 作为一次 chat completion 返回——其 content 是本
地构造的，而不是模型生成的。

```bash
make build        # dist/weighd，版本与构建日期由 git 注入
make test         # hermetic 测试套件：无网络、无 GPU、无权重
make lint         # gofmt --check + go vet
just deploy -- --model /models/gemma-3-27b --url http://localhost:8000
```

`--url` 是后端的 **root**，不带 `/v1`：服务自己追加路径（`/version`、`/v1/models`、
`/tokenize`）。

## 作为库使用

同一套读出逻辑可以作为库引用：`github.com/wweir/weigh`。提示词由调用方渲染——库自己不渲染任何
聊天模板——返回的是声明槽位上的子集 softmax，以及可为它归因的 provenance：

```go
client, metadata, err := weigh.New(ctx, weigh.Config{
	URLs:    []string{"http://localhost:8000"},
	Backend: weigh.BackendVLLM,
	Readout: weigh.ReadoutExactSlot,
	Timeout: 120 * time.Second,
	// 不填 Source：本进程自己渲染提示词，没有本地 checkpoint 可读。
})
// 提示词是你自己构造的字节；optionIDs[i] 是你绑定到答案槽位 i（A..P）的标签。
scored, err := client.Decide(ctx, "row-1", finishedPrompt, optionIDs)
// scored.Probabilities[i] 是 optionIDs[i] 在那唯一评分位置上的分数。
// scored.PromptSHA256 就是你传入的那串字节的摘要。
```

`Config.Source` 可以不填。没有它就没有可检测的 checkpoint，因此：显式命名的模板族仍会被记录且
可渲染，`TemplateAuto` 会被拒绝，未指定的模板会让 `Metadata.PromptTemplate` 留空。想要服务端
那套 `direct-options-v1` 字节的调用方，可以用 `weigh.ValidateRow` + `weigh.RenderDirectOptions`
按自己配置的模板族构造出来，而不必重新实现这份契约。`weigh.New` 的探测与拒绝行为与二进制完全一
致；没有任何东西会静默降级。

## 为什么要有它

要忠实地做到这件事有三个要求，而本服务在启动时就强制这三条，而不是指望它们成立：

1. **槽位是具名的。** `logprob_token_ids`（vLLM）/ `token_ids_logprob`（SGLang）按构造返回每一个
   已声明的槽位，因此不存在某个槽位被静默遗漏的可能。
2. **答案是一个 token。** 客户端要求后端证明 `tokenize(prompt + letter) == ids + [slot]`，于是
   引擎评分的位置就是客户端所指的位置——而且这个证明是后端用自己的 tokenizer 做出来的，不是用
   第二个可能与之不一致的实现（见 `docs/tokenize-contract.md`）。
3. **数字可以被归因。** 每个响应都携带提示词哈希、读出路径、服务配置与所用回退——因为两种不同
   的读出永远不能合池，而没有 provenance 的数字不构成证据（见 `docs/provenance.md`）。

服务拒绝的每一件事都是大声拒绝：无法识别的聊天模板、无法证明 exact-slot 路由的后端、有两个字段
的 schema、出现在 criterion 里的图片。另一条路——猜——会产出看似合理但含义不同的数字。

## 目录结构

| 路径 | 说明 |
|---|---|
| `weigh.go`、`types.go`、`assemble.go` | 库的对外面：在调用方渲染好的提示词上使用 `weigh.New` / `Client.Decide`，以及组装原语与再导出的类型 |
| `cmd/weighd` | 入口：argv、配置、logger、退出码 |
| `internal/serve` | HTTP 契约：路由、校验、错误信封、SSE、`/health`、优雅退出、指标 |
| `internal/{schema,prompt,template,slots}` | 决策契约：拒绝集、载荷、聊天模板注册表、答案槽位 |
| `internal/{readout,tokenize,media}` | 读出：后端探测、exact-slot 请求、softmax、provenance、图像 |
| `api/openapi.json` | 机器可读契约，已内嵌并在 `GET /openapi.json` 提供 |
| `config/`、`deploy/` | 配置与 systemd unit |
| `docs/` | 设计文档；`ARCHITECTURE.md` 是索引 |

## 致谢

本服务实现的 `direct-options-v1` 契约，以及它继承的测量纪律，来自 **SemIf** 项目——一个从模型
选项 logits 中读出决策的独立研究工作。谨此致谢。

## 许可证

MPL-2.0。见 `LICENSE`。
