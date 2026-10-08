# 测试

刻意分开的两层：一套在任何地方都能跑、可用于卡住一次提交的 hermetic 套件，以及一次只有它才有
能力抓住线格式错误的实弹检查。

## 1. Hermetic 套件

`make test`（或 `go test ./...`）。无网络、无 GPU、无模型权重、无密钥：每个集成测试都在
`127.0.0.1` 上起一个假推理服务器，并让真实客户端指向它，因此 CI 除了一套 Go 工具链什么都不
需要。

| 包 | 它的测试负责什么 |
|---|---|
| `config` | 默认值、显式零值规则、每一种拒绝，以及 `Load()` 绑定每个 flag 与配置文件 |
| `jsonx` | 已发布的提示词布局、转义、非有限值拒绝、无序对象 |
| `schema` | 单字段单类型的拒绝集、槽位绑定、JSON 整数边界 |
| `template` | 每个族的真实渲染、基于标记的探测、`--template` 与 `tokenizer_config.json` 的关系 |
| `prompt` | `direct-options-v1` 载荷逐字节、evidence 编码、行级拒绝 |
| `tokenize` | 答案边界证明、提示词尾部的备忘、合并型 tokenizer、槽位碰撞 |
| `readout` | 后端探测、exact-slot 与 best-effort 两条路由、SGLang 适配器、softmax、媒体 |
| `media` | 接受的 content 形状、拒绝集、字节指纹、逐字透传 |
| `serve` | 路由与守卫顺序、校验顺序、错误信封、SSE、批量逐项状态、优雅退出、媒体 |

有三条性质值得明说，因为否则假后端会把它们掩盖掉：

- **假后端说的是实测的线格式，而不是方便的格式。** vLLM 的两条路由在 `logprobs` 的形状上并不
  一致，所以假后端按实测实现两种形状（`docs/media-contract.md` §5）。只返回解析器恰好期待的
  那一种形状的假后端，什么也证明不了。
- **断言针对本服务构建出的请求**，而不只是它解析的响应：提示词必须以 token id 形式到达、槽位
  必须被钉住、`add_special_tokens` 必须为 false。
- **拒绝是被对抗性地测试的。** 把 `" A"` 合并成一个 token 的 tokenizer、忽略钉住 id 的构建、
  落在返回的 top-N 之外的槽位——启动或该请求必须失败，而不是产生一个数字。

## 2. 实弹检查

可选，不属于套件，而且是至今唯一抓住过线格式错误的一层。它需要一个真实的多模态后端和一个
checkpoint 目录（被服务根目录的名字必须与本地目录名一致，否则需要
`--allow-tokenizer-mismatch`）。

```bash
make build
mkdir -p /tmp/weigh-e2e/<basename of the served root>
./dist/weighd --model /tmp/weigh-e2e/<basename> --url http://<host>:<port> \
  --revision live --allow-media --timeout 120 &
curl -sS http://127.0.0.1:8080/health | python3 -m json.tool     # tier_a, media_support, max_model_len
curl -sS -X POST -H 'content-type: application/json' \
  -d @decision.json http://127.0.0.1:8080/v1/chat/completions
kill %1
```

`--url` 是根地址，不带 `/v1`。要回读的东西：`choices[0].semif` 必须携带真实的槽位 id、那对摘
要、读出名称与服务配置；媒体请求必须携带 `modality: text+image`、`request_sha256`，以及后端
自己的 `input_tokens`。

以这种方式找到的两个缺陷，两者对一套通过的 hermetic 套件都是不可见的：

1. **每个多词 flag 都绑定到了空气**——包括 `--api-key`，它让服务在运维以为鉴权已开启的情况下
   处于开放状态。套件里没有任何测试调用真实的 `Load()`，所以 flag 到配置这条路径从未被执行过。
2. **chat 路由的 `logprobs` 形状被按 completions 路由的形状解析了**，于是媒体探测把一个完全
   正常的答案读成不可读。

教训同时在这两件事里：一个从不跨越真实边界的测试看不见形状错误，而一个映照解析器假设的假后端
同样看不见。
