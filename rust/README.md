# mini-SWE-agent Rust runtime

这是一个面向高性能与可维护性的 Rust 重写版本，保留了 Python 参考实现的核心架构：

```text
Model       -> 语言模型边界（OpenAI-compatible chat completions / text-based）
Environment -> 命令执行边界（local / docker）
DefaultAgent-> 线性 agent 控制循环
```

## 快速开始

```bash
cd rust
cargo run --bin mini -- --model openai/gpt-4o-mini --task "fix the failing test"
```

也可以传入自定义配置：

```bash
cargo run --bin mini -- -c mini.yaml -c model.model_kwargs.temperature=0.2 -t "..."
```

内置配置会嵌入二进制，因此即使没有 Python 源码目录也能启动。当前内置配置包括：
`default.yaml`、`mini.yaml`、`mini_textbased.yaml`。

## 架构

| 模块 | 职责 |
| --- | --- |
| `config` | YAML/键值配置加载、递归合并、内置配置嵌入 |
| `models` | 基于 `llm-connector` 的通用 LLM 接入、tool-call 与 text-based 动作解析 |
| `environments` | 本地 shell 与 Docker 执行环境，异步进程与超时控制 |
| `agent` | 线性 query -> execute -> observe 循环、限额、格式错误恢复、轨迹保存 |
| `template` | 基于 minijinja 的 Jinja 兼容模板渲染 |
| `run` | `mini` CLI |

## 性能设计

- **异步 I/O**：模型 HTTP 请求和子进程执行都基于 Tokio，避免 Python GIL 与线程阻塞。
- **线性热路径**：每个 agent step 只做模型调用、动作解析、进程执行、消息追加，不引入额外间接层。
- **零复制倾向的数据结构**：消息使用 `serde_json::Value` + 扁平字段，协议扩展无需维护两套类型；序列化/反序列化只在边界发生。
- **进程并发读取**：stdout/stderr 并行读取，避免管道缓冲区互相阻塞。
- **显式超时与进程回收**：子进程设置 `kill_on_drop`，Docker 环境支持清理。
- **可静态分发的兼容层**：内置配置通过 `include_str!` 嵌入，减少启动时文件系统与解析开销。
- **细粒度测试**：核心状态机、配置合并、动作解析、本地执行均有集成测试。

## 当前范围与限制

- 模型层通过 `llm-connector = "=1.4.0"` 提供通用协议支持，默认使用 OpenAI-compatible `/chat/completions`；可通过 `provider` / `service_name` 选择 `llm-connector` 支持的 provider。
- `llm-connector` 的 provider 协议已接入，但 mini-SWE-agent 侧的成本计算、Anthropic cache control、Responses API 与多模态扩展仍在逐步补齐。
- `interactive` 模式提供显式终端确认（human / confirm / yolo）；暂不包含 Python 版本的 rich/textual TUI。
- Docker 环境要求宿主机安装 `docker` 或 `podman`，并通过 `--environment-class docker` 选择。

这些边界都可以通过实现对应 trait 逐步替换，而不需要修改 agent 控制循环。


