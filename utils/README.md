# utils

运行时模块集合，涵盖配置解析、LLM 调用、reasoning 展示、日志与会话保存。

## 模块说明

| 文件 | 作用 | 关键点 |
| --- | --- | --- |
| `runtime_config.py` | 统一解析 CLI + ENV | 输出 `RuntimeOptions`；提供 `add_runtime_args(parser)` |
| `thinking_policy.py` | 处理 thinking 能力与策略 | 支持 `auto/toggle/always/never`；生成 `enable_thinking`/`reasoning_effort` |
| `llm_call.py` | 统一 LLM 调用入口 | 兼容 stream/non-stream；返回 `assistant_content`/`assistant_reasoning`/`tool_calls`/`raw_metadata`；失败时可去参重试 |
| `reasoning_renderer.py` | reasoning 展示与交互 | 预览、折叠、下展（快捷键 `r`） |
| `trace_logger.py` | LLM 响应日志控制 | 输出 assistant/tool/reasoning 摘要 |
| `session_store.py` | 会话落盘 | JSONL 格式；文件名 `<model>_<YYYYMMDD_HHMMSS>.jsonl` |
