# Claude Code From Scratch

从 0 到 1 构建一个可运行、可扩展的 Claude Code-like Agent。当前仓库已实现 `v01` 最小闭环：模型可调用 `bash` 工具，执行命令后将结果写回上下文，持续循环直到返回最终回答。

## What This Repo Contains
- `agents/v01_agent_loop.py`: 当前主程序，最小可用的 tool-calling agent loop。
- `prompts/v01_agent_loop.md`: `v01` 系统提示词模板。
- `docs/v01_agent_loop.md`: `v01` 设计说明与运行示例。
- `tests/test_utils_refactor.py`: 功能覆盖型测试入口。
- `tests/reports/`: 测试 JSON 报告输出目录。

## Quick Start
### 1) Environment
建议 Python 3.10+。

```bash
python --version
```

### 2) Install Dependencies
```bash
pip install openai python-dotenv
```

### 3) Configure `.env`
在项目根目录创建或更新 `.env`：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-compatible-endpoint
LLM_MODEL=your_model_name
```

### 4) Run the Agent
```bash
python agents/v01_agent_loop.py
```

常用参数：

```bash
python agents/v01_agent_loop.py --max-turns 30
python agents/v01_agent_loop.py --thinking on --show-thinking
python agents/v01_agent_loop.py --api-mode responses
```

## How the Loop Works
1. 读取系统提示词与环境变量，初始化客户端。  
2. 发送用户消息给模型。  
3. 若模型返回 `tool_calls`，执行工具（当前为 `bash`）。  
4. 将工具结果写回 `messages`，继续下一轮。  
5. 无工具调用时结束并输出最终回答。

## Testing
运行测试：

```bash
python tests/test_utils_refactor.py --mode mock --output-json tests/reports/utils_function_test_report.json
```

可选：

```bash
python tests/test_utils_refactor.py --pattern runtime --verbosity 2
```

## Development Notes
- 版本演进采用 `vNN_*` 命名（例如 `v01_agent_loop.py`）。
- 新增功能优先保持“最小改动 + 可验证”。
- 工具执行能力存在安全风险，扩展前请先做命令白名单/权限边界设计。
