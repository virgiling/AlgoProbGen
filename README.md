# algogen

`algogen` 是一个多 Agent 自动出题工具：读取题面、改写描述、生成数据、调用模板题解产出标准输出，并打包 zip。

## 安装

```bash
uv sync
```

## 配置

1. 复制配置模板：

```bash
cp .env.example .env
```

2. 至少填写以下变量：

- `DESC_AGENT_MODEL`
- `BENCH_AGENT_MODEL`
- 对应模型提供商 API Key（例如 `OPENAI_API_KEY`）

可选变量：

- `DB_DIR`（默认 `./db`）
- `TEMPLATE_DIR`（默认 `./template`）
- `OUTPUT_DIR`（默认 `./output`）
- `BENCH_NUMBER`（默认 `5`）
- `THREADS`（默认 `1`）
- `LANGUAGE`（默认自动检测模板后缀）

## 运行

推荐使用 console script：

```bash
uv run algogen run 1000
```

等价模块方式：

```bash
uv run python -m algogen.cli run 1000
```

常用参数：

```bash
uv run algogen run 1000 --bench-number 10 --threads 4 --output-dir ./output
uv run algogen run 1000 --language cpp
uv run algogen run 1000 --no-progress
```

## 进度显示

CLI 默认显示两层进度：

- 阶段进度（rewrite / extract_spec / gen_sampler / generate_data / solve_data / archive）
- 子任务进度（数据生成 `current/total`）

示例：

```txt
[stage] [--------------------] 0/6 开始: 改写题面
[stage] [###-----------------] 1/6 完成: 改写题面
[stage] [##########----------] 3/6 开始: 生成输入输出数据
    [subtask] [######--------------] 3/10 进行中: 生成数据 case=3
    [subtask] [####################] 10/10 完成: 生成数据
[stage] [#############-------] 4/6 完成: 生成输入输出数据
```

## 输入与输出约定

- 输入题面：`db/<problem_id>.md`
- 模板题解：`template/<problem_id>.<suffix>`
- 输出目录：`output/<problem_id>/`

典型输出文件：

- `description.md`
- `spec.json`
- `sampler.py`
- `*.in` / `*.out`
- `output/<problem_id>.zip`

## 常见错误

- `[config-error] Missing required config ...`：缺少模型名或 API Key。
- `[run-error] Problem markdown not found ...`：缺少 `db/<problem_id>.md`。
- `Multiple solver files detected ...`：同题号模板有多个语言后缀，需显式传 `--language`。
