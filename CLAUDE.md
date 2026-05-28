# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`algogen` — 算法自动出题系统：从题目描述出发，自动生成测试数据、判题、打包导出为 Kattis/DOMJudge 格式。

## Build / Test / Run

```bash
bun install                 # 安装依赖（registry = npmmirror.com）
bun test                    # 运行所有测试
bun run <file>.ts           # 直接执行 TypeScript 文件
bun x tsx <file>.ts         # 或使用 tsx 执行
```

## Architecture

Monorepo（Bun workspaces），`packages/*` 分层，各包职责见 [PLAN.md](PLAN.md) 第 2.1 节。

**核心设计原则：**
- **acquisition、sandbox、judge、generator、exporter 都是可替换模块** — Agent 只能通过这些模块的接口工作，不能直接读写文件系统、启动进程、绕过本地数据库
- **所有不可信代码（用户 solution、LLM 生成的 generator、validator、checker）都走 sandbox**，不能由主进程直接执行
- **本地 SQLite 是事实来源**，不是缓存；在线获取只是来源之一
- **CLI 先行，TUI 后置** — 所有功能必须能 headless 调用

**模块依赖方向：** `core` ← 所有包，`db` ← 持久化层，`sandbox`/`judge`/`generator`/`acquisition`/`exporters` 平级，`agents` 跨切编排，`cli`/TUI 在顶层

## Tech Stack

| 层 | 选择 |
|---|---|
| 语言 | TypeScript 5.x strict |
| 运行时 | Bun 1.3.x + bun:sqlite |
| DB Access | `bun:sqlite` + raw SQL schema + module singleton |
| CLI | clipanion (rc.4) |
| TUI | Ink（后置 milestone） |
| LLM SDK | Vercel AI SDK v6 + `generateText`/`streamText` + `Output.object` |
| Schema/Validation | Zod v4 |
| 沙盒 | `@algogen/sandbox` 抽象接口，默认 Linux 后端参考 `simple-sandbox` |

## Key Patterns

- **跨模块边界统一使用 Zod schema** — 不在模块间传裸 object
- **LLM 调用统一走 `LLMClient` 封装层**，业务代码不直接依赖 AI SDK 版本细节
- **结构化输出**使用 `Output.object({ schema })`，不使用旧式 `generateObject`
- **SandboxRunner 抽象** — `compile`/`run`/`cleanup` 三个方法，backend 可替换（simple-sandbox / docker / mock）
- **Generator 基类** — `GeneratorBase<TParams>` 含 `setup` → `generate`(AsyncIterable) → `teardown` 生命周期
- **判题流水线**：prepare → generate → validate-input → answer → judge → report
- **ProblemSourceProvider 接口** — 所有题源（Codeforces、web search、本地导入）实现同一接口

## Current State

**Milestone 0（架构冻结 + 骨架）基本完成。**

- M0 精简 packages 已创建：`core`、`db`、`sandbox`、`judge`、`generator`、`exporters`、`cli`
- `packages/core/` 定义 M0 领域接口和 Zod schema
- `packages/db/` 使用 `bun:sqlite`，已定义 M0 核心表 `SCHEMA` 与 TypeScript entity interfaces
- `packages/db/index.ts` 提供 `createDatabase()`、`initDatabase()`、`openDatabase()` 和懒加载模块单例 `getDb()`
- `packages/cli/` 支持 `algogen --version` 和 `algogen init`
- DB 层按 Bun module singleton 模式提供 `getDb()`；测试、CLI init 和可配置场景使用 `openDatabase(path)`

详细路线图见 [PLAN.md](PLAN.md) 第 11 节。下一步应进入 Phase 1：本地导入闭环。
