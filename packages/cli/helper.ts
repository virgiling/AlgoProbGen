import { cancel, intro, isCancel, log, note, outro } from "@clack/prompts";
import pc from "picocolors";

export type CommandStatus = "ready" | "planned";

export interface CommandSpec {
  readonly name: string;
  readonly usage: string;
  readonly description: string;
  readonly phase: string;
  readonly status: CommandStatus;
}

export const COMMANDS: readonly CommandSpec[] = [
  {
    name: "init",
    usage: "algogen init [--dir <dir>] [--db <path>]",
    description: "初始化本地题目工作区和 SQLite 数据库",
    phase: "Phase 0",
    status: "ready",
  },
  {
    name: "import",
    usage: "algogen import <problem.md> --solution <std.*>",
    description: "导入题面与标准解代码",
    phase: "Phase 1",
    status: "planned",
  },
  {
    name: "generate",
    usage: "algogen generate <problem-id> --generator random",
    description: "基于生成器批量创建测试数据",
    phase: "Phase 3",
    status: "planned",
  },
  {
    name: "judge",
    usage: "algogen judge <problem-id>",
    description: "执行 DOMjudge 风格的本地判题",
    phase: "Phase 2",
    status: "planned",
  },
  {
    name: "export",
    usage: "algogen export <problem-id> --format simple-zip",
    description: "导出可交付的问题包",
    phase: "Phase 4",
    status: "planned",
  },
  {
    name: "inspect",
    usage: "algogen inspect <problem-id>",
    description: "查看题目状态、数据和报告摘要",
    phase: "Phase 1",
    status: "planned",
  },
  {
    name: "search",
    usage: "algogen search <query>",
    description: "检索本地或远程题源",
    phase: "Phase 7",
    status: "planned",
  },
  {
    name: "fetch",
    usage: "algogen fetch <platform> <id>",
    description: "抓取外部题目元数据",
    phase: "Phase 7",
    status: "planned",
  },
  {
    name: "constraints",
    usage: "algogen constraints extract|edit <problem-id>",
    description: "提取或编辑题目约束",
    phase: "Phase 7",
    status: "planned",
  },
  {
    name: "plugin",
    usage: "algogen plugin list",
    description: "管理生成器、导出器和题源插件",
    phase: "Phase 6",
    status: "planned",
  },
  {
    name: "tui",
    usage: "algogen tui <problem-id>",
    description: "启动交互式工作台",
    phase: "Phase 8",
    status: "planned",
  },
];

export function printHelp(version: string): void {
  intro(`${pc.bgCyan(pc.black(" algogen "))} ${pc.gray(`v${version}`)}  本地优先的算法题生成与判题工具`);

  note(formatCommands("ready"), pc.bold("可用命令"));
  note(formatCommands("planned"), pc.bold("路线图命令"));

  outro(`执行 ${pc.cyan("algogen <command> --help")} 查看命令细节`);
}

export function printCommandHelp(command: CommandSpec): void {
  note(
    [
      `${pc.bold("Usage")}`,
      `  ${pc.cyan(command.usage)}`,
      "",
      `${pc.bold("Status")}`,
      `  ${formatStatus(command)}`,
      "",
      `${pc.bold("Description")}`,
      `  ${command.description}`,
    ].join("\n"),
    command.name,
  );
}

export function printPlannedCommand(command: CommandSpec): void {
  log.warn(`${pc.bold(command.name)} 还未实现，计划在 ${command.phase} 交付。`);
  printCommandHelp(command);
}

export function findCommand(name: string): CommandSpec | undefined {
  return COMMANDS.find((command) => command.name === name);
}

export function readOption(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
}

export function hasHelpFlag(args: string[]): boolean {
  return args.includes("--help") || args.includes("-h");
}

export function stopIfCancel<T>(value: T | symbol): T {
  if (isCancel(value)) {
    cancel("操作已取消");
    process.exit(0);
  }

  return value;
}

export function canPrompt(): boolean {
  return Boolean(process.stdin.isTTY && process.stdout.isTTY);
}

function formatCommands(status: CommandStatus): string {
  return COMMANDS.filter((command) => command.status === status)
    .map((command) => {
      const color = status === "ready" ? pc.green : pc.yellow;
      return [
        `${color(command.name.padEnd(12))} ${pc.gray(command.usage)}`,
        `             ${command.description} ${pc.dim(`(${command.phase})`)}`,
      ].join("\n");
    })
    .join("\n");
}

function formatStatus(command: CommandSpec): string {
  if (command.status === "ready") {
    return `${pc.green("ready")} ${pc.dim(command.phase)}`;
  }

  return `${pc.yellow("planned")} ${pc.dim(command.phase)}`;
}
