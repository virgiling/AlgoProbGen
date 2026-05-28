#!/usr/bin/env bun
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { intro, log, outro, spinner, text } from "@clack/prompts";
import pc from "picocolors";
import packageJson from "@package.json" with { type: "json" };
import { openDatabase } from "../db";
import {
  canPrompt,
  findCommand,
  hasHelpFlag,
  printCommandHelp,
  printHelp,
  printPlannedCommand,
  readOption,
  stopIfCancel,
} from "./helper";

interface InitOptions {
  dir: string;
  dbPath: string;
}

async function main(argv: string[]): Promise<number> {
  const [commandName, ...args] = argv;
  const version = packageJson.version ?? "0.0.0";

  if (!commandName || commandName === "help" || hasHelpFlag([commandName])) {
    printHelp(version);
    return 0;
  }

  if (commandName === "--version" || commandName === "-v") {
    console.log(version);
    return 0;
  }

  const command = findCommand(commandName);
  if (!command) {
    log.error(`未知命令: ${pc.bold(commandName)}`);
    printHelp(version);
    return 1;
  }

  if (hasHelpFlag(args)) {
    printCommandHelp(command);
    return 0;
  }

  if (command.status === "planned") {
    printPlannedCommand(command);
    return 1;
  }

  if (command.name === "init") {
    await init(args);
    return 0;
  }

  log.error(`命令未绑定处理器: ${pc.bold(command.name)}`);
  return 1;
}

async function init(args: string[]): Promise<void> {
  const options = await resolveInitOptions(args);

  intro(`${pc.bgCyan(pc.black(" algogen init "))} ${pc.dim("Phase 0")}`);

  const configPath = join(options.dir, "algogen.json");
  if (existsSync(configPath)) {
    throw new Error(`配置文件已存在: ${configPath}`);
  }

  const s = spinner();
  s.start("初始化工作区");

  mkdirSync(options.dir, { recursive: true });
  const db = openDatabase(options.dbPath);
  db.close();
  writeConfig(configPath, options);

  s.stop("工作区已就绪");
  log.success(`Config   ${pc.cyan(configPath)}`);
  log.success(`Database ${pc.cyan(options.dbPath)}`);
  outro(`下一步: ${pc.cyan("algogen import <problem.md> --solution <std.*>")}`);
}

async function resolveInitOptions(args: string[]): Promise<InitOptions> {
  const flagDir = readOption(args, "--dir");
  const flagDb = readOption(args, "--db");

  if (!canPrompt()) {
    const dir = flagDir ?? ".";
    return {
      dir,
      dbPath: flagDb ?? join(dir, "algogen.sqlite"),
    };
  }

  const dir = flagDir ?? stopIfCancel(
    await text({
      message: "工作区目录",
      placeholder: ".",
      defaultValue: ".",
      validate(value) {
        if (!value?.trim()) return "目录不能为空";
      },
    }),
  );

  const dbPath = flagDb ?? stopIfCancel(
    await text({
      message: "SQLite 数据库路径",
      placeholder: join(dir, "algogen.sqlite"),
      defaultValue: join(dir, "algogen.sqlite"),
      validate(value) {
        if (!value?.trim()) return "数据库路径不能为空";
      },
    }),
  );

  return { dir, dbPath };
}

function writeConfig(configPath: string, options: InitOptions): void {
  writeFileSync(
    configPath,
    `${JSON.stringify({ dbPath: options.dbPath }, null, 2)}\n`,
    { flag: "wx" },
  );
}

const exitCode = await main(Bun.argv.slice(2)).catch((error) => {
  log.error(error instanceof Error ? error.message : String(error));
  return 1;
});

process.exit(exitCode);
