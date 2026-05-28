export type RunStatus = "OK" | "TLE" | "MLE" | "OLE" | "RE" | "SG" | "JE";

export interface CompileRequest {
  sourcePath: string;
  language: string;
  workDir: string;
}

export interface CompileResult {
  ok: boolean;
  executablePath?: string;
  message?: string;
}

export interface RunRequest {
  executablePath: string;
  workDir: string;
  stdinPath?: string;
  stdinText?: string;
  timeLimitMs: number;
  wallTimeLimitMs: number;
  memoryLimitBytes: number;
  outputLimitBytes: number;
}

export interface RunResult {
  status: RunStatus;
  exitCode: number | null;
  signal: string | null;
  timeMs: number;
  wallTimeMs: number;
  memoryBytes: number;
  stdoutPath: string;
  stderrPath: string;
}

export interface SandboxRunner {
  backend: "mock" | "native" | "docker" | "simple-sandbox" | string;
  compile(req: CompileRequest): Promise<CompileResult>;
  run(req: RunRequest): Promise<RunResult>;
  cleanup(sessionId: string): Promise<void>;
}
