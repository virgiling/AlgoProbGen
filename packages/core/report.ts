export type JudgeVerdict =
  | "AC"
  | "WA"
  | "TLE"
  | "MLE"
  | "OLE"
  | "RE"
  | "CE"
  | "JE";

export interface JudgeTestcaseReport {
  testcaseId: string;
  name: string;
  verdict: JudgeVerdict;
  timeMs: number;
  wallTimeMs: number;
  memoryKb: number;
  message?: string;
}

export interface GenerationReport {
  problemId: string;
  runId: string;
  summary: Partial<Record<JudgeVerdict, number>>;
  testcases: JudgeTestcaseReport[];
}
