import type { GenerationReport } from "./report";

export type TestcaseGroup =
  | "sample"
  | "secret"
  | "random"
  | "boundary"
  | "custom";

export type StatementFormat = "md" | "html" | "pdf" | "plain";

export interface ProblemMetadata {
  id: string;
  title: string;
  slug?: string;
  sourceUrl?: string;
}

export interface Statement {
  language: string;
  format: StatementFormat;
  contentPath: string;
  contentHash: string;
}

export interface Testcase {
  id: string;
  name: string;
  group: TestcaseGroup;
  inputPath: string;
  answerPath?: string;
}

export interface ConstraintSet {
  items: unknown[];
}

export interface Artifact {
  id: string;
  language?: string;
  path: string;
}

export interface ProblemBundle {
  metadata: ProblemMetadata;
  statements: Statement[];
  constraints: ConstraintSet;
  samples: Testcase[];
  secrets: Testcase[];
  generators: Artifact[];
  validators: Artifact[];
  solutions: Artifact[];
  reports: GenerationReport[];
}
