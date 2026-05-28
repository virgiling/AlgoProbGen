export const CORE_TABLES = [
  "problems",
  "problem_statements",
  "solutions",
  "generator_runs",
  "testcases",
  "judge_runs",
  "source_documents",
  "llm_cache",
] as const;

export type CoreTable = (typeof CORE_TABLES)[number];

export interface Problem {
  id: string;
  title: string;
  slug: string | null;
  source_platform: string | null;
  source_url: string | null;
  license: string | null;
  rights_owner: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProblemStatement {
  id: string;
  problem_id: string;
  language: string;
  format: string;
  content_path: string;
  content_hash: string;
}

export interface Solution {
  id: string;
  problem_id: string;
  kind: string;
  language: string;
  source_url: string | null;
  content_path: string;
  content_hash: string;
  provenance_json: string | null;
}

export interface GeneratorRun {
  id: string;
  problem_id: string;
  generator_id: string;
  seed: number | null;
  params_json: string | null;
  artifact_dir: string | null;
  status: string;
}

export interface Testcase {
  id: string;
  problem_id: string;
  group_name: string;
  name: string;
  input_path: string;
  answer_path: string | null;
  generator_run_id: string | null;
  description: string | null;
}

export interface JudgeRun {
  id: string;
  problem_id: string;
  testcase_id: string | null;
  solution_id: string | null;
  verdict: string;
  time_ms: number | null;
  wall_time_ms: number | null;
  memory_kb: number | null;
  stdout_digest: string | null;
  stderr_digest: string | null;
  created_at: string;
}

export interface SourceDocument {
  id: string;
  problem_id: string | null;
  source_type: string;
  url: string | null;
  fetched_at: string | null;
  license_hint: string | null;
  raw_path: string | null;
  normalized_path: string | null;
  content_hash: string | null;
}

export interface LLMCacheEntry {
  cache_key: string;
  model: string;
  prompt_hash: string;
  output_json: string;
  created_at: string;
}

export const SCHEMA = `
CREATE TABLE IF NOT EXISTS problems (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  slug TEXT,
  source_platform TEXT,
  source_url TEXT,
  license TEXT,
  rights_owner TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS problem_statements (
  id TEXT PRIMARY KEY,
  problem_id TEXT NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
  language TEXT NOT NULL,
  format TEXT NOT NULL,
  content_path TEXT NOT NULL,
  content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS solutions (
  id TEXT PRIMARY KEY,
  problem_id TEXT NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  language TEXT NOT NULL,
  source_url TEXT,
  content_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  provenance_json TEXT
);

CREATE TABLE IF NOT EXISTS generator_runs (
  id TEXT PRIMARY KEY,
  problem_id TEXT NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
  generator_id TEXT NOT NULL,
  seed INTEGER,
  params_json TEXT,
  artifact_dir TEXT,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS testcases (
  id TEXT PRIMARY KEY,
  problem_id TEXT NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
  group_name TEXT NOT NULL,
  name TEXT NOT NULL,
  input_path TEXT NOT NULL,
  answer_path TEXT,
  generator_run_id TEXT REFERENCES generator_runs(id),
  description TEXT
);

CREATE TABLE IF NOT EXISTS judge_runs (
  id TEXT PRIMARY KEY,
  problem_id TEXT NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
  testcase_id TEXT REFERENCES testcases(id),
  solution_id TEXT REFERENCES solutions(id),
  verdict TEXT NOT NULL,
  time_ms INTEGER,
  wall_time_ms INTEGER,
  memory_kb INTEGER,
  stdout_digest TEXT,
  stderr_digest TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS source_documents (
  id TEXT PRIMARY KEY,
  problem_id TEXT REFERENCES problems(id) ON DELETE CASCADE,
  source_type TEXT NOT NULL,
  url TEXT,
  fetched_at TEXT,
  license_hint TEXT,
  raw_path TEXT,
  normalized_path TEXT,
  content_hash TEXT
);

CREATE TABLE IF NOT EXISTS llm_cache (
  cache_key TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  prompt_hash TEXT NOT NULL,
  output_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
`;
