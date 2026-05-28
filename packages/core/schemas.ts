import { z } from "zod";

export const TestcaseGroupSchema = z.enum([
  "sample",
  "secret",
  "random",
  "boundary",
  "custom",
]);

export const JudgeVerdictSchema = z.enum([
  "AC",
  "WA",
  "TLE",
  "MLE",
  "OLE",
  "RE",
  "CE",
  "JE",
]);

export const RunStatusSchema = z.enum([
  "OK",
  "TLE",
  "MLE",
  "OLE",
  "RE",
  "SG",
  "JE",
]);

export const ProblemMetadataSchema = z.object({
  id: z.string(),
  title: z.string(),
  slug: z.string().optional(),
  sourceUrl: z.string().optional(),
});

export const StatementSchema = z.object({
  language: z.string(),
  format: z.enum(["md", "html", "pdf", "plain"]),
  contentPath: z.string(),
  contentHash: z.string(),
});

export const TestcaseSchema = z.object({
  id: z.string(),
  name: z.string(),
  group: TestcaseGroupSchema,
  inputPath: z.string(),
  answerPath: z.string().optional(),
});

export const ArtifactSchema = z.object({
  id: z.string(),
  language: z.string().optional(),
  path: z.string(),
});

export const GenerationReportSchema = z.object({
  problemId: z.string(),
  runId: z.string(),
  summary: z.partialRecord(JudgeVerdictSchema, z.number()),
  testcases: z.array(
    z.object({
      testcaseId: z.string(),
      name: z.string(),
      verdict: JudgeVerdictSchema,
      timeMs: z.number(),
      wallTimeMs: z.number(),
      memoryKb: z.number(),
      message: z.string().optional(),
    }),
  ),
});

export const ProblemBundleSchema = z.object({
  metadata: ProblemMetadataSchema,
  statements: z.array(StatementSchema),
  constraints: z.object({ items: z.array(z.unknown()) }),
  samples: z.array(TestcaseSchema),
  secrets: z.array(TestcaseSchema),
  generators: z.array(ArtifactSchema),
  validators: z.array(ArtifactSchema),
  solutions: z.array(ArtifactSchema),
  reports: z.array(GenerationReportSchema),
});
