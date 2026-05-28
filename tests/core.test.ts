import { expect, test } from "bun:test";
import {
  GenerationReportSchema,
  ProblemBundleSchema,
} from "../packages/core";

test("ProblemBundleSchema accepts a minimal empty bundle", () => {
  const parsed = ProblemBundleSchema.parse({
    metadata: { id: "p1", title: "A + B" },
    statements: [],
    constraints: { items: [] },
    samples: [],
    secrets: [],
    generators: [],
    validators: [],
    solutions: [],
    reports: [],
  });

  expect(parsed.metadata.id).toBe("p1");
});

test("GenerationReportSchema validates verdict values", () => {
  const report = GenerationReportSchema.parse({
    problemId: "p1",
    runId: "r1",
    summary: { AC: 1 },
    testcases: [
      {
        testcaseId: "t1",
        name: "sample-1",
        verdict: "AC",
        timeMs: 1,
        wallTimeMs: 2,
        memoryKb: 128,
      },
    ],
  });

  expect(report.testcases[0]?.verdict).toBe("AC");
});
