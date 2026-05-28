import type { ProblemBundle, TestcaseGroup } from "./problem";

export interface GeneratorContext<TParams = unknown> {
  problem: ProblemBundle;
  params: TParams;
  seed: number;
}

export interface GeneratedTestcase {
  name: string;
  group: TestcaseGroup;
  input: string;
  description?: string;
}

export abstract class GeneratorBase<TParams = unknown> {
  abstract id: string;
  abstract version: string;
  abstract generate(
    ctx: GeneratorContext<TParams>,
  ): AsyncIterable<GeneratedTestcase>;
}
