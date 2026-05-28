import type { ProblemBundle } from "./problem";

export interface ExportOptions {
  outputPath: string;
}

export interface ExportResult {
  format: string;
  outputPath: string;
  files: string[];
  warnings: string[];
}

export interface ExporterPlugin {
  id: string;
  displayName: string;
  export(bundle: ProblemBundle, options: ExportOptions): Promise<ExportResult>;
}
