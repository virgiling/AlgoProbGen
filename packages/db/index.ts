import { Database, type SQLQueryBindings } from "bun:sqlite";
import { SCHEMA } from "./schema";

export const DEFAULT_DATABASE_PATH = process.env.DB_NAME ?? "algogen.sqlite";

export type DBValue = SQLQueryBindings;
export type DBConnection = Database;

export interface OpenDatabaseOptions {
  readonly create?: boolean;
  readonly strict?: boolean;
}

export function createDatabase(
  path: string = DEFAULT_DATABASE_PATH,
  options: OpenDatabaseOptions = {},
): Database {
  return new Database(path, {
    create: options.create ?? true,
    strict: options.strict ?? true,
  });
}

export function initDatabase(db: DBConnection): void {
  db.run("PRAGMA journal_mode = WAL");
  db.run("PRAGMA foreign_keys = ON");
  db.run(SCHEMA);
}

export function openDatabase(
  path: string = DEFAULT_DATABASE_PATH,
  options: OpenDatabaseOptions = {},
): Database {
  const database = createDatabase(path, options);
  initDatabase(database);
  return database;
}

let defaultDb: Database | undefined;

export function getDb(): Database {
  defaultDb ??= openDatabase(DEFAULT_DATABASE_PATH);
  return defaultDb;
}
