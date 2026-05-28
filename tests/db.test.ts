import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test } from "bun:test";
import { openDatabase } from "../packages/db";
import { CORE_TABLES } from "../packages/db/schema";

function tempDatabasePath(): string {
  return join(mkdtempSync(join(tmpdir(), "algogen-db-")), "test.sqlite");
}

function tableNames(db: ReturnType<typeof openDatabase>): string[] {
  return db
    .query<{ name: string }, []>(
      `SELECT name
       FROM sqlite_schema
       WHERE type = 'table'
       ORDER BY name ASC`,
    )
    .all()
    .map((row) => row.name);
}

test("openDatabase initializes sqlite pragmas and core tables", () => {
  const db = openDatabase(tempDatabasePath());

  const journal = db
    .query<{ journal_mode: string }, []>("PRAGMA journal_mode")
    .get();
  const foreignKeys = db
    .query<{ foreign_keys: number }, []>("PRAGMA foreign_keys")
    .get();
  const tables = new Set(tableNames(db));

  expect(journal?.journal_mode).toBe("wal");
  expect(foreignKeys?.foreign_keys).toBe(1);

  for (const table of CORE_TABLES) {
    expect(tables.has(table)).toBe(true);
  }

  db.close();
});

test("schema initialization is idempotent", () => {
  const path = tempDatabasePath();
  const first = openDatabase(path);
  const firstTables = tableNames(first);
  first.close();

  const second = openDatabase(path);
  expect(tableNames(second)).toEqual(firstTables);
  second.close();
});
