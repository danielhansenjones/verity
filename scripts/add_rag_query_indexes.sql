-- Add the indexes backing the rag_queries audit-log read paths:
--   - composite (job_id, created_at DESC): "this job's queries, newest first"
--     and plain WHERE job_id = ? lookups (job_id is the leading column)
--   - (outcome): the admin listing's outcome filter
--   - (created_at DESC): the unfiltered created_at-DESC keyset listing
--
-- Idempotent: CREATE INDEX IF NOT EXISTS converges with a fresh
-- Base.metadata.create_all bootstrap, which builds the same three indexes from
-- RagQuery.__table_args__. Safe to run against an empty or a populated table,
-- and safe to re-run.

BEGIN;

CREATE INDEX IF NOT EXISTS ix_rag_queries_job_id_created_at
  ON rag_queries (job_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_rag_queries_outcome
  ON rag_queries (outcome);

CREATE INDEX IF NOT EXISTS ix_rag_queries_created_at
  ON rag_queries (created_at DESC);

-- The composite's leading job_id column now serves WHERE job_id = ? lookups, so
-- the standalone index from the original mapped_column(index=True) is redundant.
-- Drop it to match a fresh bootstrap, which no longer creates it.
DROP INDEX IF EXISTS ix_rag_queries_job_id;

COMMIT;
