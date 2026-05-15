-- One-shot DDL migration to bring the existing local Postgres in line with the
-- post-alembic-removal model definitions in shared/models.py.
--
-- Idempotent where it can be (CREATE INDEX IF NOT EXISTS, DROP TABLE IF EXISTS).
-- The ALTER COLUMN TYPE and ADD CONSTRAINT statements are not; re-running this
-- after a successful first pass will error on the duplicate constraint, which
-- is the intended signal that the migration has already been applied.

BEGIN;

-- Refuse to add the unique constraint if existing rows would violate it.
DO $$
DECLARE
  dup_count integer;
BEGIN
  SELECT count(*) INTO dup_count FROM (
    SELECT job_id, index FROM chunks GROUP BY job_id, index HAVING count(*) > 1
  ) t;
  IF dup_count > 0 THEN
    RAISE EXCEPTION 'cannot add unique constraint: % duplicate (job_id, index) rows', dup_count;
  END IF;
END$$;

-- Naive timestamps were written by datetime.now(timezone.utc), so reinterpret
-- them as UTC when widening to timestamptz.
ALTER TABLE jobs
  ALTER COLUMN created_at TYPE timestamptz USING created_at AT TIME ZONE 'UTC',
  ALTER COLUMN updated_at TYPE timestamptz USING updated_at AT TIME ZONE 'UTC';
ALTER TABLE chunks
  ALTER COLUMN created_at TYPE timestamptz USING created_at AT TIME ZONE 'UTC';
ALTER TABLE job_dedup
  ALTER COLUMN created_at TYPE timestamptz USING created_at AT TIME ZONE 'UTC';
ALTER TABLE risk_results
  ALTER COLUMN created_at TYPE timestamptz USING created_at AT TIME ZONE 'UTC';

-- Composite unique serves both as the integrity constraint and as the b-tree
-- index for WHERE job_id = ? queries (job_id is the leading column).
ALTER TABLE chunks
  ADD CONSTRAINT chunks_job_id_index_uq UNIQUE (job_id, index);

-- SQLAlchemy's default index name for mapped_column(index=True).
CREATE INDEX IF NOT EXISTS ix_job_dedup_job_id ON job_dedup (job_id);

-- Orphan from the alembic era.
DROP TABLE IF EXISTS alembic_version;

COMMIT;