-- Drop the RETRYING value from the jobstatus enum.
--
-- The value was set transiently in the worker between a stage failure and the
-- re-enqueue, but never observed anywhere downstream and immediately superseded
-- by RUNNING on the next dequeue. retry_count is the real signal.
--
-- Postgres has no DROP VALUE for enums, so this rebuilds the type. Safe only
-- when no rows currently hold the value; the DO block enforces that.

BEGIN;

DO $$
DECLARE
  stale_count integer;
BEGIN
  SELECT count(*) INTO stale_count FROM jobs WHERE status::text = 'RETRYING';
  IF stale_count > 0 THEN
    RAISE EXCEPTION 'cannot drop RETRYING: % rows still hold this status', stale_count;
  END IF;
END$$;

ALTER TYPE jobstatus RENAME TO jobstatus_old;

CREATE TYPE jobstatus AS ENUM ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED');

ALTER TABLE jobs
  ALTER COLUMN status DROP DEFAULT,
  ALTER COLUMN status TYPE jobstatus USING status::text::jobstatus;

DROP TYPE jobstatus_old;

COMMIT;
