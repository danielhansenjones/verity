"""
Locust load tests for the contract-risk pipeline.

Three user classes target different scenarios:

  SmokeUser  - 1 VU, 1 job, full end-to-end (submit -> poll -> report). Pass/fail.
  LoadUser   - 3 VUs, sustained throughput, 5-10 s idle between iterations.
  StressUser - 20 VUs, submit-only flood. Validates rate limiting and API stability.

Each run ends with a shutdown block followed by a percentile table, e.g.:

  [timestamp] INFO/locust.main: --run-time limit reached, shutting down
  [timestamp] INFO/locust.main: Shutting down (exit code 0)
  Type     Name          # reqs  # fails | Avg  Min  Max  Med | req/s  failures/s
  ---------|------------|-------|--------|-----|-----|-----|-----|------|----------
  POST     /jobs [submit]   ...
  job      completion       ...
  ...
  Response time percentiles (approximated)
  Type     Name              50%   66%   75% ...
  POST     /jobs [submit]    ...

Usage (requires: uv sync --group dev, or pip install locust):

  # Smoke - 1 job end-to-end, exits automatically after the job completes
  locust -f locustfile.py SmokeUser --headless -u 1 -r 1 --host http://localhost:8000

  # Load - 3 concurrent users, 5 min total; console stays active throughout
  locust -f locustfile.py LoadUser --headless -u 3 -r 1 \
    --run-time 5m --host http://localhost:8000

  # Stress - 20 VUs flooding submit for 1 min
  locust -f locustfile.py StressUser --headless -u 20 -r 20 \
    --run-time 1m --host http://localhost:8000

  # Web UI - pick class and VU count interactively at http://localhost:8089
  locust -f locustfile.py --host http://localhost:8000

Set CONTRACT_API_KEY env var if API auth is enabled.
"""

import os
import time
import uuid

from locust import HttpUser, between, constant, events, task

_API_KEY = os.getenv("CONTRACT_API_KEY", "")
_POLL_INTERVAL_S = 2
_POLL_TIMEOUT_S = 600  # 10 minutes

with open("tests/test_documents/Document.pdf", "rb") as _f:
    _PDF_BYTES = _f.read()


class _PipelineUser(HttpUser):
    abstract = True

    def on_start(self):
        if _API_KEY:
            self.client.headers["X-API-Key"] = _API_KEY

    def _submit(self, idempotency_key: str):
        return self.client.post(
            "/jobs",
            files={"file": ("contract.pdf", _PDF_BYTES, "application/pdf")},
            headers={"Idempotency-Key": idempotency_key},
            name="/jobs [submit]",
        )

    def _poll_until_done(self, job_id: str) -> dict | None:
        deadline = time.time() + _POLL_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL_S)
            res = self.client.get(f"/jobs/{job_id}", name="/jobs/{id} [poll]")
            if res.status_code != 200:
                continue
            body = res.json()
            if body.get("status") in ("completed", "failed"):
                return body
        return None

    def _record_completion(self, duration_ms: float, ok: bool):
        events.request.fire(
            request_type="job",
            name="completion",
            response_time=duration_ms,
            response_length=0,
            exception=None if ok else Exception("job did not reach completed status"),
        )


class SmokeUser(_PipelineUser):
    """1 VU, 1 job end-to-end. Calls runner.quit() when done to flush stats."""
    wait_time = constant(0)

    @task
    def end_to_end(self):
        key = f"smoke-{uuid.uuid4()}"
        start = time.time()

        res = self._submit(key)
        if res.status_code not in (200, 201):
            self.environment.runner.quit()
            return

        job_id = res.json().get("job_id")
        if not job_id:
            self.environment.runner.quit()
            return

        job = self._poll_until_done(job_id)
        ok = job is not None and job.get("status") == "completed"
        self._record_completion((time.time() - start) * 1000, ok)

        if ok:
            self.client.get(f"/jobs/{job_id}/report", name="/jobs/{id}/report")

        self.environment.runner.quit()


class LoadUser(_PipelineUser):
    """Full end-to-end. 5-10 s idle staggers VUs so the console stays active."""
    wait_time = between(5, 10)

    @task
    def end_to_end(self):
        key = f"load-{uuid.uuid4()}"
        start = time.time()

        res = self._submit(key)
        if res.status_code not in (200, 201):
            self._record_completion((time.time() - start) * 1000, ok=False)
            return

        job_id = res.json().get("job_id")
        if not job_id:
            self._record_completion((time.time() - start) * 1000, ok=False)
            return

        job = self._poll_until_done(job_id)
        ok = job is not None and job.get("status") == "completed"
        self._record_completion((time.time() - start) * 1000, ok)

        if ok:
            self.client.get(f"/jobs/{job_id}/report", name="/jobs/{id}/report")


class StressUser(_PipelineUser):
    """Submit-only flood. Does not poll for completions."""
    wait_time = constant(0)

    @task(10)
    def submit_job(self):
        key = f"stress-{uuid.uuid4()}"
        with self.client.post(
            "/jobs",
            files={"file": ("contract.pdf", _PDF_BYTES, "application/pdf")},
            headers={"Idempotency-Key": key},
            name="/jobs [submit]",
            catch_response=True,
        ) as res:
            if res.status_code == 429:
                res.success()
                retry_after = int(res.headers.get("Retry-After", "1"))
                time.sleep(retry_after)

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
