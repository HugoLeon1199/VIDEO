"""
Low-level RunPod Serverless Queue API client.

Handles async job submission, status polling, exponential backoff,
cancellation, and secret redaction in logs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from typing import Any, Optional

import httpx

from image_generation.exceptions import (
    BackendUnavailableError,
    JobFailedError,
    JobTimeoutError,
)

logger = logging.getLogger(__name__)

# Terminal job states
_DONE_STATES = {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}
_SUCCESS_STATE = "COMPLETED"


def _redact(key: str) -> str:
    """Return last 4 chars of a secret for safe logging."""
    return f"...{key[-4:]}" if len(key) > 4 else "****"


class RunPodClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        timeout: int = 1800,
        poll_interval: float = 3.0,
        max_retries: int = 3,
    ):
        self._api_key = api_key or os.environ["RUNPOD_API_KEY"]
        self._endpoint_id = endpoint_id or os.environ["RUNPOD_ENDPOINT_ID"]
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._max_retries = max_retries
        self._base = f"https://api.runpod.ai/v2/{self._endpoint_id}"
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        logger.debug("RunPodClient endpoint=%s key=%s", self._endpoint_id, _redact(self._api_key))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, job_input: dict) -> str:
        """Submit a job asynchronously. Returns job_id."""
        url = f"{self._base}/run"
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = httpx.post(url, json={"input": job_input}, headers=self._headers, timeout=30)
                resp.raise_for_status()
                job_id = resp.json()["id"]
                logger.info("Job submitted: %s (attempt %d)", job_id, attempt)
                return job_id
            except httpx.HTTPStatusError as e:
                logger.warning("Submit HTTP %d attempt %d: %s", e.response.status_code, attempt, e)
            except httpx.RequestError as e:
                logger.warning("Submit network error attempt %d: %s", attempt, e)
            if attempt < self._max_retries:
                self._backoff(attempt)
        raise BackendUnavailableError(f"Failed to submit job after {self._max_retries} attempts")

    def poll_until_done(self, job_id: str) -> dict:
        """
        Poll /status/{job_id} until terminal state.
        Returns the full status response dict.
        Raises JobTimeoutError or JobFailedError on bad outcomes.
        """
        url = f"{self._base}/status/{job_id}"
        deadline = time.monotonic() + self._timeout
        poll = self._poll_interval

        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, headers=self._headers, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "UNKNOWN")
                logger.debug("Job %s status=%s", job_id, status)

                if status in _DONE_STATES:
                    if status != _SUCCESS_STATE:
                        err = data.get("error", status)
                        raise JobFailedError(f"Job {job_id} ended with status={status}: {err}")
                    return data

                # Exponential backoff capped at 30s
                time.sleep(min(poll, 30.0))
                poll = min(poll * 1.5, 30.0)

            except (JobFailedError, JobTimeoutError):
                raise
            except httpx.RequestError as e:
                logger.warning("Poll network error for %s: %s", job_id, e)
                time.sleep(self._poll_interval)

        self.cancel(job_id)
        raise JobTimeoutError(f"Job {job_id} timed out after {self._timeout}s")

    def cancel(self, job_id: str) -> None:
        """Cancel a running job."""
        url = f"{self._base}/cancel/{job_id}"
        try:
            resp = httpx.post(url, headers=self._headers, timeout=15)
            resp.raise_for_status()
            logger.info("Job %s cancelled.", job_id)
        except httpx.HTTPError as e:
            logger.warning("Failed to cancel job %s: %s", job_id, e)

    def run_sync(self, job_input: dict) -> dict:
        """Submit and block until done. Convenience wrapper."""
        job_id = self.submit(job_input)
        return self.poll_until_done(job_id)

    def health_check(self) -> bool:
        """Ping the endpoint to verify connectivity."""
        url = f"{self._base}/health"
        try:
            resp = httpx.get(url, headers=self._headers, timeout=10)
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _backoff(attempt: int) -> None:
        delay = (2 ** attempt) + random.uniform(0, 1)
        logger.debug("Backoff %.1fs before retry", delay)
        time.sleep(delay)
