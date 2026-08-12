"""
Storage backend abstract base class.
Defines the unified interface for WorkflowRun persistence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..core.models import WorkflowRun, WorkflowStatus, JobInstance, StepInstance


class StorageBackend(ABC):
    """
    Abstract base class for WorkflowRun storage backends.
    All backends (memory/file/database) implement this interface,
    decoupling the engine from storage.

    Fine-grained write interfaces (reduce redundant requests):
        save_run_status      - update only top-level run status fields
        save_job_instance    - update a single JobInstance
        save_step_instance   - update a single StepInstance
        flush_pending        - wait for all pending background writes (optional)

    These methods provide default implementations (degrade to full save_run).
    High-performance backends (e.g. Supabase) can override for true
    fine-grained writes and async queues.
    """

    # -- Write operations (full) --

    @abstractmethod
    def save_run(self, run: WorkflowRun) -> None:
        """
        Create or update a WorkflowRun record (full, including jobs/steps).
        If run.id already exists, overwrite; otherwise create new.
        """

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """
        Delete the specified run record.
        :returns: True if deleted, False if record does not exist
        """

    # -- Fine-grained write interfaces (optional override) --

    def save_run_status(self, run: WorkflowRun) -> None:
        """
        Persist only WorkflowRun's top-level status (status / start_time / end_time / env).
        Does not recursively save jobs/steps. Suitable for workflow status changes.
        Default: degrades to full save_run (compatible with all backends).
        """
        self.save_run(run)

    def save_job_instance(self, run_id: str, job: JobInstance) -> None:
        """
        Persist only a single JobInstance (status / start_time / end_time / outputs).
        Does not recursively save steps. Suitable for job status changes.
        Default: read run from storage, update job, full save.
        """
        run = self.get_run(run_id)
        if run:
            # Find the job key by matching job.id or job_def.id
            job_key = None
            for k, j in run.jobs.items():
                if j.id == job.id or (j.job_def and job.job_def and j.job_def.id == job.job_def.id):
                    job_key = k
                    break
            if job_key:
                run.jobs[job_key] = job
                self.save_run(run)

    def save_step_instance(self, run_id: str, job_id: str, step: StepInstance) -> None:
        """
        Persist only a single StepInstance (status / log / outputs / ...).
        Suitable for step status changes and log appends.
        Default: read run from storage, update step, full save.
        """
        run = self.get_run(run_id)
        if run:
            job = run.jobs.get(job_id)
            if job:
                for i, s in enumerate(job.steps):
                    if s.id == step.id:
                        job.steps[i] = step
                        break
                self.save_run(run)

    def flush_pending(self) -> None:
        """
        Wait for all pending background write tasks to complete.
        Useful before process shutdown or test assertions.
        Default: no-op (synchronous backends need no flush).
        """

    # -- Read operations --

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        """Get a single record by run_id, return None if not found."""

    @abstractmethod
    def list_runs(
        self,
        workflow_name: Optional[str]       = None,
        status:        Optional[WorkflowStatus] = None,
        limit:         int                 = 100,
        offset:        int                 = 0,
    ) -> List[WorkflowRun]:
        """
        Query WorkflowRun list, with optional workflow name and status filters,
        paginated. Results sorted by start_time descending (newest first).
        """

    @abstractmethod
    def count_runs(
        self,
        workflow_name: Optional[str]            = None,
        status:        Optional[WorkflowStatus] = None,
    ) -> int:
        """Count total records matching the criteria."""

    # -- Optional: bulk operations --

    def bulk_save(self, runs: List[WorkflowRun]) -> None:
        """Bulk save (default: call save_run per item, subclasses can override)."""
        for run in runs:
            self.save_run(run)

    def clear(self) -> None:
        """Clear all records (test/dev use)."""
        raise NotImplementedError("This storage backend does not support clear()")

    # -- Lifecycle --

    def open(self) -> None:
        """Initialize storage (connect DB / open files etc.), optional."""

    def close(self) -> None:
        """Close storage (disconnect / flush etc.), optional."""

    def __enter__(self) -> "StorageBackend":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
