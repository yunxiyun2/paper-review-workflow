"""M6.3: integration tests for cancel_run and graceful shutdown.

Verifies:
1. ``engine.cancel_run(run_id)`` marks an actively running workflow as
   CANCELLED via the state machine, persists the final state to
   ``JsonFileStorage``, and unblocks the blocking ``run_from_file`` call
   running in a worker thread.
2. ``engine.shutdown(timeout)`` cancels every active run and returns the
   count cancelled -- this is the path the SIGINT/SIGTERM signal handlers
   in ``cli.py`` call into for graceful shutdown.

The engine's cancel machinery (coordinator registered via
``on_coordinator_created``, ``cancel_run`` + ``shutdown``) was built in M1.8
(with the C1 fix applied during the M1.8 review). These tests are
test-only; no production code changes are expected here.
"""
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.llm.base import LLMResponse
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.actions.registry import ActionRegistry


@pytest.fixture
def fake_paper():
    return str(Path("tests/fixtures/sample_paper.pdf"))


def test_cancel_marks_run_as_cancelled(fake_paper, tmp_path, monkeypatch):
    """cancel_run(run_id) cancels an active workflow and persists state."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    session_dir = tmp_path / "session"

    # Use sentinel placeholders instead of f-string brace escaping for the
    # ${{ ... }} expression syntax -- the same approach used by
    # test_full_review_mocked.py and test_resume.py.
    yaml_template = """\
name: test-cancel
on: {workflow_dispatch: {}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSIONS_ROOT: __SESSIONS_ROOT__
  VENUE: neurips
jobs:
  extract:
    runs-on: local
    outputs:
      paper_id: ${{ steps.extract.outputs.paper_id }}
      full_text_path: ${{ steps.extract.outputs.full_text_path }}
      metadata_path: ${{ steps.extract.outputs.metadata_path }}
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: __FAKE_PAPER__
          session_dir: __SESSION_DIR__
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [soundness, presentation, contribution]
      max-parallel: 3
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: __SESSION_DIR__
          full_text_path: ${{ needs.extract.outputs.full_text_path }}
          metadata_path: ${{ needs.extract.outputs.metadata_path }}
"""
    yaml_content = yaml_template
    yaml_content = yaml_content.replace("__FAKE_PAPER__", fake_paper)
    yaml_content = yaml_content.replace("__SESSION_DIR__", str(session_dir))
    yaml_content = yaml_content.replace("__SESSIONS_ROOT__", str(tmp_path))

    yaml = tmp_path / "test_cancel.yaml"
    yaml.write_text(yaml_content)

    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))

    fake_score_obj = MagicMock()
    fake_score_obj.score = 7
    fake_score_obj.confidence = 0.8
    fake_score_obj.strengths = ["a"]
    fake_score_obj.weaknesses = ["b"]
    fake_score_obj.justification = "x" * 200
    fake_score_obj.evidence = []

    fake_response = MagicMock(spec=LLMResponse)
    fake_response.structured = fake_score_obj
    fake_response.usage = {"input_tokens": 100, "output_tokens": 50,
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 30000}
    fake_response.model = "test-model"

    # Reset singleton so builtin actions (real DimensionAction etc.) are
    # re-registered cleanly -- previous tests may have overridden dim_score@v1.
    ActionRegistry._instance = None

    started = threading.Event()
    run_id_holder = {}

    def slow_score(**kwargs):
        # Signal that the dim LLM call has started, then block long enough
        # for the main thread to observe the run as active and cancel it.
        started.set()
        time.sleep(2)
        return fake_response

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.complete.side_effect = slow_score
        mock_from_env.return_value = mock_client

        engine = ReviewEngine(storage=storage)

        # run_from_file is blocking; run it in a worker thread so the main
        # thread can call cancel_run while the workflow is mid-flight.
        def run_in_thread():
            run = engine.run_from_file(str(yaml), payload={})
            run_id_holder["run"] = run

        t = threading.Thread(target=run_in_thread)
        t.start()

        # Wait for the dim LLM call to actually start -- this proves the
        # workflow has reached the dimensions job and is actively running.
        assert started.wait(timeout=10), (
            "LLM call did not start within 10s -- workflow never reached "
            "the dimensions job, cancel could not be observed mid-flight"
        )

        # Give the engine a beat to register the run in _active_runs.
        # started.set() fires inside the LLM mock, which runs inside the
        # matrix job's step executor -- by this point _active_runs[run.id]
        # is already populated (it's set in _do_execute before execute()).
        time.sleep(0.3)
        assert len(engine._active_runs) == 1, (
            f"expected 1 active run during execution, got "
            f"{len(engine._active_runs)}"
        )
        run_id = list(engine._active_runs.keys())[0]

        # Cancel the active run. The coordinator (registered via
        # on_coordinator_created when execution started) transitions the
        # workflow to CANCELLED and cancels all pending jobs/steps. The
        # slow LLM call is still blocking the step executor thread, but
        # once cancel_workflow() flips the state machine to terminal the
        # matrix executor's post-step gate short-circuits.
        cancelled = engine.cancel_run(run_id)
        assert cancelled, (
            "cancel_run should return True for an active, non-terminal run"
        )

        # The workflow thread should now terminate (the slow LLM call
        # returns its response, but the post-step gate sees the workflow
        # is terminal and stops scheduling further matrix sub-jobs).
        t.join(timeout=10)
        assert not t.is_alive(), (
            "worker thread should have terminated after cancel"
        )

    run = run_id_holder["run"]
    assert run.status.value == "cancelled", (
        f"expected run status 'cancelled', got {run.status.value!r}"
    )

    # Verify the cancelled state was persisted to JsonFileStorage -- the
    # engine's persist hook (subscribed to WorkflowEvent) calls save_run
    # on every transition, including the CANCELLED transition.
    loaded = storage.get_run(run.id)
    assert loaded is not None, "cancelled run should be persisted to storage"
    assert loaded.status.value == "cancelled", (
        f"persisted run status should be 'cancelled', got "
        f"{loaded.status.value!r}"
    )


def test_shutdown_cancels_all_active_runs(fake_paper, tmp_path, monkeypatch):
    """engine.shutdown() cancels every active run and returns the count.

    This is the path called by the SIGINT/SIGTERM handlers in cli.py for
    graceful shutdown. We simulate an active run by inserting a fake
    WorkflowRun + coordinator into the engine's internal dicts (the same
    state the engine itself would produce mid-execution) and verify
    shutdown tears it down cleanly.
    """
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    yaml = tmp_path / "test_shutdown.yaml"
    yaml.write_text(
        "name: test-shutdown\n"
        "on: {workflow_dispatch: {}}\n"
        "env:\n"
        "  LLM_PROVIDER: anthropic\n"
        "  LLM_MODEL: test-model\n"
        "jobs:\n"
        "  slow:\n"
        "    runs-on: local\n"
        "    steps:\n"
        "      - uses: paper-review/echo@v1\n"
        "        with:\n"
        "          message: hi\n"
    )

    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))

    # Reset singleton so builtin actions are re-registered cleanly.
    ActionRegistry._instance = None

    engine = ReviewEngine(storage=storage)

    # Simulate an active run by adding to _active_runs (test-only).
    # In normal operation these dicts are populated by _do_execute before
    # the workflow reaches a blocking step; here we inject the same state
    # directly so we can test shutdown() in isolation without racing the
    # engine's executor thread.
    from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus
    from paper_review_workflow.core.event_bus import EventBus
    from paper_review_workflow.core.state_machine import (
        WorkflowStateMachineCoordinator,
    )

    fake_run = WorkflowRun(status=WorkflowStatus.PENDING)
    # Move the run into RUNNING so cancel_workflow() has a valid transition
    # (PENDING -> RUNNING -> CANCELLED per the workflow transition table).
    fake_coord = WorkflowStateMachineCoordinator(fake_run, EventBus())
    fake_coord.start_workflow()
    engine._active_runs[fake_run.id] = fake_run
    engine._coordinators[fake_run.id] = fake_coord

    cancelled = engine.shutdown(timeout=5)
    assert cancelled == 1, (
        f"expected shutdown to cancel 1 active run, got {cancelled}"
    )
    assert fake_run.status.value == "cancelled", (
        f"expected fake run to be cancelled, got {fake_run.status.value!r}"
    )
