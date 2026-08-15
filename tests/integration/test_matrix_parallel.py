"""M4.3: integration test verifying the 3 dimension calls happen in parallel.

The matrix strategy in ``job_executor._execute_matrix`` submits all matrix
combinations to a ``ThreadPoolExecutor(max_workers=max-parallel)``. This test
asserts that behaviour end-to-end: a workflow with a 3-way ``matrix`` job must
issue all 3 LLM calls within a single 0.5 s latency window. If the calls were
serial we would observe ~1.5 s of wall time instead.

The matrix executor injects ``MATRIX_<KEY>`` env vars into each sub-job (see
``JobExecutor._execute_matrix_job``); the test action reads the dimension from
``env["MATRIX_DIMENSION"]`` so it works without ``${{ }}`` substitution in
``with`` params (the step executor does not yet resolve expressions in
``with``).
"""
import time
import pytest

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.actions.base import BaseAction, ActionResult


class TimingDimensionAction(BaseAction):
    """Stand-in for DimensionAction that records call times and sleeps.

    Reads the dimension from the MATRIX_DIMENSION env var injected by the
    matrix executor (avoids relying on ``${{ }}`` substitution in ``with``
    params, which the step executor does not perform).
    """

    def __init__(self, call_times: list, sleep_seconds: float = 0.5):
        self._call_times = call_times
        self._sleep_seconds = sleep_seconds

    @property
    def description(self) -> str:
        return "Timing stand-in for DimensionAction (matrix parallelism test)"

    def run(self, params, env, context, log_callback=None):
        dimension = env.get("MATRIX_DIMENSION")
        if not dimension:
            return ActionResult(success=False, message="MATRIX_DIMENSION env var not set")

        # record start time, simulate LLM latency
        self._call_times.append(time.time())
        time.sleep(self._sleep_seconds)

        if log_callback:
            log_callback(f"[{dimension}] scored (mock)")

        return ActionResult(
            success=True,
            outputs={
                "score": 7,
                "confidence": 0.8,
                "dimension": dimension,
            },
            log_lines=[f"[{dimension}] score=7"],
        )


@pytest.fixture
def fake_paper_session(tmp_path):
    """Pre-create extract outputs so the dimensions job has inputs to read."""
    session_dir = tmp_path / "session"
    extract_dir = session_dir / "00_extract"
    extract_dir.mkdir(parents=True)
    (extract_dir / "full_text.md").write_text("# Paper\n\nContent. " * 50)
    (extract_dir / "metadata.json").write_text('{"title":"T","authors":[],"abstract":""}')
    return session_dir


def test_3_dimensions_run_in_parallel(fake_paper_session, tmp_path):
    """Verify 3 LLM calls happen in parallel, not serially.

    With max-parallel=3 and a 0.5s simulated LLM latency per call:
    - Parallel: all 3 calls start within ~0.5s (one sleep cycle)
    - Serial: would take ~1.5s (3 * 0.5s)
    """
    call_times: list = []

    # Register the timing action under the dim_score name so the workflow's
    # `uses: paper-review/dim_score@v1` resolves to our stand-in.
    from paper_review_workflow.actions.registry import ActionRegistry
    ActionRegistry._instance = None

    engine = ReviewEngine(storage=MemoryStorage())
    engine.registry.register(
        "paper-review/dim_score@v1",
        TimingDimensionAction(call_times=call_times, sleep_seconds=0.5),
    )

    yaml = tmp_path / "test.yaml"
    yaml.write_text(f"""
name: parallel-test
on: {{workflow_dispatch: {{}}}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSION_DIR: "{fake_paper_session}"
  VENUE: neurips
jobs:
  dimensions:
    runs-on: local
    strategy:
      matrix:
        dimension: [soundness, presentation, contribution]
      max-parallel: 3
    steps:
      - uses: paper-review/dim_score@v1
        with:
          session_dir: "${{ env.SESSION_DIR }}"
""")

    run = engine.run_from_file(str(yaml), payload={})

    assert run.status.value == "success", (
        f"workflow should succeed; got {run.status.value}"
    )
    assert len(call_times) == 3, (
        f"expected 3 LLM calls, got {len(call_times)}"
    )
    # If parallel, all 3 calls start within ~0.5s (one sleep cycle).
    # If serial, the spread would be ~1.5s.
    elapsed = max(call_times) - min(call_times)
    assert elapsed < 1.0, (
        f"calls not parallel: elapsed={elapsed:.2f}s "
        f"(parallel would be <0.5s, serial would be ~1.5s)"
    )


def test_dimensions_run_serially_when_max_parallel_is_1(fake_paper_session, tmp_path):
    """Sanity check: with max-parallel=1, the same 3 calls are serial.

    This validates that the parallel assertion in the test above is actually
    detecting parallelism (not just fast execution).
    """
    call_times: list = []

    from paper_review_workflow.actions.registry import ActionRegistry
    ActionRegistry._instance = None

    engine = ReviewEngine(storage=MemoryStorage())
    engine.registry.register(
        "paper-review/dim_score@v1",
        TimingDimensionAction(call_times=call_times, sleep_seconds=0.3),
    )

    yaml = tmp_path / "test_serial.yaml"
    yaml.write_text(f"""
name: serial-test
on: {{workflow_dispatch: {{}}}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSION_DIR: "{fake_paper_session}"
  VENUE: neurips
jobs:
  dimensions:
    runs-on: local
    strategy:
      matrix:
        dimension: [soundness, presentation, contribution]
      max-parallel: 1
    steps:
      - uses: paper-review/dim_score@v1
        with:
          session_dir: "${{ env.SESSION_DIR }}"
""")

    run = engine.run_from_file(str(yaml), payload={})

    assert run.status.value == "success"
    assert len(call_times) == 3
    # With max-parallel=1, calls are serial: spread should be >= 2 * 0.3s
    elapsed = max(call_times) - min(call_times)
    assert elapsed >= 0.5, (
        f"calls should be serial (>=0.5s spread), got {elapsed:.2f}s"
    )


def test_matrix_with_param_resolution(fake_paper_session, tmp_path):
    """Verify ${{ matrix.X }} in with params is resolved correctly."""
    from paper_review_workflow.actions.registry import ActionRegistry
    from paper_review_workflow.engine import ReviewEngine
    from paper_review_workflow.storage.memory import MemoryStorage
    from paper_review_workflow.llm.base import LLMResponse
    from unittest.mock import MagicMock, patch

    ActionRegistry._instance = None
    engine = ReviewEngine(storage=MemoryStorage())

    # Re-register dimension action (was reset above)
    from paper_review_workflow.actions.builtin import register_builtin_actions
    register_builtin_actions(engine.registry)

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
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 30000}
    fake_response.model = "test-model"

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.complete.return_value = fake_response
        mock_client.model = "claude-sonnet-4-6"
        mock_from_env.return_value = mock_client

        yaml = tmp_path / "with_param_test.yaml"
        yaml.write_text(f"""
name: with-param-test
on: {{workflow_dispatch: {{}}}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSION_DIR: "{fake_paper_session}"
  VENUE: neurips
jobs:
  dimensions:
    runs-on: local
    strategy:
      matrix:
        dimension: [soundness, presentation]
      max-parallel: 2
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{{{ matrix.dimension }}}}
          session_dir: "${{{{ env.SESSION_DIR }}}}"
          full_text_path: "${{{{ env.SESSION_DIR }}}}/00_extract/full_text.md"
          metadata_path: "${{{{ env.SESSION_DIR }}}}/00_extract/metadata.json"
""")

        run = engine.run_from_file(str(yaml), payload={})

    assert run.status.value == "success", f"run failed: {run.status.value}"
    # Verify both dimensions completed
    assert "dimensions_soundness" in run.jobs
    assert "dimensions_presentation" in run.jobs
    assert run.jobs["dimensions_soundness"].status.value == "success"
    assert run.jobs["dimensions_presentation"].status.value == "success"
