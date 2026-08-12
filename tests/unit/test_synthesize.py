import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.actions.synthesize import SynthesizeAction
from paper_review_workflow.llm.schemas import SynthesisResult
from paper_review_workflow.llm.base import LLMResponse


@pytest.fixture
def session_with_8_dims(tmp_path):
    session_dir = tmp_path / "session"
    dims = ["novelty", "soundness", "significance", "clarity",
            "reproducibility", "related_work", "positioning", "presentation"]
    for dim in dims:
        d = session_dir / f"10_dim_{dim}"
        d.mkdir(parents=True)
        (d / "score.json").write_text(json.dumps({
            "dimension": dim, "score": 4, "confidence": 0.8,
            "strengths": ["x"], "weaknesses": ["y"],
            "justification": "z" * 200,
            "evidence": [], "model_used": "test", "usage": {},
        }))
    return session_dir


def test_synthesize_reads_8_dims_and_calls_llm(session_with_8_dims):
    action = SynthesizeAction()
    fake_synth = SynthesisResult(
        summary="x" * 250,
        key_strengths=["a"], key_weaknesses=["b"],
        questions_for_authors=["q1"],
        overall_assessment="good paper",
    )

    fake_response = MagicMock(spec=LLMResponse)
    fake_response.structured = fake_synth
    fake_response.usage = {"input_tokens": 100, "output_tokens": 50,
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    fake_response.model = "test-model"

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.complete.return_value = fake_response
        mock_client.model = "test-model"
        mock_from_env.return_value = mock_client

        result = action.run(
            params={"session_dir": str(session_with_8_dims)},
            env={}, context={}, log_callback=lambda x: None,
        )

    assert result.success
    assert (session_with_8_dims / "50_synthesize" / "review.md").exists()
    scores = json.loads((session_with_8_dims / "50_synthesize" / "scores.json").read_text())
    assert len(scores) == 8
    assert scores["novelty"]["score"] == 4


def test_synthesize_fails_with_too_many_missing(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    # Only 5 dims present (< 6 threshold)
    for dim in ["novelty", "soundness", "significance", "clarity", "reproducibility"]:
        d = session_dir / f"10_dim_{dim}"
        d.mkdir()
        (d / "score.json").write_text(json.dumps({"score": 4}))

    action = SynthesizeAction()
    result = action.run(
        params={"session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "missing" in result.message.lower()
