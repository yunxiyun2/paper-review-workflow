import pytest
from pydantic import ValidationError
from paper_review_workflow.llm.schemas import (
    DimensionScore, PaperMetadata, SynthesisResult,
)


def test_dimension_score_valid():
    s = DimensionScore(
        score=4, confidence=0.85,
        strengths=["a", "b"], weaknesses=["c"],
        justification="x" * 200,
        evidence=[{"section": "3.2", "quote": "...", "page": 5}],
    )
    assert s.score == 4
    assert s.confidence == 0.85


def test_dimension_score_out_of_range_high():
    with pytest.raises(ValidationError):
        DimensionScore(score=6, confidence=0.5, strengths=["a"],
                       weaknesses=["b"], justification="x" * 200)


def test_dimension_score_out_of_range_low():
    with pytest.raises(ValidationError):
        DimensionScore(score=0, confidence=0.5, strengths=["a"],
                       weaknesses=["b"], justification="x" * 200)


def test_dimension_score_justification_too_short():
    with pytest.raises(ValidationError):
        DimensionScore(score=3, confidence=0.5, strengths=["a"],
                       weaknesses=["b"], justification="too short")


def test_dimension_score_confidence_range():
    with pytest.raises(ValidationError):
        DimensionScore(score=3, confidence=1.5, strengths=["a"],
                       weaknesses=["b"], justification="x" * 200)


def test_paper_metadata_minimal():
    m = PaperMetadata(title="T", authors=["A"], abstract="abs")
    assert m.doi is None
    assert m.arxiv_id is None


def test_synthesis_result_valid():
    s = SynthesisResult(
        summary="x" * 200,
        key_strengths=["a"], key_weaknesses=["b"],
        questions_for_authors=["q1"],
        overall_assessment="good",
    )
    assert s.summary.startswith("x")
