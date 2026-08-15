import pytest
from pathlib import Path
from pydantic import ValidationError
from paper_review_workflow.core.venue_config import VenueConfig


def test_venue_config_load_neurips():
    config = VenueConfig.load("neurips")
    assert config.name == "neurips"
    assert config.display_name == "NeurIPS 2025"
    assert config.dimensions == ["soundness", "presentation", "contribution"]
    assert config.score_min == 1
    assert config.score_max == 10
    assert config.weights["soundness"] == 1.3
    assert config.prompts_dir == "prompts/venues/neurips"


def test_venue_config_load_icml():
    config = VenueConfig.load("icml")
    assert config.name == "icml"
    assert config.dimensions == ["soundness", "significance", "originality", "clarity"]
    assert config.score_min == 1
    assert config.score_max == 4


def test_venue_config_load_acl():
    config = VenueConfig.load("acl")
    assert config.name == "acl"
    assert config.dimensions == ["soundness", "excitement", "reproducibility", "overall"]
    assert config.score_min == 1
    assert config.score_max == 4


def test_venue_config_load_unknown_raises():
    with pytest.raises(ValueError, match="venue not found"):
        VenueConfig.load("nonexistent")


def test_venue_config_thresholds_correct_neurips():
    config = VenueConfig.load("neurips")
    assert (9.0, "strong_accept") in config.thresholds
    assert (0.0, "strong_reject") in config.thresholds


def test_venue_config_load_is_cached():
    VenueConfig._cache = {}
    c1 = VenueConfig.load("neurips")
    c2 = VenueConfig.load("neurips")
    assert c1 is c2


def test_neurips_schema_score_range_1_to_10():
    config = VenueConfig.load("neurips")
    Schema = config.get_dimension_score_schema()
    # Score 5 (within 1-10) should pass
    s = Schema(score=5, confidence=0.8, strengths=["a"], weaknesses=["b"],
               justification="x" * 200, evidence=[])
    assert s.score == 5
    # Score 0 (below min) should fail
    with pytest.raises(ValidationError):
        Schema(score=0, confidence=0.5, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)
    # Score 11 (above max) should fail
    with pytest.raises(ValidationError):
        Schema(score=11, confidence=0.5, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)
    # Score 10 (boundary) should pass
    s = Schema(score=10, confidence=0.5, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)
    assert s.score == 10


def test_icml_schema_score_range_1_to_4():
    config = VenueConfig.load("icml")
    Schema = config.get_dimension_score_schema()
    s = Schema(score=3, confidence=0.8, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)
    assert s.score == 3
    with pytest.raises(ValidationError):
        Schema(score=5, confidence=0.5, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)


def test_acl_schema_score_range_1_to_4():
    config = VenueConfig.load("acl")
    Schema = config.get_dimension_score_schema()
    s = Schema(score=2, confidence=0.5, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)
    assert s.score == 2
    with pytest.raises(ValidationError):
        Schema(score=5, confidence=0.5, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)


def test_schema_name_includes_venue():
    config = VenueConfig.load("neurips")
    Schema = config.get_dimension_score_schema()
    assert Schema.__name__ == "DimensionScore_neurips"


def test_schema_confidence_range():
    config = VenueConfig.load("neurips")
    Schema = config.get_dimension_score_schema()
    with pytest.raises(ValidationError):
        Schema(score=5, confidence=1.5, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)
    with pytest.raises(ValidationError):
        Schema(score=5, confidence=-0.1, strengths=["a"], weaknesses=["b"],
               justification="x" * 200)
