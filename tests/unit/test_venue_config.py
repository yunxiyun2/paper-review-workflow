import pytest
from pathlib import Path
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
