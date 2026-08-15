"""Venue-specific review configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Type
import yaml
from pydantic import BaseModel, Field, create_model


@dataclass
class VenueConfig:
    """A venue's review configuration: dimensions, scoring, weights, thresholds, prompts."""
    name: str
    display_name: str
    dimensions: List[str]
    score_min: int
    score_max: int
    confidence_min: float
    confidence_max: float
    weights: Dict[str, float]
    thresholds: List[tuple]
    prompts_dir: str

    _cache: ClassVar[Dict[str, "VenueConfig"]] = {}

    @classmethod
    def load(cls, name: str, venues_dir: str = "configs/venues") -> "VenueConfig":
        """Load a venue config by name. Cached at class level."""
        if name in cls._cache:
            return cls._cache[name]
        yaml_path = Path(venues_dir) / f"{name}.yaml"
        if not yaml_path.exists():
            raise ValueError(f"venue not found: {name} (looked at {yaml_path})")
        config = cls.from_yaml(str(yaml_path))
        cls._cache[name] = config
        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VenueConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw["name"],
            display_name=raw["display_name"],
            dimensions=raw["dimensions"],
            score_min=raw["score_min"],
            score_max=raw["score_max"],
            confidence_min=raw.get("confidence_min", 0.0),
            confidence_max=raw.get("confidence_max", 1.0),
            weights=raw["weights"],
            thresholds=[tuple(t) for t in raw["thresholds"]],
            prompts_dir=raw["prompts_dir"],
        )

    def get_dimension_score_schema(self) -> Type[BaseModel]:
        """Dynamically build a Pydantic schema for this venue's score range."""
        return create_model(
            f"DimensionScore_{self.name}",
            score=(int, Field(ge=self.score_min, le=self.score_max,
                              description=f"{self.score_min}-{self.score_max} scale")),
            confidence=(float, Field(ge=self.confidence_min, le=self.confidence_max)),
            strengths=(List[str], Field(min_length=1, max_length=5)),
            weaknesses=(List[str], Field(min_length=1, max_length=5)),
            justification=(str, Field(min_length=100, max_length=800)),
            evidence=(List[dict], Field(default_factory=list)),
            __base__=BaseModel,
        )
