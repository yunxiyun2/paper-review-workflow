"""Pydantic schemas for LLM structured output."""
from typing import List, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """One concrete piece of evidence cited from the paper."""
    location: str = Field(min_length=1, description="Paper location, e.g. '§3.2', 'Table 1', 'Figure 3'")
    quote: str = Field(min_length=1, description="Short verbatim quote or concrete data point from that location")


class DimensionScore(BaseModel):
    """Single dimension scoring result (1-5 OpenReview scale)."""
    score: int = Field(ge=1, le=5, description="1-5 OpenReview scale")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0-1.0")
    strengths: List[str] = Field(min_length=1, max_length=5)
    weaknesses: List[str] = Field(min_length=1, max_length=5)
    justification: str = Field(min_length=100, max_length=800)
    evidence: List[EvidenceItem] = Field(default_factory=list, max_length=5)


class PaperMetadata(BaseModel):
    """Paper metadata extracted from PDF/arXiv."""
    title: str
    authors: List[str]
    abstract: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    """Synthesized review across all dimensions."""
    summary: str = Field(min_length=200, max_length=1500)
    key_strengths: List[str]
    key_weaknesses: List[str]
    questions_for_authors: List[str]
    overall_assessment: str
