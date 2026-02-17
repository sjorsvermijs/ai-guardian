"""Data models for clinical guideline retrieval."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class GuidelineSource(Enum):
    NICE_NG143 = "NICE NG143"
    WHO_IMCI = "WHO IMCI"
    PEWS = "PEWS"


class ClinicalDomain(Enum):
    RESPIRATORY = "respiratory"
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL = "neurological"
    GENERAL = "general"
    CRY = "cry"


@dataclass
class AgeRange:
    min_months: int  # inclusive
    max_months: int  # inclusive (use 60 for "under 5")

    def contains(self, age_months: int) -> bool:
        return self.min_months <= age_months <= self.max_months


@dataclass
class GuidelineChunk:
    """A single retrievable unit of clinical guideline knowledge."""
    id: str
    source: GuidelineSource
    domain: ClinicalDomain
    age_range: AgeRange
    relevant_parameters: Set[str]
    title: str
    content: str
    thresholds: Dict[str, float] = field(default_factory=dict)
    citation: str = ""


@dataclass
class RetrievalQuery:
    """Query built from pipeline outputs."""
    age_months: Optional[int] = None
    heart_rate: Optional[float] = None
    respiratory_rate: Optional[float] = None
    spo2: Optional[float] = None
    has_adventitious_sounds: bool = False
    cry_type: Optional[str] = None
    domains: Set[ClinicalDomain] = field(default_factory=set)


@dataclass
class RetrievalResult:
    """A guideline chunk with relevance scoring."""
    chunk: GuidelineChunk
    relevance_score: float
    match_reasons: List[str]
