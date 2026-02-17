"""
GuidelineStore: deterministic retrieval of clinical guidelines based on pipeline findings.
Uses metadata filtering and threshold-based scoring — no vector database required.
"""

import logging
from typing import Dict, List, Optional, Set, Any

from .models import (
    ClinicalDomain,
    GuidelineChunk,
    RetrievalQuery,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class GuidelineStore:
    """
    Retrieves relevant clinical guideline chunks based on patient findings.

    Retrieval algorithm:
      1. Filter by patient age range
      2. Filter by active clinical domains (derived from available data)
      3. Score candidates by parameter overlap + threshold breach
      4. Return top-k by relevance score
    """

    def __init__(self):
        self._chunks: List[GuidelineChunk] = []
        self._loaded = False

    def initialize(self) -> None:
        """Load all guideline chunks from data modules."""
        from .data.nice_ng143 import NICE_CHUNKS
        from .data.who_imci import IMCI_CHUNKS
        from .data.pews import PEWS_CHUNKS

        self._chunks = NICE_CHUNKS + IMCI_CHUNKS + PEWS_CHUNKS
        self._loaded = True
        logger.info(
            "GuidelineStore loaded %d chunks from %d sources",
            len(self._chunks),
            len({c.source for c in self._chunks}),
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: RetrievalQuery, max_results: int = 4
    ) -> List[RetrievalResult]:
        """Retrieve the most relevant guideline chunks for the clinical findings."""
        if not self._loaded:
            return []

        candidates = self._filter_by_age(query.age_months)
        candidates = self._filter_by_domain(candidates, query.domains)
        scored = self._score_candidates(candidates, query)
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        return scored[:max_results]

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    @staticmethod
    def build_query(
        vital_signs: Dict[str, Any],
        acoustic_indicators: Dict[str, Any],
        patient_age: Optional[int] = None,
    ) -> RetrievalQuery:
        """Build a RetrievalQuery from raw pipeline output dicts."""
        domains: Set[ClinicalDomain] = set()

        hr = vital_signs.get("heart_rate")
        rr = vital_signs.get("respiratory_rate")
        spo2 = vital_signs.get("spo2")

        if hr is not None:
            domains.add(ClinicalDomain.CARDIOVASCULAR)
        if rr is not None or spo2 is not None:
            domains.add(ClinicalDomain.RESPIRATORY)

        # Acoustic / HeAR
        hear_data = acoustic_indicators.get("hear", {})
        binary = hear_data.get("binary_classification", {})
        has_adventitious = (
            binary.get("prediction") == "Adventitious"
            and binary.get("confidence", 0) > 0.5
        )
        if has_adventitious:
            domains.add(ClinicalDomain.RESPIRATORY)

        # Cry
        cry_data = acoustic_indicators.get("cry", {})
        cry_type = cry_data.get("prediction")
        if cry_type:
            domains.add(ClinicalDomain.CRY)

        # Always include GENERAL so broad danger-sign guidelines are considered
        domains.add(ClinicalDomain.GENERAL)

        return RetrievalQuery(
            age_months=patient_age,
            heart_rate=hr,
            respiratory_rate=rr,
            spo2=spo2,
            has_adventitious_sounds=has_adventitious,
            cry_type=cry_type,
            domains=domains,
        )

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_for_prompt(
        self,
        results: List[RetrievalResult],
        max_tokens_approx: int = 300,
    ) -> str:
        """Format retrieved guidelines as concise text for MedGemma prompt injection."""
        if not results:
            return ""

        lines = ["RELEVANT CLINICAL GUIDELINES:"]
        token_estimate = 5

        for r in results:
            chunk_text = f"[{r.chunk.citation}] {r.chunk.title}: {r.chunk.content}"
            if r.match_reasons:
                chunk_text += f" (Matched: {'; '.join(r.match_reasons)})"

            # Rough token estimate: ~1.3 tokens per word
            chunk_tokens = len(chunk_text.split()) * 1.3
            if token_estimate + chunk_tokens > max_tokens_approx:
                break

            lines.append(f"- {chunk_text}")
            token_estimate += chunk_tokens

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal filtering & scoring
    # ------------------------------------------------------------------

    def _filter_by_age(
        self, age_months: Optional[int]
    ) -> List[GuidelineChunk]:
        if age_months is None:
            return list(self._chunks)
        return [c for c in self._chunks if c.age_range.contains(age_months)]

    def _filter_by_domain(
        self,
        chunks: List[GuidelineChunk],
        domains: Set[ClinicalDomain],
    ) -> List[GuidelineChunk]:
        if not domains:
            return chunks
        return [c for c in chunks if c.domain in domains]

    def _score_candidates(
        self,
        chunks: List[GuidelineChunk],
        query: RetrievalQuery,
    ) -> List[RetrievalResult]:
        results = []
        for chunk in chunks:
            score, reasons = self._score_chunk(chunk, query)
            if score > 0.05:
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        relevance_score=min(score, 1.0),
                        match_reasons=reasons,
                    )
                )
        return results

    def _score_chunk(
        self, chunk: GuidelineChunk, query: RetrievalQuery
    ) -> tuple:
        """Score a single chunk. Returns (score, reasons)."""
        score = 0.0
        reasons: List[str] = []

        # --- Parameter overlap (0.0 – 0.3) ---
        available = set()
        if query.heart_rate is not None:
            available.add("heart_rate")
        if query.respiratory_rate is not None:
            available.add("respiratory_rate")
        if query.spo2 is not None:
            available.add("spo2")
        if query.has_adventitious_sounds:
            available.add("adventitious_sounds")
        if query.cry_type:
            available.add("cry_type")

        overlap = available & chunk.relevant_parameters
        if chunk.relevant_parameters:
            score += 0.3 * (len(overlap) / len(chunk.relevant_parameters))

        # --- Threshold breach (0.0 – 0.7) ---
        breach_score, breach_reasons = self._check_thresholds(
            chunk.thresholds, query
        )
        score += 0.7 * breach_score
        reasons.extend(breach_reasons)

        return score, reasons

    def _check_thresholds(
        self, thresholds: Dict[str, float], query: RetrievalQuery
    ) -> tuple:
        """Check whether patient values breach guideline thresholds."""
        if not thresholds:
            return 0.0, []

        breaches = 0
        total = 0
        reasons: List[str] = []

        if "rr_upper" in thresholds and query.respiratory_rate is not None:
            total += 1
            if query.respiratory_rate > thresholds["rr_upper"]:
                breaches += 1
                reasons.append(
                    f"RR {query.respiratory_rate:.0f} exceeds "
                    f"threshold {thresholds['rr_upper']:.0f}"
                )

        if "spo2_lower" in thresholds and query.spo2 is not None:
            total += 1
            if query.spo2 < thresholds["spo2_lower"]:
                breaches += 1
                reasons.append(
                    f"SpO2 {query.spo2:.0f}% below "
                    f"threshold {thresholds['spo2_lower']:.0f}%"
                )

        if "hr_upper" in thresholds and query.heart_rate is not None:
            total += 1
            if query.heart_rate > thresholds["hr_upper"]:
                breaches += 1
                reasons.append(
                    f"HR {query.heart_rate:.0f} exceeds "
                    f"threshold {thresholds['hr_upper']:.0f} bpm"
                )

        if total == 0:
            return 0.1, []  # small base score for parameter overlap alone

        return breaches / total, reasons
