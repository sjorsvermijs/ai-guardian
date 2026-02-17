"""Tests for the clinical guidelines retrieval system."""

import pytest
from src.core.guidelines.store import GuidelineStore
from src.core.guidelines.models import (
    ClinicalDomain,
    GuidelineSource,
    RetrievalQuery,
)


@pytest.fixture
def store():
    s = GuidelineStore()
    s.initialize()
    return s


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInitialization:
    def test_loads_all_sources(self, store):
        sources = {c.source for c in store._chunks}
        assert GuidelineSource.NICE_NG143 in sources
        assert GuidelineSource.WHO_IMCI in sources
        assert GuidelineSource.PEWS in sources

    def test_loads_sufficient_chunks(self, store):
        assert len(store._chunks) >= 15

    def test_is_loaded_flag(self, store):
        assert store.is_loaded is True

    def test_uninitialized_returns_empty(self):
        s = GuidelineStore()
        query = RetrievalQuery(age_months=6, respiratory_rate=65)
        assert s.retrieve(query) == []


# ------------------------------------------------------------------
# Age filtering
# ------------------------------------------------------------------

class TestAgeFiltering:
    def test_newborn_gets_psbi(self, store):
        query = RetrievalQuery(
            age_months=1,
            respiratory_rate=65,
            domains={ClinicalDomain.GENERAL, ClinicalDomain.RESPIRATORY},
        )
        results = store.retrieve(query)
        ids = [r.chunk.id for r in results]
        assert "imci_psbi_danger_signs" in ids

    def test_newborn_excludes_older_child_pneumonia(self, store):
        query = RetrievalQuery(
            age_months=1,
            respiratory_rate=55,
            domains={ClinicalDomain.RESPIRATORY, ClinicalDomain.GENERAL},
        )
        results = store.retrieve(query)
        ids = [r.chunk.id for r in results]
        # IMCI pneumonia 12-60mo should not appear for a 1-month-old
        assert "imci_pneumonia_12_60mo" not in ids

    def test_no_age_returns_results(self, store):
        query = RetrievalQuery(
            age_months=None,
            respiratory_rate=65,
            domains={ClinicalDomain.RESPIRATORY, ClinicalDomain.GENERAL},
        )
        results = store.retrieve(query)
        assert len(results) > 0


# ------------------------------------------------------------------
# Threshold-based scoring
# ------------------------------------------------------------------

class TestThresholdScoring:
    def test_high_rr_high_relevance(self, store):
        query = RetrievalQuery(
            age_months=8,
            respiratory_rate=65,
            domains={ClinicalDomain.RESPIRATORY, ClinicalDomain.GENERAL},
        )
        results = store.retrieve(query)
        assert len(results) > 0
        # Top result should have high relevance (threshold breached)
        assert results[0].relevance_score > 0.5
        assert any("RR" in r for r in results[0].match_reasons)

    def test_low_spo2_retrieves_nice_red(self, store):
        query = RetrievalQuery(
            age_months=8,
            spo2=93.0,
            domains={ClinicalDomain.RESPIRATORY, ClinicalDomain.GENERAL},
        )
        results = store.retrieve(query)
        ids = [r.chunk.id for r in results]
        # NICE amber chunks have spo2_lower=95 threshold
        assert any("nice_resp" in i for i in ids)

    def test_high_hr_retrieves_cardiovascular(self, store):
        query = RetrievalQuery(
            age_months=8,
            heart_rate=170,
            domains={ClinicalDomain.CARDIOVASCULAR, ClinicalDomain.GENERAL},
        )
        results = store.retrieve(query)
        ids = [r.chunk.id for r in results]
        assert any("circ" in i or "cardiovascular" in i.lower() for i in ids)
        assert any("HR" in r for res in results for r in res.match_reasons)

    def test_normal_values_low_relevance(self, store):
        query = RetrievalQuery(
            age_months=6,
            heart_rate=120,
            respiratory_rate=35,
            spo2=98,
            domains={
                ClinicalDomain.RESPIRATORY,
                ClinicalDomain.CARDIOVASCULAR,
                ClinicalDomain.GENERAL,
            },
        )
        results = store.retrieve(query)
        # All results should have low relevance (no thresholds breached)
        for r in results:
            assert r.relevance_score < 0.5


# ------------------------------------------------------------------
# Query building
# ------------------------------------------------------------------

class TestBuildQuery:
    def test_basic_vital_signs(self):
        query = GuidelineStore.build_query(
            vital_signs={"heart_rate": 170, "respiratory_rate": 55, "spo2": 94},
            acoustic_indicators={},
            patient_age=4,
        )
        assert query.heart_rate == 170
        assert query.respiratory_rate == 55
        assert query.spo2 == 94
        assert query.age_months == 4
        assert ClinicalDomain.CARDIOVASCULAR in query.domains
        assert ClinicalDomain.RESPIRATORY in query.domains

    def test_adventitious_sounds(self):
        query = GuidelineStore.build_query(
            vital_signs={},
            acoustic_indicators={
                "hear": {
                    "binary_classification": {
                        "prediction": "Adventitious",
                        "confidence": 0.8,
                    }
                }
            },
        )
        assert query.has_adventitious_sounds is True
        assert ClinicalDomain.RESPIRATORY in query.domains

    def test_cry_type(self):
        query = GuidelineStore.build_query(
            vital_signs={},
            acoustic_indicators={"cry": {"prediction": "discomfort"}},
        )
        assert query.cry_type == "discomfort"
        assert ClinicalDomain.CRY in query.domains

    def test_empty_inputs(self):
        query = GuidelineStore.build_query(
            vital_signs={}, acoustic_indicators={}
        )
        assert query.heart_rate is None
        assert query.respiratory_rate is None
        assert ClinicalDomain.GENERAL in query.domains


# ------------------------------------------------------------------
# Prompt formatting
# ------------------------------------------------------------------

class TestFormatForPrompt:
    def test_empty_results(self, store):
        assert store.format_for_prompt([]) == ""

    def test_respects_token_budget(self, store):
        query = RetrievalQuery(
            age_months=1,
            respiratory_rate=70,
            spo2=90,
            domains={ClinicalDomain.RESPIRATORY, ClinicalDomain.GENERAL},
        )
        results = store.retrieve(query, max_results=10)
        formatted = store.format_for_prompt(results, max_tokens_approx=300)
        word_count = len(formatted.split())
        # 300 tokens / ~1.3 tokens per word ≈ 230 words max
        assert word_count < 250

    def test_contains_guideline_header(self, store):
        query = RetrievalQuery(
            age_months=8,
            respiratory_rate=65,
            domains={ClinicalDomain.RESPIRATORY, ClinicalDomain.GENERAL},
        )
        results = store.retrieve(query)
        formatted = store.format_for_prompt(results)
        assert formatted.startswith("RELEVANT CLINICAL GUIDELINES:")

    def test_contains_citations(self, store):
        query = RetrievalQuery(
            age_months=8,
            respiratory_rate=65,
            domains={ClinicalDomain.RESPIRATORY, ClinicalDomain.GENERAL},
        )
        results = store.retrieve(query)
        formatted = store.format_for_prompt(results)
        # Should contain at least one guideline citation
        assert "NICE" in formatted or "WHO" in formatted or "PEWS" in formatted
