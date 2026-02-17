"""
PEWS: Pediatric Early Warning Score.
Validated clinical scoring system for detecting deterioration in hospitalized children.
Source: Monaghan A. (2005) and subsequent validation studies.
"""

from ..models import AgeRange, ClinicalDomain, GuidelineChunk, GuidelineSource

_SRC = GuidelineSource.PEWS
_CITE = "Pediatric Early Warning Score (PEWS)"

PEWS_CHUNKS = [
    GuidelineChunk(
        id="pews_behavior",
        source=_SRC,
        domain=ClinicalDomain.NEUROLOGICAL,
        age_range=AgeRange(0, 60),
        relevant_parameters={"cry_type"},
        title="PEWS Behavior Domain",
        content=(
            "Behavior scoring: 0 = playing or appropriate for age, "
            "1 = sleeping, 2 = irritable, 3 = lethargic/confused or reduced "
            "response to pain. Irritability (score 2) or lethargy (score 3) "
            "are early signs of clinical deterioration in infants."
        ),
        thresholds={},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="pews_cardiovascular",
        source=_SRC,
        domain=ClinicalDomain.CARDIOVASCULAR,
        age_range=AgeRange(0, 60),
        relevant_parameters={"heart_rate"},
        title="PEWS Cardiovascular Domain",
        content=(
            "Cardiovascular scoring: 0 = pink, capillary refill 1-2s. "
            "1 = pale, capillary refill 3s. 2 = grey, capillary refill 4s, "
            "or tachycardia 20 bpm above normal for age. "
            "3 = grey and mottled, capillary refill >=5s, tachycardia 30 bpm "
            "above normal, or bradycardia. Score >=2 warrants increased monitoring."
        ),
        thresholds={},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="pews_respiratory",
        source=_SRC,
        domain=ClinicalDomain.RESPIRATORY,
        age_range=AgeRange(0, 60),
        relevant_parameters={"respiratory_rate", "spo2"},
        title="PEWS Respiratory Domain",
        content=(
            "Respiratory scoring: 0 = normal parameters, no retractions. "
            "1 = RR >10 above normal, using accessory muscles, or FiO2 >=30%. "
            "2 = RR >20 above normal with retractions, or FiO2 >=40%. "
            "3 = RR 5 below normal with retractions and grunting, or FiO2 >=50%. "
            "Score >=2 indicates significant respiratory compromise."
        ),
        thresholds={},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="pews_scoring",
        source=_SRC,
        domain=ClinicalDomain.GENERAL,
        age_range=AgeRange(0, 60),
        relevant_parameters={"heart_rate", "respiratory_rate", "spo2"},
        title="PEWS Total Score Interpretation",
        content=(
            "PEWS total score (sum of behavior + cardiovascular + respiratory, "
            "range 0-9). Score 0-3: low concern, routine monitoring. "
            "Score 4-6: moderate concern, increase monitoring frequency and "
            "notify physician. Score >=7: high concern, consider rapid response "
            "team activation and potential ICU transfer. AUCROC 0.91 at threshold >=8."
        ),
        thresholds={},
        citation=_CITE,
    ),
]
