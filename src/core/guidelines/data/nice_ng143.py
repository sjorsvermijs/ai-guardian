"""
NICE NG143: Fever in under 5s - assessment and initial management.
Traffic light system for identifying risk of serious illness.
Source: https://www.nice.org.uk/guidance/ng143
"""

from ..models import AgeRange, ClinicalDomain, GuidelineChunk, GuidelineSource

_SRC = GuidelineSource.NICE_NG143
_CITE = "NICE Guideline NG143: Fever in under 5s (2021)"

NICE_CHUNKS = [
    # --- Respiratory ---
    GuidelineChunk(
        id="nice_resp_amber_6_12mo",
        source=_SRC,
        domain=ClinicalDomain.RESPIRATORY,
        age_range=AgeRange(6, 12),
        relevant_parameters={"respiratory_rate", "spo2"},
        title="Respiratory Amber Features (6-12mo)",
        content=(
            "Intermediate risk: tachypnoea with respiratory rate >50 breaths/min, "
            "nasal flaring, oxygen saturation <=95% in room air, or crackles on "
            "auscultation. Warrants same-day clinical assessment."
        ),
        thresholds={"rr_upper": 50, "spo2_lower": 95},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="nice_resp_amber_12_60mo",
        source=_SRC,
        domain=ClinicalDomain.RESPIRATORY,
        age_range=AgeRange(12, 60),
        relevant_parameters={"respiratory_rate", "spo2"},
        title="Respiratory Amber Features (>12mo)",
        content=(
            "Intermediate risk: tachypnoea with respiratory rate >40 breaths/min, "
            "nasal flaring, oxygen saturation <=95% in room air, or crackles on "
            "auscultation. Warrants same-day clinical assessment."
        ),
        thresholds={"rr_upper": 40, "spo2_lower": 95},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="nice_resp_red",
        source=_SRC,
        domain=ClinicalDomain.RESPIRATORY,
        age_range=AgeRange(0, 60),
        relevant_parameters={"respiratory_rate", "spo2"},
        title="Respiratory Red Flags",
        content=(
            "High risk: grunting, respiratory rate >60 breaths/min, or moderate-to-severe "
            "chest indrawing. These findings indicate high risk of serious illness. "
            "Requires immediate assessment by a senior clinician."
        ),
        thresholds={"rr_upper": 60},
        citation=_CITE,
    ),

    # --- Cardiovascular ---
    GuidelineChunk(
        id="nice_circ_amber_6_12mo",
        source=_SRC,
        domain=ClinicalDomain.CARDIOVASCULAR,
        age_range=AgeRange(6, 12),
        relevant_parameters={"heart_rate"},
        title="Cardiovascular Amber Features (6-12mo)",
        content=(
            "Intermediate risk: tachycardia with heart rate >160 bpm in infants "
            "6-12 months. Also consider pallor reported by parent, capillary refill "
            ">=3 seconds, dry mucous membranes, and poor feeding."
        ),
        thresholds={"hr_upper": 160},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="nice_circ_amber_12_24mo",
        source=_SRC,
        domain=ClinicalDomain.CARDIOVASCULAR,
        age_range=AgeRange(12, 24),
        relevant_parameters={"heart_rate"},
        title="Cardiovascular Amber Features (12-24mo)",
        content=(
            "Intermediate risk: tachycardia with heart rate >150 bpm in children "
            "12-24 months. Also consider pallor, capillary refill >=3 seconds, "
            "dry mucous membranes, and reduced urine output."
        ),
        thresholds={"hr_upper": 150},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="nice_circ_amber_24_60mo",
        source=_SRC,
        domain=ClinicalDomain.CARDIOVASCULAR,
        age_range=AgeRange(24, 60),
        relevant_parameters={"heart_rate"},
        title="Cardiovascular Amber Features (2-5y)",
        content=(
            "Intermediate risk: tachycardia with heart rate >140 bpm in children "
            "2-5 years. Also consider pallor, capillary refill >=3 seconds, "
            "dry mucous membranes, and reduced urine output."
        ),
        thresholds={"hr_upper": 140},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="nice_circ_red",
        source=_SRC,
        domain=ClinicalDomain.CARDIOVASCULAR,
        age_range=AgeRange(0, 60),
        relevant_parameters={"heart_rate"},
        title="Cardiovascular Red Flags",
        content=(
            "High risk: pale, mottled, ashen, or blue skin colour. Reduced skin "
            "turgor. These indicate possible circulatory compromise requiring "
            "immediate senior clinical assessment."
        ),
        thresholds={},
        citation=_CITE,
    ),

    # --- Activity / Neurological ---
    GuidelineChunk(
        id="nice_activity_amber",
        source=_SRC,
        domain=ClinicalDomain.CRY,
        age_range=AgeRange(0, 60),
        relevant_parameters={"cry_type"},
        title="Activity Amber Features",
        content=(
            "Intermediate risk: not responding normally to social cues, no smile, "
            "wakes only with prolonged stimulation, decreased activity. These "
            "behavioural changes warrant further clinical assessment."
        ),
        thresholds={},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="nice_activity_red",
        source=_SRC,
        domain=ClinicalDomain.CRY,
        age_range=AgeRange(0, 60),
        relevant_parameters={"cry_type"},
        title="Activity Red Flags",
        content=(
            "High risk: no response to social cues, appears ill to a healthcare "
            "professional, unable to stay awake even when roused, weak, high-pitched "
            "or continuous cry. Requires immediate assessment."
        ),
        thresholds={},
        citation=_CITE,
    ),
]
