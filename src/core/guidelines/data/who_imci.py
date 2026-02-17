"""
WHO IMCI: Integrated Management of Childhood Illness.
Classification criteria for young infants and children under 5.
Source: https://www.who.int/teams/maternal-newborn-child-adolescent-health-and-ageing/child-health/integrated-management-of-childhood-illness
"""

from ..models import AgeRange, ClinicalDomain, GuidelineChunk, GuidelineSource

_SRC = GuidelineSource.WHO_IMCI
_CITE = "WHO IMCI: Integrated Management of Childhood Illness"

IMCI_CHUNKS = [
    # --- PSBI in young infants (0-59 days) ---
    GuidelineChunk(
        id="imci_psbi_danger_signs",
        source=_SRC,
        domain=ClinicalDomain.GENERAL,
        age_range=AgeRange(0, 2),
        relevant_parameters={"respiratory_rate", "heart_rate", "spo2"},
        title="PSBI Danger Signs (0-59 days)",
        content=(
            "Possible Serious Bacterial Infection (PSBI) in young infants 0-59 days. "
            "Seven danger signs: (1) fast breathing RR >=60, (2) severe chest indrawing, "
            "(3) fever >=38C, (4) hypothermia <35.5C, (5) movement only on stimulation, "
            "(6) inability to feed, (7) convulsions. Any sign warrants urgent hospital referral."
        ),
        thresholds={"rr_upper": 60},
        citation=_CITE,
    ),

    # --- Respiratory classification 2-12 months ---
    GuidelineChunk(
        id="imci_pneumonia_2_12mo",
        source=_SRC,
        domain=ClinicalDomain.RESPIRATORY,
        age_range=AgeRange(2, 12),
        relevant_parameters={"respiratory_rate"},
        title="Fast Breathing / Pneumonia (2-12mo)",
        content=(
            "In children 2-12 months with cough or difficult breathing: respiratory "
            "rate >=50 breaths/min classifies as pneumonia (non-severe). Treat with "
            "oral amoxicillin. If danger signs present (inability to drink, convulsions, "
            "lethargy, stridor, severe malnutrition), classify as severe pneumonia "
            "requiring hospital admission."
        ),
        thresholds={"rr_upper": 50},
        citation=_CITE,
    ),
    GuidelineChunk(
        id="imci_pneumonia_12_60mo",
        source=_SRC,
        domain=ClinicalDomain.RESPIRATORY,
        age_range=AgeRange(12, 60),
        relevant_parameters={"respiratory_rate"},
        title="Fast Breathing / Pneumonia (12-59mo)",
        content=(
            "In children 12-59 months with cough or difficult breathing: respiratory "
            "rate >=40 breaths/min classifies as pneumonia (non-severe). Treat with "
            "oral amoxicillin. If danger signs present or severe chest indrawing, "
            "classify as severe pneumonia requiring hospital admission with "
            "parenteral antibiotics."
        ),
        thresholds={"rr_upper": 40},
        citation=_CITE,
    ),

    # --- Severe pneumonia ---
    GuidelineChunk(
        id="imci_severe_pneumonia",
        source=_SRC,
        domain=ClinicalDomain.RESPIRATORY,
        age_range=AgeRange(2, 60),
        relevant_parameters={"respiratory_rate", "spo2"},
        title="Severe Pneumonia Classification",
        content=(
            "Classify as severe pneumonia if fast breathing is accompanied by any "
            "general danger sign (inability to drink, persistent vomiting, convulsions, "
            "lethargy or unconsciousness) OR severe chest indrawing. Requires urgent "
            "hospital referral and parenteral antibiotics (ampicillin + gentamicin)."
        ),
        thresholds={"rr_upper": 50},
        citation=_CITE,
    ),

    # --- Wheezing ---
    GuidelineChunk(
        id="imci_wheezing",
        source=_SRC,
        domain=ClinicalDomain.RESPIRATORY,
        age_range=AgeRange(2, 60),
        relevant_parameters={"adventitious_sounds"},
        title="Wheezing Classification",
        content=(
            "Wheezing (expiratory wheeze on auscultation) is classified separately "
            "from pneumonia. First episode of wheezing: give inhaled salbutamol and "
            "reassess after 30 minutes. Recurrent wheezing: assess for asthma. "
            "If respiratory distress persists after bronchodilator, classify as "
            "severe pneumonia."
        ),
        thresholds={},
        citation=_CITE,
    ),

    # --- General danger signs ---
    GuidelineChunk(
        id="imci_general_danger",
        source=_SRC,
        domain=ClinicalDomain.GENERAL,
        age_range=AgeRange(2, 60),
        relevant_parameters={"heart_rate", "respiratory_rate", "spo2"},
        title="General Danger Signs (2-59mo)",
        content=(
            "General danger signs in children 2-59 months: inability to drink or "
            "breastfeed, persistent vomiting, convulsions, lethargic or unconscious. "
            "Any child with a general danger sign requires urgent hospital referral "
            "regardless of other classifications."
        ),
        thresholds={},
        citation=_CITE,
    ),
]
