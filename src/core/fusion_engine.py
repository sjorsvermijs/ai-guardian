"""
Fusion Engine - Combines outputs from HeAR, rPPG, and Cry pipelines
Uses MedGemma 4B for all clinical reasoning: triage priority, alerts, and messaging
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .base_pipeline import PipelineResult

logger = logging.getLogger(__name__)

# MedGemma imports
try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MEDGEMMA_AVAILABLE = True
except ImportError:
    MEDGEMMA_AVAILABLE = False
    logger.warning("MedGemma (MLX) dependencies not available. Install: pip install mlx mlx-lm")


class TriagePriority(Enum):
    """Triage priority levels"""
    CRITICAL = "CRITICAL"  # Immediate intervention required
    URGENT = "URGENT"      # Needs prompt attention
    MODERATE = "MODERATE"  # Can wait for standard care
    LOW = "LOW"           # Non-urgent monitoring


VALID_PRIORITIES = {p.value for p in TriagePriority}


@dataclass
class TriageReport:
    """Final triage assessment combining all pipeline outputs"""
    timestamp: datetime
    priority: TriagePriority
    confidence: float
    summary: str
    parent_message: str          # Parent-friendly explanation
    specialist_message: str      # Clinical summary for medical professionals
    vital_signs: Dict[str, Any]
    acoustic_indicators: Dict[str, Any]
    visual_indicators: Dict[str, Any]
    critical_alerts: List[str]
    recommendations: List[str]
    individual_results: Dict[str, PipelineResult]
    metadata: Dict[str, Any]


class FusionEngine:
    """
    Combines and analyzes results from all pipelines.
    Delegates all clinical reasoning (priority, alerts, recommendations, messaging)
    to MedGemma 4B, with a simple rule-based fallback when MedGemma is unavailable.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False

        # Pipeline weights for confidence calculation
        self.weights = {
            'hear': self.config.get('hear_weight', 0.25),
            'rppg': self.config.get('rppg_weight', 0.35),
            'medgemma_vqa': self.config.get('vqa_weight', 0.40)
        }

        # MedGemma model for clinical interpretation
        self.medgemma_model = None
        self.medgemma_tokenizer = None
        self.use_medgemma = self.config.get('use_medgemma', True) and MEDGEMMA_AVAILABLE

        # Clinical guidelines retrieval
        self.guideline_store = None

    def initialize(self):
        """Initialize the fusion engine and load MedGemma model"""
        if self.use_medgemma:
            self._initialize_medgemma()

        # Initialize clinical guidelines store (instant, no model loading)
        try:
            from .guidelines.store import GuidelineStore
            self.guideline_store = GuidelineStore()
            self.guideline_store.initialize()
        except Exception as e:
            logger.warning("GuidelineStore failed to load: %s", e)
            self.guideline_store = None

        self.is_initialized = True

    def _initialize_medgemma(self):
        """Initialize MedGemma model for clinical interpretation (MLX optimized)"""
        try:
            logger.info("Loading MedGemma model for clinical interpretation (MLX)...")

            model_id = "mlx-community/medgemma-4b-it-4bit"
            self.medgemma_model, self.medgemma_tokenizer = load(model_id)

            logger.info("MedGemma loaded with MLX (M4 optimized)")
        except Exception as e:
            error_msg = str(e)
            if "gated repo" in error_msg or "authenticated" in error_msg:
                logger.warning(
                    "MedGemma requires authentication:\n"
                    "  1. Request access: https://huggingface.co/mlx-community/medgemma-4b-it-4bit\n"
                    "  2. Get token: https://huggingface.co/settings/tokens\n"
                    "  3. Login: huggingface-cli login"
                )
            else:
                logger.warning("Failed to load MedGemma: %s", e)
            self.use_medgemma = False

    def fuse(self,
             hear_result: PipelineResult = None,
             rppg_result: PipelineResult = None,
             cry_result: PipelineResult = None,
             vqa_result: PipelineResult = None,
             patient_age: int = None,
             patient_sex: str = None,
             parent_notes: str = None) -> TriageReport:
        """
        Fuse results from all pipelines into a comprehensive triage report.
        MedGemma performs all clinical reasoning; a rule-based fallback is used
        when MedGemma is unavailable.
        """
        timestamp = datetime.now()

        # Calculate weighted confidence from pipeline outputs
        confidences = []
        if hear_result and hear_result.confidence > 0:
            confidences.append(('hear', hear_result.confidence))
        if rppg_result and rppg_result.confidence > 0:
            confidences.append(('rppg', rppg_result.confidence))
        if vqa_result and vqa_result.confidence > 0:
            confidences.append(('medgemma_vqa', vqa_result.confidence))

        overall_confidence = self._calculate_weighted_confidence(confidences)

        # Extract structured data from pipeline results
        vital_signs = rppg_result.data.copy() if rppg_result else {}

        acoustic_indicators = {}
        if hear_result:
            acoustic_indicators['hear'] = hear_result.data.copy()
        if cry_result:
            acoustic_indicators['cry'] = cry_result.data.copy()

        visual_indicators = vqa_result.data.copy() if vqa_result else {}

        # Build clinical data summary for reasoning
        clinical_data = self._build_clinical_data(
            vital_signs, acoustic_indicators, visual_indicators,
            patient_age, patient_sex, parent_notes
        )

        # Retrieve relevant clinical guidelines
        guideline_context = ""
        guideline_ids = []
        if self.guideline_store and self.guideline_store.is_loaded:
            try:
                from .guidelines.store import GuidelineStore
                query = GuidelineStore.build_query(
                    vital_signs, acoustic_indicators, patient_age
                )
                guideline_results = self.guideline_store.retrieve(query, max_results=4)
                guideline_context = self.guideline_store.format_for_prompt(guideline_results)
                guideline_ids = [r.chunk.id for r in guideline_results]
            except Exception as e:
                logger.warning("Guideline retrieval failed: %s", e)

        # Let MedGemma handle all clinical reasoning
        if self.use_medgemma and self.medgemma_model:
            triage = self._medgemma_clinical_reasoning(clinical_data, guideline_context)
        else:
            triage = self._fallback_clinical_reasoning(
                vital_signs, acoustic_indicators, visual_indicators, guideline_context
            )

        priority = triage['priority']
        critical_alerts = triage['critical_alerts']
        recommendations = triage['recommendations']
        parent_message = triage['parent_message']
        specialist_message = triage['specialist_message']

        summary = (
            f"Triage Priority: {priority.value}. "
            f"Detected {len(critical_alerts)} concerning indicator(s). "
            f"Vital signs - HR: {vital_signs.get('heart_rate', 'N/A')}, "
            f"RR: {vital_signs.get('respiratory_rate', 'N/A')}, "
            f"SpO2: {vital_signs.get('spo2', 'N/A')}."
        )

        return TriageReport(
            timestamp=timestamp,
            priority=priority,
            confidence=overall_confidence,
            summary=summary,
            parent_message=parent_message,
            specialist_message=specialist_message,
            vital_signs=vital_signs,
            acoustic_indicators=acoustic_indicators,
            visual_indicators=visual_indicators,
            critical_alerts=list(set(critical_alerts)),
            recommendations=recommendations,
            individual_results={
                'hear': hear_result,
                'rppg': rppg_result,
                'cry': cry_result,
                'vga': vqa_result
            },
            metadata={
                'fusion_engine_version': '0.3.0',
                'num_pipelines_active': len(confidences),
                'weights_used': self.weights,
                'medgemma_enabled': self.use_medgemma and self.medgemma_model is not None,
                'reasoning_mode': 'medgemma' if (self.use_medgemma and self.medgemma_model) else 'fallback',
                'guidelines_retrieved': guideline_ids,
            }
        )

    def _calculate_weighted_confidence(self, confidences: List[tuple]) -> float:
        """Calculate weighted average of pipeline confidences"""
        if not confidences:
            return 0.0

        total_weight = sum(self.weights.get(name, 0) for name, _ in confidences)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            self.weights.get(name, 0) * conf
            for name, conf in confidences
        )

        return weighted_sum / total_weight

    def _build_clinical_data(self,
                            vital_signs: Dict[str, Any],
                            acoustic_indicators: Dict[str, Any],
                            visual_indicators: Dict[str, Any],
                            patient_age: int = None,
                            patient_sex: str = None,
                            parent_notes: str = None) -> str:
        """Build clinical data summary string for MedGemma prompt"""
        parts = ["INFANT HEALTH ASSESSMENT - Multi-Modal Analysis:\n"]

        # Patient Context
        parts.append("Patient Information:")
        if patient_age is not None:
            age_str = f"{patient_age} months old" 
            parts.append(f"- Age: {age_str}")
        if patient_sex:
            parts.append(f"- Sex: {patient_sex}")
        if parent_notes:
            parts.append(f"- Parent Observations: {parent_notes}")
        parts.append("")

        # Vital Signs
        if vital_signs:
            parts.append("Vital Signs (via rPPG - contactless video-based):")
            hr = vital_signs.get('heart_rate')
            if hr is not None:
                parts.append(f"- Heart Rate: {hr:.1f} BPM")
            rr = vital_signs.get('respiratory_rate')
            if rr is not None:
                parts.append(f"- Respiratory Rate: {rr:.1f} breaths/min")
            spo2 = vital_signs.get('spo2')
            if spo2 is not None:
                parts.append(f"- SpO2: {spo2:.1f}%")
            sqi = vital_signs.get('signal_quality')
            if sqi is not None:
                sqi_desc = "Excellent" if sqi > 0.8 else "Good" if sqi > 0.6 else "Acceptable" if sqi > 0.5 else "Poor"
                parts.append(f"- Signal Quality: {sqi:.1%} ({sqi_desc})")
            parts.append("")

        # Acoustic Indicators (HeAR + Cry)
        if acoustic_indicators:
            parts.append("Acoustic Analysis:")
            if 'cry' in acoustic_indicators:
                cry_data = acoustic_indicators['cry']
                if 'prediction' in cry_data:
                    confidence = cry_data.get('confidence', 0) * 100
                    parts.append(f"- Cry Classification: {cry_data['prediction']} ({confidence:.0f}% confidence)")
                    probs = cry_data.get('probabilities', {})
                    if probs:
                        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                        parts.append(f"  Top predictions: {', '.join(f'{k}={v*100:.0f}%' for k,v in top_3)}")

            if 'hear' in acoustic_indicators:
                hear_data = acoustic_indicators['hear']
                if 'binary_classification' in hear_data:
                    bc = hear_data['binary_classification']
                    parts.append(f"- Respiratory Sounds: {bc['prediction']} ({bc['confidence']*100:.0f}% confidence)")
                if 'multiclass_classification' in hear_data:
                    mc = hear_data['multiclass_classification']
                    parts.append(f"- Sound Classification: {mc['prediction']} ({mc['confidence']*100:.0f}% confidence)")
            parts.append("")

        # Visual Indicators (Skin Classification)
        if visual_indicators:
            skin = visual_indicators.get('skin_assessment', {})
            if skin and skin.get('classification'):
                parts.append("Visual Assessment (Skin Classification):")
                classification = skin['classification']
                confidence = skin.get('confidence', 0) * 100
                parts.append(f"- Skin Condition: {classification} ({confidence:.0f}% confidence)")
                if classification != 'healthy':
                    parts.append(f"- Status: {skin.get('overall_status', 'concerning').upper()}")
                per_shot = visual_indicators.get('per_screenshot', [])
                if per_shot:
                    preds = [s['prediction'] for s in per_shot]
                    parts.append(f"- Screenshots analyzed: {len(per_shot)} ({', '.join(preds)})")
                parts.append("")
            else:
                parts.append("Visual Assessment:")
                for key, value in visual_indicators.items():
                    if key not in ('status', 'message', 'image_shapes', 'num_screenshots_analyzed',
                                   'skin_assessment', 'per_screenshot', 'labels'):
                        parts.append(f"- {key.replace('_', ' ').title()}: {value}")
                parts.append("")

        return "\n".join(parts)

    def _medgemma_clinical_reasoning(self, clinical_data: str,
                                     guideline_context: str = "") -> Dict[str, Any]:
        """
        Two-pass MedGemma reasoning to prevent specialist message cutoff.
        Pass 1: Priority, alerts, and parent message (~400 tokens).
        Pass 2: Specialist message with full guideline context (~400 tokens).
        """
        try:
            # --- Pass 1: Parent-facing content ---
            parent_result = self._medgemma_pass_parent(clinical_data)
            parsed = self._parse_parent_response(parent_result)

            # --- Pass 2: Specialist message (separate token budget) ---
            specialist_message = self._medgemma_pass_specialist(
                clinical_data, guideline_context
            )
            parsed['specialist_message'] = specialist_message

            return parsed

        except Exception as e:
            logger.error("MedGemma clinical reasoning failed: %s", e, exc_info=True)
            return self._fallback_clinical_reasoning({}, {})

    def _medgemma_pass_parent(self, clinical_data: str) -> str:
        """Pass 1: Generate priority, alerts, and parent message."""
        system_content = (
            "You are a pediatric triage AI. Given multi-modal sensor data from an infant "
            "monitoring system, produce a structured clinical assessment.\n\n"
            "INFANT NORMAL RANGES:\n"
            "- Heart Rate: 100-160 BPM (newborns), 90-150 BPM (infants 1-12 months)\n"
            "- Respiratory Rate: 30-60 breaths/min (newborns), 25-40 (infants 1-12 months)\n"
            "- SpO2: >= 95% is normal; < 92% is concerning; < 88% is critical\n\n"
            "IMPORTANT RULES:\n"
            "- Refer to the baby as 'your baby' or 'the infant'. Never use placeholders like [name].\n"
            "- Do NOT generate code, functions, or programming syntax.\n"
            "- Write ONLY natural language text.\n"
            "- If signal quality is poor, stress reduced confidence but still assess available data.\n"
            "- Cross-validate findings: if multiple modalities agree (e.g., low SpO2 + adventitious "
            "respiratory sounds), increase urgency."
        )

        # Few-shot example so MedGemma sees what a real response looks like
        example_user = (
            "INFANT HEALTH ASSESSMENT - Multi-Modal Analysis:\n\n"
            "Vital Signs (via rPPG):\n- Heart Rate: 155.0 BPM\n- Respiratory Rate: 52.0 breaths/min\n"
            "- SpO2: 94.0%\n- Signal Quality: 72.0% (Good)\n\n"
            "Acoustic Analysis:\n- Cry Classification: discomfort (78% confidence)\n"
            "- Respiratory Sounds: Normal (85% confidence)\n\n"
            "Based on the above data, produce your assessment."
        )
        example_response = (
            "PRIORITY: MODERATE\n\n"
            "ALERTS:\n"
            "- Breathing rate slightly above normal range for age\n"
            "- Oxygen levels just below the typical healthy range\n\n"
            "PARENT MESSAGE:\n"
            "We checked your baby's breathing and heart rate from the video. "
            "The breathing rate is a little faster than usual, and oxygen levels are "
            "slightly lower than what we normally expect. This does not mean something is "
            "wrong right now, but it is worth keeping an eye on. If your baby seems "
            "uncomfortable, is breathing hard, or you are worried, call your doctor for advice."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": example_user},
            {"role": "assistant", "content": example_response},
            {
                "role": "user",
                "content": (
                    f"{clinical_data}\n"
                    "Based on the above data, produce your assessment in the same format "
                    "(PRIORITY, ALERTS, PARENT MESSAGE). "
                    "The parent message must use simple everyday language (max 120 words), "
                    "no medical terms like SpO2 or respiratory rate. "
                    "Say 'breathing', 'heart rate', 'oxygen levels' instead."
                )
            }
        ]

        prompt = self._format_chat_prompt(messages)
        result = generate(
            self.medgemma_model,
            self.medgemma_tokenizer,
            prompt=prompt,
            max_tokens=400,
            verbose=False
        )
        return self._clean_llm_output(result)

    def _medgemma_pass_specialist(self, clinical_data: str,
                                  guideline_context: str = "") -> str:
        """Pass 2: Generate specialist message with dedicated token budget."""
        system_content = (
            "You are a pediatric clinical specialist. Write a concise clinical note "
            "based on the sensor data provided. Use medical terminology appropriate "
            "for healthcare professionals.\n\n"
            "INFANT NORMAL RANGES:\n"
            "- Heart Rate: 100-160 BPM (newborns), 90-150 BPM (infants 1-12 months)\n"
            "- Respiratory Rate: 30-60 breaths/min (newborns), 25-40 (infants 1-12 months)\n"
            "- SpO2: >= 95% is normal; < 92% is concerning; < 88% is critical\n\n"
            "RULES:\n"
            "- Do NOT generate code or programming syntax.\n"
            "- Write ONLY a clinical note in natural language.\n"
            "- No markdown formatting.\n"
            "- Refer to the patient as 'the infant'."
        )

        if guideline_context:
            system_content += (
                "\n\nCite the clinical guidelines below when they support your findings. "
                "Use [Guideline Name] format for citations.\n\n"
                + guideline_context
            )

        # Few-shot example for specialist note
        example_specialist_user = (
            "INFANT HEALTH ASSESSMENT - Multi-Modal Analysis:\n\n"
            "Vital Signs (via rPPG):\n- Heart Rate: 155.0 BPM\n- Respiratory Rate: 52.0 breaths/min\n"
            "- SpO2: 94.0%\n- Signal Quality: 72.0% (Good)\n\n"
            "Acoustic Analysis:\n- Cry Classification: discomfort (78% confidence)\n"
            "- Respiratory Sounds: Normal (85% confidence)\n\n"
            "Write a clinical note (max 200 words)."
        )
        example_specialist_response = (
            "Clinical assessment of infant via contactless multi-modal monitoring. "
            "rPPG-derived vital signs show HR 155 bpm (within normal limits for age), "
            "RR 52 breaths/min (elevated above the 30-40 normal range for infants 1-12 months), "
            "and SpO2 94% (borderline, just below the 95% threshold). Signal quality acceptable at 72%. "
            "Acoustic analysis reveals discomfort-type cry pattern (78% confidence) without adventitious "
            "respiratory sounds. The combination of mild tachypnea and borderline oxygen saturation "
            "warrants monitoring for evolving lower respiratory tract pathology. Differential includes "
            "early bronchiolitis, viral URI with reactive airway component, or physiological variation "
            "in the context of crying. Recommend repeat assessment in 1-2 hours, clinical correlation "
            "with temperature and feeding history, and low threshold for in-person evaluation if "
            "symptoms persist or worsen."
        )

        specialist_instruction = (
            "Write a clinical note (max 200 words) covering: "
            "findings summary, clinical significance with pediatric normal ranges, "
            "differential considerations, and recommended follow-up."
        )
        if guideline_context:
            specialist_instruction += (
                " Reference applicable clinical guidelines with specific thresholds."
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": example_specialist_user},
            {"role": "assistant", "content": example_specialist_response},
            {
                "role": "user",
                "content": f"{clinical_data}\n{specialist_instruction}"
            }
        ]

        prompt = self._format_chat_prompt(messages)
        result = generate(
            self.medgemma_model,
            self.medgemma_tokenizer,
            prompt=prompt,
            max_tokens=400,
            verbose=False
        )
        return self._clean_llm_output(result)

    def _parse_parent_response(self, text: str) -> Dict[str, Any]:
        """Parse the parent-facing MedGemma response (Pass 1) into components."""
        priority = TriagePriority.MODERATE
        critical_alerts = []
        parent_message = ""

        # Extract priority
        priority_match = re.search(r'PRIORITY:\s*(CRITICAL|URGENT|MODERATE|LOW)', text, re.IGNORECASE)
        if priority_match:
            priority_str = priority_match.group(1).upper()
            if priority_str in VALID_PRIORITIES:
                priority = TriagePriority(priority_str)

        # Extract alerts (everything between ALERTS: and PARENT MESSAGE:)
        alerts_match = re.search(
            r'ALERTS:\s*\n(.*?)(?=PARENT\s*MESSAGE:|$)', text, re.DOTALL | re.IGNORECASE
        )
        if alerts_match:
            alerts_text = alerts_match.group(1).strip()
            if alerts_text.lower() != 'none':
                for line in alerts_text.split('\n'):
                    line = line.strip().lstrip('- ').strip()
                    # Skip empty, 'none', and echoed template text
                    if (line and line.lower() != 'none'
                            and not line.startswith('<')
                            and 'each concerning' not in line.lower()):
                        critical_alerts.append(line)

        # Extract parent message
        parent_match = re.search(
            r'PARENT\s*MESSAGE:\s*\n?(.*)', text, re.DOTALL | re.IGNORECASE
        )
        if parent_match:
            parent_message = parent_match.group(1).strip()

        return {
            'priority': priority,
            'critical_alerts': critical_alerts,
            'recommendations': ["Continue monitoring and consult your healthcare provider if concerned."],
            'parent_message': parent_message,
            'specialist_message': '',
        }

    def _fallback_clinical_reasoning(self,
                                     vital_signs: Dict[str, Any],
                                     acoustic_indicators: Dict[str, Any],
                                     visual_indicators: Dict[str, Any] = None,
                                     guideline_context: str = "") -> Dict[str, Any]:
        """
        Simple rule-based fallback when MedGemma is unavailable.
        Provides basic triage without LLM reasoning.
        """
        critical_alerts = []

        # Check vital signs with None-safe access
        hr = vital_signs.get('heart_rate')
        rr = vital_signs.get('respiratory_rate')
        spo2 = vital_signs.get('spo2')

        if hr is not None:
            if hr > 180 or hr < 60:
                critical_alerts.append(f"Abnormal heart rate: {hr:.0f} BPM")
        if rr is not None:
            if rr > 60 or rr < 20:
                critical_alerts.append(f"Abnormal respiratory rate: {rr:.0f} breaths/min")
        if spo2 is not None:
            if spo2 < 92:
                critical_alerts.append(f"Low oxygen saturation: {spo2:.0f}%")

        # Check skin classification
        if visual_indicators:
            skin = visual_indicators.get('skin_assessment', {})
            classification = skin.get('classification', '')
            skin_confidence = skin.get('confidence', 0)
            if classification in ('eczema', 'chickenpox') and skin_confidence > 0.6:
                critical_alerts.append(
                    f"Skin condition detected: {classification} ({skin_confidence*100:.0f}% confidence)"
                )

        # Check acoustic findings
        hear_data = acoustic_indicators.get('hear', {})
        binary = hear_data.get('binary_classification', {})
        if binary.get('prediction') == 'Adventitious' and binary.get('confidence', 0) > 0.6:
            critical_alerts.append("Adventitious respiratory sounds detected")

        cry_data = acoustic_indicators.get('cry', {})
        if cry_data.get('prediction') == 'discomfort' and cry_data.get('confidence', 0) > 0.6:
            critical_alerts.append("Baby crying due to discomfort")

        # Determine priority from alert count and severity
        has_critical_vital = (spo2 is not None and spo2 < 88) or (hr is not None and (hr > 200 or hr < 40))
        if has_critical_vital:
            priority = TriagePriority.CRITICAL
        elif len(critical_alerts) >= 2:
            priority = TriagePriority.URGENT
        elif len(critical_alerts) == 1:
            priority = TriagePriority.MODERATE
        else:
            priority = TriagePriority.LOW

        # Generate basic messages
        if priority == TriagePriority.CRITICAL:
            parent_message = (
                "Our analysis has detected readings that need immediate medical attention. "
                "Please contact your pediatrician or call emergency services right away. "
                "Stay calm and keep your baby comfortable while you seek help."
            )
            recommendations = [
                "Seek immediate medical attention",
                "Call emergency services (911) if baby appears in distress",
                "Monitor vital signs continuously"
            ]
        elif priority == TriagePriority.URGENT:
            parent_message = (
                "Our analysis has found some readings outside the normal range for your baby. "
                "We recommend contacting your pediatrician within the next few hours for guidance. "
                "Keep monitoring your baby and note any changes."
            )
            recommendations = [
                "Contact your pediatrician within 1-2 hours",
                "Continue monitoring symptoms",
                "Note any changes in behavior or appearance"
            ]
        elif priority == TriagePriority.MODERATE:
            parent_message = (
                "Our analysis shows a finding that is worth mentioning to your pediatrician "
                "at your next visit. Your baby's overall readings are mostly within normal range. "
                "Keep an eye on any changes and don't hesitate to call if you're worried."
            )
            recommendations = [
                "Schedule an appointment with your healthcare provider",
                "Monitor symptoms for changes",
                "No immediate action required"
            ]
        else:
            parent_message = (
                "Great news! Our analysis shows your baby's readings are within normal range. "
                "Everything looks healthy based on the data we collected. Continue your regular "
                "care routine, and as always, consult your pediatrician if you have any concerns."
            )
            recommendations = [
                "Continue routine monitoring",
                "Maintain regular check-up schedule"
            ]

        specialist_message = (
            "Automated triage assessment (rule-based fallback - MedGemma unavailable). "
            "Clinical correlation recommended. "
            f"Alerts: {'; '.join(critical_alerts) if critical_alerts else 'None'}."
        )
        if guideline_context:
            specialist_message += f"\n\nApplicable guidelines:\n{guideline_context}"

        return {
            'priority': priority,
            'critical_alerts': critical_alerts,
            'recommendations': recommendations,
            'parent_message': parent_message,
            'specialist_message': specialist_message,
        }

    def _clean_llm_output(self, text: str) -> str:
        """Clean up LLM output by removing artifacts, code blocks, and deduplicating"""
        # Remove echoed prompt template fragments (model parroting instructions back)
        template_patterns = [
            r'<each concerning finding[^>]*>',
            r'<one of CRITICAL[^>]*>',
            r'<A message for parents[^>]*>',
            r'<Clinical note[^>]*>',
            r'<a clinical note[^>]*>',
            r'<Write a clinical[^>]*>',
        ]
        for pattern in template_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove special tokens
        for token in ['<|assistant|>', '<|end|>', '<|sample_code>', '<|sample_code|>',
                      '<|code|>', '<|endoftext|>', '<|im_end|>', '<|separator|>',
                      '<separator|>', '<|separator>']:
            text = text.replace(token, '')

        # Remove code blocks (```...```)
        text = re.sub(r'```[\s\S]*?```', '', text)

        # Remove lines that look like code
        lines = text.split('\n')
        cleaned_lines = []
        skip_code = False
        for line in lines:
            stripped = line.strip()
            if re.match(r'^(def |class |import |from |return |if __name__|    def |    return |print\(|>>>)', stripped):
                skip_code = True
                continue
            if skip_code and (not stripped or (stripped and not stripped.startswith('    '))):
                skip_code = False
            if skip_code:
                continue
            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines).strip()

        # Remove placeholder names
        text = re.sub(r"\[baby.?s?\s*name\]", "your baby", text, flags=re.IGNORECASE)
        text = re.sub(r"\[child.?s?\s*name\]", "your baby", text, flags=re.IGNORECASE)
        text = re.sub(r"\[infant.?s?\s*name\]", "your baby", text, flags=re.IGNORECASE)
        text = re.sub(r"\[name\]", "your baby", text, flags=re.IGNORECASE)

        # Strip markdown bold/italic formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)

        return text

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a chat prompt for MedGemma"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt
