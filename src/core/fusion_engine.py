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

    def initialize(self):
        """Initialize the fusion engine and load MedGemma model"""
        if self.use_medgemma:
            self._initialize_medgemma()
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

        # Let MedGemma handle all clinical reasoning
        if self.use_medgemma and self.medgemma_model:
            triage = self._medgemma_clinical_reasoning(clinical_data)
        else:
            triage = self._fallback_clinical_reasoning(vital_signs, acoustic_indicators)

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
                'fusion_engine_version': '0.2.0',
                'num_pipelines_active': len(confidences),
                'weights_used': self.weights,
                'medgemma_enabled': self.use_medgemma and self.medgemma_model is not None,
                'reasoning_mode': 'medgemma' if (self.use_medgemma and self.medgemma_model) else 'fallback'
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

        # Visual Indicators
        if visual_indicators:
            parts.append("Visual Assessment:")
            for key, value in visual_indicators.items():
                if key not in ('status', 'message', 'image_shapes', 'num_screenshots_analyzed', 'skin_assessment'):
                    parts.append(f"- {key.replace('_', ' ').title()}: {value}")
            parts.append("")

        return "\n".join(parts)

    def _medgemma_clinical_reasoning(self, clinical_data: str) -> Dict[str, Any]:
        """
        Single MedGemma call that performs ALL clinical reasoning:
        priority, alerts, recommendations, parent message, specialist message.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
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
                },
                {
                    "role": "user",
                    "content": (
                        f"{clinical_data}\n"
                        "Based on the above data, produce your assessment in EXACTLY this format "
                        "(use the exact headers and delimiters shown):\n\n"
                        "PRIORITY: <one of CRITICAL, URGENT, MODERATE, LOW>\n\n"
                        "ALERTS:\n"
                        "- <each concerning finding on its own line, or 'None' if nothing concerning>\n\n"
                        "RECOMMENDATIONS:\n"
                        "- <each recommendation on its own line>\n\n"
                        "---SEPARATOR---\n\n"
                        "PARENT MESSAGE:\n"
                        "<A reassuring message for parents in simple language (max 100 words). "
                        "Explain findings and what to do next. No headers or bullet points, just paragraphs.>\n\n"
                        "---SEPARATOR---\n\n"
                        "SPECIALIST MESSAGE:\n"
                        "<Brief clinical note (max 150 words) using medical terminology. "
                        "Include: findings summary, clinical significance with pediatric normal ranges, "
                        "differential considerations, and recommended follow-up. No markdown formatting.>"
                    )
                }
            ]

            prompt = self._format_chat_prompt(messages)
            result = generate(
                self.medgemma_model,
                self.medgemma_tokenizer,
                prompt=prompt,
                max_tokens=450,
                verbose=False
            )
            result = self._clean_llm_output(result)

            return self._parse_medgemma_response(result)

        except Exception as e:
            logger.error("MedGemma clinical reasoning failed: %s", e, exc_info=True)
            return self._fallback_clinical_reasoning({}, {})

    def _parse_medgemma_response(self, text: str) -> Dict[str, Any]:
        """Parse the structured MedGemma response into components"""
        priority = TriagePriority.MODERATE
        critical_alerts = []
        recommendations = []
        parent_message = ""
        specialist_message = ""

        # Extract priority
        priority_match = re.search(r'PRIORITY:\s*(CRITICAL|URGENT|MODERATE|LOW)', text, re.IGNORECASE)
        if priority_match:
            priority_str = priority_match.group(1).upper()
            if priority_str in VALID_PRIORITIES:
                priority = TriagePriority(priority_str)

        # Extract alerts
        alerts_match = re.search(r'ALERTS:\s*\n(.*?)(?=\nRECOMMENDATIONS:)', text, re.DOTALL | re.IGNORECASE)
        if alerts_match:
            alerts_text = alerts_match.group(1).strip()
            if alerts_text.lower() != 'none':
                for line in alerts_text.split('\n'):
                    line = line.strip().lstrip('- ').strip()
                    if line and line.lower() != 'none':
                        critical_alerts.append(line)

        # Extract recommendations
        recs_match = re.search(r'RECOMMENDATIONS:\s*\n(.*?)(?=---SEPARATOR---|$)', text, re.DOTALL | re.IGNORECASE)
        if recs_match:
            recs_text = recs_match.group(1).strip()
            for line in recs_text.split('\n'):
                line = line.strip().lstrip('- ').strip()
                if line:
                    recommendations.append(line)

        # Split parent and specialist messages by separator
        parts = text.split('---SEPARATOR---')
        if len(parts) >= 3:
            # parts[0] = priority/alerts/recs, parts[1] = parent msg, parts[2] = specialist msg
            parent_message = self._extract_message_body(parts[1], 'PARENT MESSAGE:')
            specialist_message = self._extract_message_body(parts[2], 'SPECIALIST MESSAGE:')
        elif len(parts) == 2:
            parent_message = self._extract_message_body(parts[1], 'PARENT MESSAGE:')

        # Ensure we have at least basic recommendations if none were parsed
        if not recommendations:
            recommendations = ["Continue monitoring and consult your healthcare provider if concerned."]

        return {
            'priority': priority,
            'critical_alerts': critical_alerts,
            'recommendations': recommendations,
            'parent_message': parent_message,
            'specialist_message': specialist_message,
        }

    def _extract_message_body(self, text: str, header: str) -> str:
        """Extract message body after a header, cleaning up artifacts"""
        text = text.strip()
        # Remove the header if present
        for prefix in [header, header.rstrip(':')]:
            if text.upper().startswith(prefix.upper()):
                text = text[len(prefix):].strip()
        return text

    def _fallback_clinical_reasoning(self,
                                     vital_signs: Dict[str, Any],
                                     acoustic_indicators: Dict[str, Any]) -> Dict[str, Any]:
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

        return {
            'priority': priority,
            'critical_alerts': critical_alerts,
            'recommendations': recommendations,
            'parent_message': parent_message,
            'specialist_message': specialist_message,
        }

    def _clean_llm_output(self, text: str) -> str:
        """Clean up LLM output by removing artifacts, code blocks, and deduplicating"""
        # Remove special tokens
        for token in ['<|assistant|>', '<|end|>', '<|sample_code>', '<|sample_code|>',
                      '<|code|>', '<|endoftext|>', '<|im_end|>']:
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
