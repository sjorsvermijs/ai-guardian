"""
Fusion Engine - Combines outputs from HeAR, rPPG, and MedGemma VQA
Generates final triage report with high-confidence medical reasoning
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import warnings

from .base_pipeline import PipelineResult

# MedGemma imports
try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MEDGEMMA_AVAILABLE = True
except ImportError:
    MEDGEMMA_AVAILABLE = False
    warnings.warn("MedGemma (MLX) dependencies not available. Install: pip install mlx mlx-lm")


class TriagePriority(Enum):
    """Triage priority levels"""
    CRITICAL = "CRITICAL"  # Immediate intervention required
    URGENT = "URGENT"      # Needs prompt attention
    MODERATE = "MODERATE"  # Can wait for standard care
    LOW = "LOW"           # Non-urgent monitoring


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
    Combines and analyzes results from all three pipelines
    Uses weighted voting and cross-validation to produce high-confidence triage
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize fusion engine

        Args:
            config: Configuration including weights and thresholds
        """
        self.config = config or {}
        self.is_initialized = False

        # Pipeline weights for final decision
        self.weights = {
            'hear': self.config.get('hear_weight', 0.25),
            'rppg': self.config.get('rppg_weight', 0.35),
            'medgemma_vqa': self.config.get('vqa_weight', 0.40)
        }

        # Thresholds for triage priority
        self.thresholds = {
            'critical': 0.85,
            'urgent': 0.65,
            'moderate': 0.40
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
            print("Loading MedGemma model for clinical interpretation (MLX)...")
            
            # Use 4-bit quantized model for best M4 performance
            model_id = "mlx-community/medgemma-4b-it-4bit"
            
            self.medgemma_model, self.medgemma_tokenizer = load(model_id)
            
            print(f"✓ MedGemma loaded with MLX (M4 optimized)")
        except Exception as e:
            error_msg = str(e)
            if "gated repo" in error_msg or "authenticated" in error_msg:
                print(f"\n⚠️ MedGemma requires authentication:")
                print(f"   1. Request access: https://huggingface.co/mlx-community/medgemma-4b-it-4bit")
                print(f"   2. Get token: https://huggingface.co/settings/tokens")
                print(f"   3. Login: huggingface-cli login")
            else:
                print(f"⚠️ Failed to load MedGemma: {e}")
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
        Fuse results from all pipelines into a comprehensive triage report

        Args:
            hear_result: Result from HeAR pipeline
            rppg_result: Result from rPPG pipeline
            cry_result: Result from Cry classification pipeline
            vqa_result: Result from VGA/MedGemma VQA pipeline
            patient_age: Patient age in months (for infants) or years
            patient_sex: Patient sex/gender
            parent_notes: Additional observations from parents

        Returns:
            TriageReport: Comprehensive medical assessment
        """
        timestamp = datetime.now()
        critical_alerts = []
        recommendations = []
        
        # Aggregate confidence scores
        confidences = []
        if hear_result and hear_result.confidence > 0:
            confidences.append(('hear', hear_result.confidence))
        if rppg_result and rppg_result.confidence > 0:
            confidences.append(('rppg', rppg_result.confidence))
        if vqa_result and vqa_result.confidence > 0:
            confidences.append(('medgemma_vqa', vqa_result.confidence))
        
        # Calculate weighted confidence
        overall_confidence = self._calculate_weighted_confidence(confidences)
        
        # Extract vital signs from rPPG
        vital_signs = {}
        if rppg_result:
            vital_signs = rppg_result.data.copy()
            
            # Check for abnormal vital signs
            hr = vital_signs.get('heart_rate') or 0
            rr = vital_signs.get('respiratory_rate')
            spo2 = vital_signs.get('spo2') or 100

            if hr and (hr > 120 or hr < 50):
                critical_alerts.append(f"Abnormal heart rate: {hr:.1f} BPM")
            
            # Respiratory rate assessment (if available from HRV)
            if rr is not None:
                if rr > 30:
                    critical_alerts.append(f"Severe tachypnea: {rr:.1f} breaths/min (normal: 12-20)")
                elif rr > 20:
                    critical_alerts.append(f"Tachypnea: {rr:.1f} breaths/min (normal: 12-20)")
                elif rr < 8:
                    critical_alerts.append(f"Severe bradypnea: {rr:.1f} breaths/min (normal: 12-20)")
                elif rr < 12:
                    recommendations.append(f"Mild bradypnea: {rr:.1f} breaths/min - monitor breathing")
            
            if spo2 < 90:
                critical_alerts.append(f"Low oxygen saturation: {spo2}%")
        
        # Extract acoustic indicators from HeAR and Cry
        acoustic_indicators = {}
        if hear_result:
            acoustic_indicators['hear'] = hear_result.data.copy()
            if hear_result.data.get('cough_detected'):
                critical_alerts.append("Persistent coughing detected")

        if cry_result:
            acoustic_indicators['cry'] = cry_result.data.copy()
            cry_type = cry_result.data.get('prediction')
            cry_confidence = cry_result.data.get('confidence', 0)
            if cry_type and cry_confidence > 0.6:
                if cry_type == 'discomfort':
                    critical_alerts.append(f"Baby crying due to discomfort ({cry_confidence*100:.0f}% confidence)")
                elif cry_type == 'hungry':
                    recommendations.append(f"Baby may be hungry ({cry_confidence*100:.0f}% confidence)")
                elif cry_type in ['belly', 'cold']:
                    critical_alerts.append(f"Baby crying: {cry_type} issue detected ({cry_confidence*100:.0f}% confidence)")
        
        # Extract visual indicators from VQA
        visual_indicators = {}
        if vqa_result:
            visual_indicators = vqa_result.data.copy()
            vqa_flags = vqa_result.data.get('critical_flags', [])
            critical_alerts.extend(vqa_flags)
        
        # Cross-validate findings
        cross_validated_alerts = self._cross_validate(
            hear_result, rppg_result, vqa_result
        )
        critical_alerts.extend(cross_validated_alerts)
        
        # Determine triage priority
        priority = self._determine_priority(
            critical_alerts, 
            overall_confidence,
            vital_signs
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            priority, 
            critical_alerts,
            vital_signs
        )
        
        # Build clinical data summary for MedGemma
        clinical_data = self._build_clinical_data(
            vital_signs, acoustic_indicators, visual_indicators,
            critical_alerts, patient_age, patient_sex, parent_notes
        )

        # Generate both messages in a single LLM call for speed
        parent_message, specialist_message = self._generate_triage_messages(
            clinical_data, priority, recommendations
        )

        # Create summary
        summary = self._generate_summary(
            priority,
            len(critical_alerts),
            vital_signs
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
            critical_alerts=list(set(critical_alerts)),  # Remove duplicates
            recommendations=recommendations,
            individual_results={
                'hear': hear_result,
                'rppg': rppg_result,
                'cry': cry_result,
                'vga': vqa_result
            },
            metadata={
                'fusion_engine_version': '0.1.0',
                'num_pipelines_active': len(confidences),
                'weights_used': self.weights,
                'medgemma_enabled': self.use_medgemma
            }
        )
    
    def _calculate_weighted_confidence(self, confidences: List[tuple]) -> float:
        """Calculate weighted average of pipeline confidences"""
        if not confidences:
            return 0.0
        
        total_weight = sum(self.weights[name] for name, _ in confidences)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            self.weights[name] * conf 
            for name, conf in confidences
        )
        
        return weighted_sum / total_weight
    
    def _cross_validate(self, 
                       hear_result: PipelineResult,
                       rppg_result: PipelineResult,
                       vqa_result: PipelineResult) -> List[str]:
        """Cross-validate findings across pipelines"""
        alerts = []
        
        # Example: If both VQA sees cyanosis and rPPG shows low SpO2
        if vqa_result and rppg_result:
            vqa_flags = vqa_result.data.get('critical_flags', [])
            spo2 = rppg_result.data.get('spo2', 100)
            
            if any('cyanosis' in flag.lower() for flag in vqa_flags) and spo2 < 92:
                alerts.append("VALIDATED: Hypoxemia confirmed by visual and vital signs")
        
        # Example: Respiratory distress seen visually and acoustically
        if hear_result and vqa_result:
            breathing_pattern = hear_result.data.get('breathing_pattern', 'normal')
            vqa_flags = vqa_result.data.get('critical_flags', [])
            
            if breathing_pattern != 'normal' and any('respiratory' in flag.lower() for flag in vqa_flags):
                alerts.append("VALIDATED: Respiratory distress confirmed by audio and visual")
        
        return alerts
    
    def _build_clinical_data(self,
                            vital_signs: Dict[str, Any],
                            acoustic_indicators: Dict[str, Any],
                            visual_indicators: Dict[str, Any],
                            critical_alerts: List[str],
                            patient_age: int = None,
                            patient_sex: str = None,
                            parent_notes: str = None) -> str:
        """Build clinical data summary string used by both message generators"""
        clinical_summary = "INFANT HEALTH ASSESSMENT - Multi-Modal Analysis:\n\n"

        # Patient Context
        clinical_summary += "Patient Information:\n"
        if patient_age is not None:
            age_str = f"{patient_age} months old" if patient_age < 24 else f"{patient_age//12} years old"
            clinical_summary += f"- Age: {age_str}\n"
        if patient_sex:
            clinical_summary += f"- Sex: {patient_sex}\n"
        if parent_notes:
            clinical_summary += f"- Parent Observations: {parent_notes}\n"
        clinical_summary += "\n"

        # Vital Signs
        if vital_signs:
            clinical_summary += "Vital Signs (via rPPG - contactless video-based):\n"
            hr = vital_signs.get('heart_rate')
            if hr is not None:
                clinical_summary += f"- Heart Rate: {hr:.1f} BPM\n"
            rr = vital_signs.get('respiratory_rate')
            if rr is not None:
                clinical_summary += f"- Respiratory Rate: {rr:.1f} breaths/min\n"
            spo2 = vital_signs.get('spo2')
            if spo2 is not None:
                clinical_summary += f"- SpO2: {spo2:.1f}%\n"
            sqi = vital_signs.get('signal_quality')
            if sqi is not None:
                sqi_desc = "Excellent" if sqi > 0.8 else "Good" if sqi > 0.6 else "Acceptable" if sqi > 0.5 else "Poor"
                clinical_summary += f"- Signal Quality: {sqi:.1%} ({sqi_desc})\n"
            clinical_summary += "\n"

        # Acoustic Indicators (HeAR + Cry)
        if acoustic_indicators:
            clinical_summary += "Acoustic Analysis:\n"
            if 'cry' in acoustic_indicators:
                cry_data = acoustic_indicators['cry']
                if 'prediction' in cry_data:
                    confidence = cry_data.get('confidence', 0) * 100
                    clinical_summary += f"- Cry Classification: {cry_data['prediction']} ({confidence:.0f}% confidence)\n"
                    probs = cry_data.get('probabilities', {})
                    if probs:
                        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                        clinical_summary += f"  Top predictions: {', '.join(f'{k}={v*100:.0f}%' for k,v in top_3)}\n"

            if 'hear' in acoustic_indicators:
                hear_data = acoustic_indicators['hear']
                if 'binary_classification' in hear_data:
                    bc = hear_data['binary_classification']
                    clinical_summary += f"- Respiratory Sounds: {bc['prediction']} ({bc['confidence']*100:.0f}% confidence)\n"
                if 'multiclass_classification' in hear_data:
                    mc = hear_data['multiclass_classification']
                    clinical_summary += f"- Sound Classification: {mc['prediction']} ({mc['confidence']*100:.0f}% confidence)\n"
            clinical_summary += "\n"

        # Visual Indicators
        if visual_indicators:
            clinical_summary += "Visual Assessment:\n"
            for key, value in visual_indicators.items():
                if key not in ('status', 'message'):
                    clinical_summary += f"- {key.replace('_', ' ').title()}: {value}\n"
            clinical_summary += "\n"

        # Critical Alerts
        if critical_alerts:
            clinical_summary += "Flagged Alerts:\n"
            for alert in critical_alerts:
                clinical_summary += f"- {alert}\n"
            clinical_summary += "\n"

        return clinical_summary

    def _generate_triage_messages(self,
                                  clinical_data: str,
                                  priority: TriagePriority,
                                  recommendations: List[str]) -> tuple:
        """Generate both parent and specialist messages in a single LLM call"""
        if not self.use_medgemma or not self.medgemma_model:
            return ("", "")

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a pediatric medicine AI assistant. You will write TWO separate messages "
                        "based on the same clinical data. Follow the exact format below.\n\n"
                        "INFANT NORMAL RANGES:\n"
                        "- Heart Rate: 100-160 BPM (newborns), 90-150 BPM (infants)\n"
                        "- Respiratory Rate: 30-60 breaths/min (newborns), 25-40 (infants)"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"{clinical_data}\n"
                        f"Triage priority: {priority.value}\n"
                        f"Recommendations: {'; '.join(recommendations)}\n\n"
                        "Write exactly TWO sections separated by '---SEPARATOR---'.\n\n"
                        "CRITICAL RULES:\n"
                        "- Do NOT use placeholders like [baby's name], [child], or [name]. "
                        "Refer to the baby as 'your baby' or 'the infant'.\n"
                        "- Do NOT generate any code, functions, or programming syntax.\n"
                        "- Write ONLY natural language text.\n\n"
                        "SECTION 1 - FOR PARENTS:\n"
                        "Write a short, warm message (max 100 words) in simple language. "
                        "Explain what was found, whether to worry, and what to do next. "
                        "No headers, no bullet points, just natural paragraphs.\n\n"
                        "---SEPARATOR---\n\n"
                        "SECTION 2 - CLINICAL NOTE:\n"
                        "Write a brief clinical assessment (max 150 words) using medical terminology. "
                        "Include: findings summary, clinical significance with pediatric normal ranges, "
                        "differential considerations, and recommended follow-up. "
                        "Do NOT use markdown formatting like ** or #."
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

            # Split into two messages
            if '---SEPARATOR---' in result:
                parts = result.split('---SEPARATOR---', 1)
                parent_msg = parts[0].strip()
                specialist_msg = parts[1].strip()
            else:
                # Fallback: use entire output as parent message
                parent_msg = result
                specialist_msg = ""

            # Clean up section headers if the model included them
            for prefix in ['SECTION 1 - FOR PARENTS:', 'SECTION 1:', 'FOR PARENTS:']:
                if parent_msg.upper().startswith(prefix.upper()):
                    parent_msg = parent_msg[len(prefix):].strip()
            for prefix in ['SECTION 2 - CLINICAL NOTE:', 'SECTION 2:', 'CLINICAL NOTE:']:
                if specialist_msg.upper().startswith(prefix.upper()):
                    specialist_msg = specialist_msg[len(prefix):].strip()

            return (parent_msg, specialist_msg)

        except Exception as e:
            print(f"⚠️ Triage message generation failed: {e}")
            import traceback
            traceback.print_exc()
            return ("", "")

    def _clean_llm_output(self, text: str) -> str:
        """Clean up LLM output by removing artifacts, code blocks, and deduplicating"""
        import re

        # Remove special tokens
        for token in ['<|assistant|>', '<|end|>', '<|sample_code>', '<|sample_code|>',
                      '<|code|>', '<|endoftext|>', '<|im_end|>']:
            text = text.replace(token, '')

        # Remove code blocks (```...```)
        text = re.sub(r'```[\s\S]*?```', '', text)

        # Remove lines that look like code (def, import, class, return, etc.)
        lines = text.split('\n')
        cleaned_lines = []
        skip_code = False
        for line in lines:
            stripped = line.strip()
            # Skip lines that are clearly code
            if re.match(r'^(def |class |import |from |return |if __name__|    def |    return |print\(|>>>)', stripped):
                skip_code = True
                continue
            # If we were skipping code and hit an empty line or non-indented text, stop skipping
            if skip_code and (not stripped or (stripped and not stripped.startswith('    '))):
                skip_code = False
            if skip_code:
                continue
            cleaned_lines.append(line)

        # Remove duplicate lines
        seen = set()
        unique_lines = []
        for line in cleaned_lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                unique_lines.append(line)
                seen.add(stripped)
            elif not stripped:
                unique_lines.append(line)

        text = '\n'.join(unique_lines).strip()

        # Remove placeholder names like [baby's name], [child's name], etc.
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
    
    def _determine_priority(self,
                          critical_alerts: List[str],
                          confidence: float,
                          vital_signs: Dict[str, Any]) -> TriagePriority:
        """Determine triage priority based on all factors"""
        
        # Critical conditions
        critical_keywords = ['cyanosis', 'hypoxemia', 'validated']
        if any(any(keyword in alert.lower() for keyword in critical_keywords) 
               for alert in critical_alerts):
            return TriagePriority.CRITICAL
        
        # Check vital signs (use 'or' fallback since values may be explicitly None)
        spo2 = vital_signs.get('spo2') or 100
        hr = vital_signs.get('heart_rate') or 70
        rr = vital_signs.get('respiratory_rate') or 15

        if spo2 < 88 or hr > 140 or hr < 40 or rr > 35:
            return TriagePriority.CRITICAL
        
        # Urgent conditions
        if len(critical_alerts) >= 2 or spo2 < 92 or hr > 120:
            return TriagePriority.URGENT
        
        # Moderate conditions
        if len(critical_alerts) >= 1 or confidence > 0.6:
            return TriagePriority.MODERATE
        
        return TriagePriority.LOW
    
    def _generate_recommendations(self,
                                 priority: TriagePriority,
                                 alerts: List[str],
                                 vital_signs: Dict[str, Any]) -> List[str]:
        """Generate medical recommendations based on findings"""
        recommendations = []
        
        if priority == TriagePriority.CRITICAL:
            recommendations.append("IMMEDIATE medical attention required")
            recommendations.append("Call emergency services (911)")
            recommendations.append("Monitor vital signs continuously")
        elif priority == TriagePriority.URGENT:
            recommendations.append("Seek medical attention within 1-2 hours")
            recommendations.append("Continue monitoring symptoms")
        elif priority == TriagePriority.MODERATE:
            recommendations.append("Schedule appointment with healthcare provider")
            recommendations.append("Monitor symptoms for changes")
        else:
            recommendations.append("Continue routine monitoring")
            recommendations.append("Consult healthcare provider if symptoms worsen")
        
        return recommendations
    
    def _generate_summary(self,
                         priority: TriagePriority,
                         num_alerts: int,
                         vital_signs: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        
        priority_text = priority.value
        hr = vital_signs.get('heart_rate', 'N/A')
        rr = vital_signs.get('respiratory_rate', 'N/A')
        spo2 = vital_signs.get('spo2', 'N/A')
        
        summary = f"Triage Priority: {priority_text}. "
        summary += f"Detected {num_alerts} concerning indicator(s). "
        summary += f"Vital signs - HR: {hr}, RR: {rr}, SpO2: {spo2}."
        
        return summary
