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
        
        if self.use_medgemma:
            self._initialize_medgemma()
    
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
             vqa_result: PipelineResult = None) -> TriageReport:
        """
        Fuse results from all pipelines into a comprehensive triage report
        
        Args:
            hear_result: Result from HeAR pipeline
            rppg_result: Result from rPPG pipeline
            vqa_result: Result from MedGemma VQA pipeline
            
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
            hr = vital_signs.get('heart_rate', 0)
            rr = vital_signs.get('respiratory_rate')
            spo2 = vital_signs.get('spo2', 100)
            
            if hr > 120 or hr < 50:
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
        
        # Extract acoustic indicators from HeAR
        acoustic_indicators = {}
        if hear_result:
            acoustic_indicators = hear_result.data.copy()
            if hear_result.data.get('cough_detected'):
                critical_alerts.append("Persistent coughing detected")
        
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
        
        # Generate MedGemma clinical interpretation
        medgemma_interpretation = self._generate_medgemma_interpretation(
            vital_signs,
            acoustic_indicators,
            visual_indicators,
            critical_alerts
        )
        
        # Create summary
        summary = self._generate_summary(
            priority,
            len(critical_alerts),
            vital_signs
        )
        
        # Add MedGemma interpretation to summary if available
        if medgemma_interpretation:
            summary = f"{summary}\n\n--- MedGemma Clinical Interpretation ---\n{medgemma_interpretation}"
        
        return TriageReport(
            timestamp=timestamp,
            priority=priority,
            confidence=overall_confidence,
            summary=summary,
            vital_signs=vital_signs,
            acoustic_indicators=acoustic_indicators,
            visual_indicators=visual_indicators,
            critical_alerts=list(set(critical_alerts)),  # Remove duplicates
            recommendations=recommendations,
            individual_results={
                'hear': hear_result,
                'rppg': rppg_result,
                'medgemma_vqa': vqa_result
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
    
    def _generate_medgemma_interpretation(self,
                                         vital_signs: Dict[str, Any],
                                         acoustic_indicators: Dict[str, Any],
                                         visual_indicators: Dict[str, Any],
                                         critical_alerts: List[str]) -> Optional[str]:
        """Generate clinical interpretation using MedGemma"""
        if not self.use_medgemma or not self.medgemma_model:
            return None
        
        try:
            # Build clinical summary for MedGemma
            clinical_summary = "Patient Assessment via Remote Vital Signs Monitoring:\n\n"
            clinical_summary += "**Measurement Method:**\n"
            clinical_summary += "- Remote photoplethysmography (rPPG) via webcam - non-contact optical measurement\n"
            clinical_summary += "- Signal quality indicates measurement reliability\n\n"
            
            # Vital Signs
            if vital_signs:
                clinical_summary += "**Vital Signs:**\n"
                if 'heart_rate' in vital_signs:
                    clinical_summary += f"- Heart Rate: {vital_signs['heart_rate']:.1f} BPM\n"
                if 'respiratory_rate' in vital_signs:
                    clinical_summary += f"- Respiratory Rate: {vital_signs['respiratory_rate']:.1f} breaths/min\n"
                if 'spo2' in vital_signs:
                    clinical_summary += f"- SpO2: {vital_signs['spo2']:.1f}%\n"
                if 'signal_quality' in vital_signs:
                    sqi = vital_signs['signal_quality']
                    sqi_desc = "Excellent" if sqi > 0.8 else "Good" if sqi > 0.6 else "Acceptable" if sqi > 0.5 else "Poor"
                    clinical_summary += f"- Signal Quality: {sqi:.1%} ({sqi_desc})\n"
                clinical_summary += "\n"
            
            # Acoustic Indicators
            if acoustic_indicators:
                clinical_summary += "Acoustic Analysis:\n"
                for key, value in acoustic_indicators.items():
                    clinical_summary += f"- {key.replace('_', ' ').title()}: {value}\n"
                clinical_summary += "\n"
            
            # Visual Indicators
            if visual_indicators:
                clinical_summary += "Visual Observations:\n"
                for key, value in visual_indicators.items():
                    clinical_summary += f"- {key.replace('_', ' ').title()}: {value}\n"
                clinical_summary += "\n"
            
            # Critical Alerts
            if critical_alerts:
                clinical_summary += "Critical Alerts:\n"
                for alert in critical_alerts:
                    clinical_summary += f"- {alert}\n"
                clinical_summary += "\n"
            
            # Create prompt for MedGemma
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert emergency medicine physician interpreting data from a remote vital signs monitoring system. "
                        "The measurements are obtained via rPPG (remote photoplethysmography) using a webcam - this is a non-contact, "
                        "camera-based technology that detects subtle color changes in facial skin to estimate vital signs. "
                        "Signal quality indicates measurement reliability. "
                        "\n\nNORMAL RANGES:\n"
                        "- Signal Quality: >50% is acceptable, >70% is good\n"
                        "\nProvide concise, accurate clinical interpretation. Do not flag normal values as abnormal."
                    )
                },
                {
                    "role": "user",
                    "content": f"{clinical_summary}\n\nProvide:\n1. Clinical interpretation (2-3 sentences)\n2. Triage priority: CRITICAL/URGENT/MODERATE/LOW\n3. Recommendations (if needed)"
                }
            ]
            
            # Format prompt
            prompt = self._format_chat_prompt(messages)
            
            # Generate interpretation using MLX
            interpretation = generate(
                self.medgemma_model,
                self.medgemma_tokenizer,
                prompt=prompt,
                max_tokens=300,
                verbose=False
            )
            
            # Clean up repetitive assistant tokens
            interpretation = interpretation.replace('<|assistant|>', '').strip()
            # Remove duplicate paragraphs
            lines = interpretation.split('\n')
            seen = set()
            unique_lines = []
            for line in lines:
                if line.strip() and line.strip() not in seen:
                    unique_lines.append(line)
                    seen.add(line.strip())
                elif not line.strip():  # Keep empty lines for formatting
                    unique_lines.append(line)
            interpretation = '\n'.join(unique_lines)
            
            return interpretation.strip()
            
        except Exception as e:
            print(f"⚠️ MedGemma interpretation failed: {e}")
            return None
    
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
        
        # Check vital signs
        spo2 = vital_signs.get('spo2', 100)
        hr = vital_signs.get('heart_rate', 70)
        rr = vital_signs.get('respiratory_rate', 15)
        
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
