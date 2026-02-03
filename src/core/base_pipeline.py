"""
Base Pipeline Interface
All pipelines (HeAR, rPPG, MedGemma VQA) inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PipelineResult:
    """Standard result format for all pipelines"""
    pipeline_name: str
    timestamp: datetime
    confidence: float  # 0.0 to 1.0
    data: Dict[str, Any]
    warnings: list[str]
    errors: list[str]
    metadata: Dict[str, Any]


class BasePipeline(ABC):
    """Abstract base class for all AI Guardian pipelines"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration
        
        Args:
            config: Dictionary containing pipeline-specific configuration
        """
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the pipeline (load models, setup resources)
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> PipelineResult:
        """
        Process input data through the pipeline
        
        Args:
            input_data: Input data specific to the pipeline
            
        Returns:
            PipelineResult: Standardized result object
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (models, memory, etc.)"""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data format
        
        Args:
            input_data: Data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return True
