"""VGA (Visual Grading Assessment) Pipeline Package"""

from .pipeline import VGAPipeline

try:
    from .pipeline_mlx import VGAPipelineMLX
except ImportError:
    VGAPipelineMLX = None

__all__ = ['VGAPipeline', 'VGAPipelineMLX']
