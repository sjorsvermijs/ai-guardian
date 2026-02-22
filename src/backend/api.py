"""
FastAPI Backend for AI Guardian - Simplified Version
Uses batch frame processing to avoid threading issues
"""

import io
import logging
import wave
import base64
import time
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.rppg.pipeline import RPPGPipeline
from src.pipelines.cry.pipeline import CryPipeline
from src.pipelines.hear.pipeline import HeARPipeline
from src.core.config import config
from src.core.fusion_engine import FusionEngine, TriageReport

# Auto-detect Apple Silicon and prefer MLX VGA pipeline
import platform as _platform
_USE_MLX_VGA = (
    _platform.system() == "Darwin" and _platform.machine() == "arm64"
)

if _USE_MLX_VGA:
    try:
        from src.pipelines.vga.pipeline_mlx import VGAPipelineMLX as VGAPipeline
        logging.getLogger("ai_guardian").info("VGA: using MLX backend (Apple Silicon)")
    except ImportError:
        from src.pipelines.vga.pipeline import VGAPipeline
        logging.getLogger("ai_guardian").info("VGA: mlx-vlm not installed, falling back to PyTorch")
else:
    from src.pipelines.vga.pipeline import VGAPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ai_guardian")

app = FastAPI(title="AI Guardian API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global thread pool executor and pipelines
executor: Optional[ThreadPoolExecutor] = None
rppg_pipeline: Optional[RPPGPipeline] = None
cry_pipeline: Optional[CryPipeline] = None
hear_pipeline: Optional[HeARPipeline] = None
vga_pipeline: Optional[VGAPipeline] = None
fusion_engine: Optional[FusionEngine] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global executor, rppg_pipeline, cry_pipeline, hear_pipeline, vga_pipeline, fusion_engine
    logger.info("Starting AI Guardian API...")

    # Create thread pool executor (increased for parallel pipeline processing)
    executor = ThreadPoolExecutor(max_workers=4)
    logger.info("Thread pool ready (4 workers)")

    # Initialize rPPG pipeline
    logger.info("Initializing rPPG pipeline...")
    try:
        rppg_pipeline = RPPGPipeline(config.rppg_config)
        rppg_pipeline.initialize()
        logger.info("rPPG Pipeline ready (model: %s)", config.rppg_config['model_name'])
    except Exception as e:
        logger.error("Failed to initialize rPPG pipeline: %s", e)
        rppg_pipeline = None

    # Initialize Cry pipeline
    logger.info("Initializing Cry pipeline...")
    try:
        cry_pipeline = CryPipeline(config.get_pipeline_config('cry'))
        cry_pipeline.initialize()
        logger.info("Cry Pipeline ready (backend: %s)", config.cry_config['backend'])
    except Exception as e:
        logger.error("Failed to initialize Cry pipeline: %s", e)
        cry_pipeline = None

    # Initialize HeAR pipeline
    logger.info("Initializing HeAR pipeline...")
    try:
        hear_pipeline = HeARPipeline(config.get_pipeline_config('hear'))
        hear_pipeline.initialize()
        logger.info("HeAR Pipeline ready")
    except Exception as e:
        logger.error("Failed to initialize HeAR pipeline: %s", e)
        hear_pipeline = None

    # Initialize VGA pipeline (skin condition classification)
    logger.info("Initializing VGA pipeline...")
    try:
        vga_pipeline = VGAPipeline(config.get_pipeline_config('vga'))
        vga_pipeline.initialize()
        logger.info("VGA Pipeline ready")
    except Exception as e:
        logger.error("Failed to initialize VGA pipeline: %s", e)
        vga_pipeline = None

    # Initialize FusionEngine (MedGemma-powered AI reasoning)
    logger.info("Initializing FusionEngine (MedGemma 4B)...")
    try:
        fusion_engine = FusionEngine()
        fusion_engine.initialize()
        logger.info("FusionEngine ready (Medical AI Reasoning)")
    except Exception as e:
        logger.error("Failed to initialize FusionEngine: %s", e)
        fusion_engine = None

    logger.info(
        "AI Guardian API Ready! rPPG=%s Cry=%s HeAR=%s VGA=%s Fusion=%s",
        bool(rppg_pipeline), bool(cry_pipeline), bool(hear_pipeline),
        bool(vga_pipeline), bool(fusion_engine),
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global executor
    if executor:
        executor.shutdown(wait=False)
    logger.info("AI Guardian API shutdown")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.get("/ready")
async def ready_check():
    """Check if rPPG pipeline is ready"""
    return {
        "ready": rppg_pipeline is not None and rppg_pipeline.is_initialized,
        "model": config.rppg_config['model_name'] if rppg_pipeline else None
    }


class VideoProcessRequest(BaseModel):
    """Request body for video upload processing (all pipelines)"""
    video_frames: List[str]  # 300 base64-encoded frames for rPPG
    audio_data: str           # Base64-encoded WAV audio (16kHz mono)
    screenshots: List[str]    # 10 base64-encoded screenshots for VGA
    metadata: Dict[str, Any]  # FPS, duration, filename, etc.
    # Patient context for AI reasoning
    patient_age: Optional[int] = None      # Age in months (infants) or years
    patient_sex: Optional[str] = None      # Sex/gender
    parent_notes: Optional[str] = None     # Additional observations


class FrameBatchRequest(BaseModel):
    """Request body for batch frame processing"""
    frames: List[str]  # List of base64-encoded PNG frames

# Helper Functions for Video Processing

def decode_frames(frame_strings: List[str]) -> Optional[np.ndarray]:
    """Decode base64 frames to numpy array"""
    try:
        frames_list = []
        for i, frame_str in enumerate(frame_strings):
            try:
                if "," in frame_str:
                    frame_str = frame_str.split(",", 1)[1]

                frame_data = base64.b64decode(frame_str)
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    frames_list.append(frame)
            except Exception as e:
                logger.warning("Error decoding frame %d: %s", i, e)
                continue

        if len(frames_list) == 0:
            return None

        return np.array(frames_list)
    except Exception as e:
        logger.error("Error in decode_frames: %s", e)
        return None


def decode_audio(audio_string: str) -> Optional[np.ndarray]:
    """Decode base64 audio WAV to numpy array"""
    try:
        if "," in audio_string:
            audio_string = audio_string.split(",", 1)[1]

        audio_bytes = base64.b64decode(audio_string)

        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            logger.info("Audio: %dch, %dHz, %dbit, %d frames", channels, framerate, sample_width * 8, n_frames)

            audio_data = wav_file.readframes(n_frames)

            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            else:
                dtype = np.int32

            audio_array = np.frombuffer(audio_data, dtype=dtype)

            if dtype == np.uint8:
                audio_array = (audio_array.astype(np.float32) - 128) / 128.0
            else:
                audio_array = audio_array.astype(np.float32) / float(np.iinfo(dtype).max)

            if channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1)

            return audio_array

    except Exception as e:
        logger.error("Error decoding audio: %s", e, exc_info=True)
        return None


def decode_images(image_strings: List[str]) -> Optional[List[np.ndarray]]:
    """Decode base64 images to list of numpy arrays"""
    try:
        images = []
        for i, img_str in enumerate(image_strings):
            try:
                if "," in img_str:
                    img_str = img_str.split(",", 1)[1]

                img_data = base64.b64decode(img_str)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    images.append(img)
            except Exception as e:
                logger.warning("Error decoding image %d: %s", i, e)
                continue

        if len(images) == 0:
            return None

        return images
    except Exception as e:
        logger.error("Error in decode_images: %s", e)
        return None


@app.post("/api/process-video")
async def process_video_endpoint(request: VideoProcessRequest):
    """
    Process uploaded video with all pipelines (rPPG, Cry, HeAR, VGA)
    Accepts video frames, audio, and screenshots extracted by frontend
    """
    start_time = time.time()

    try:
        logger.info(
            "New video processing request: %d frames, %d screenshots, metadata=%s",
            len(request.video_frames), len(request.screenshots), request.metadata,
        )

        # Decode all inputs
        logger.info("Decoding inputs...")
        frames_array = decode_frames(request.video_frames)
        audio_array = decode_audio(request.audio_data)
        screenshots_list = decode_images(request.screenshots)

        if frames_array is None:
            return {"error": "Failed to decode video frames"}
        if audio_array is None:
            return {"error": "Failed to decode audio"}
        if screenshots_list is None:
            return {"error": "Failed to decode screenshots"}

        logger.info(
            "Decoded: %s frames, %s audio, %d screenshots",
            frames_array.shape, audio_array.shape, len(screenshots_list),
        )

        # Process all pipelines in parallel
        logger.info("Processing with all pipelines in parallel...")
        loop = asyncio.get_running_loop()

        results = await asyncio.gather(
            loop.run_in_executor(executor, process_rppg, frames_array),
            loop.run_in_executor(executor, process_cry, audio_array),
            loop.run_in_executor(executor, process_hear, audio_array),
            loop.run_in_executor(executor, process_vga, screenshots_list),
            return_exceptions=True
        )

        rppg_result, cry_result, hear_result, vga_result = results
        logger.info("\n================ PIPELINE RESULTS ================")
        def log_pipeline_result(label, result):
            if result is None:
                logger.info(f"{label} Result: None")
                return
            logger.info(f"{label} Result:")
            if hasattr(result, 'pipeline_name'):
                logger.info(f"  pipeline_name: {getattr(result, 'pipeline_name', None)}")
            if hasattr(result, 'timestamp'):
                logger.info(f"  timestamp: {getattr(result, 'timestamp', None)}")
            if hasattr(result, 'confidence'):
                logger.info(f"  confidence: {getattr(result, 'confidence', None)}")
            if hasattr(result, 'data'):
                logger.info(f"  data:")
                data = getattr(result, 'data', {})
                if isinstance(data, dict):
                    for k, v in data.items():
                        logger.info(f"    {k}: {v}")
                else:
                    logger.info(f"    {data}")
            if hasattr(result, 'warnings'):
                warnings = getattr(result, 'warnings', [])
                if warnings:
                    logger.info(f"  warnings:")
                    for w in warnings:
                        logger.info(f"    - {w}")
            if hasattr(result, 'errors'):
                errors = getattr(result, 'errors', [])
                if errors:
                    logger.info(f"  errors:")
                    for e in errors:
                        logger.info(f"    - {e}")
            if hasattr(result, 'metadata'):
                logger.info(f"  metadata:")
                metadata = getattr(result, 'metadata', {})
                if isinstance(metadata, dict):
                    for k, v in metadata.items():
                        logger.info(f"    {k}: {v}")
                else:
                    logger.info(f"    {metadata}")
            # If not a PipelineResult, fallback to string
            if not hasattr(result, 'pipeline_name'):
                logger.info(f"  {result}")

        log_pipeline_result("rPPG", rppg_result)
        log_pipeline_result("Cry", cry_result)
        log_pipeline_result("HeAR", hear_result)
        log_pipeline_result("VGA", vga_result)
        logger.info("=================================================")

        # Generate AI-powered triage report using FusionEngine
        triage_report = None
        if fusion_engine and fusion_engine.is_initialized:
            logger.info("Generating AI triage report with FusionEngine...")
            try:
                triage_report = fusion_engine.fuse(
                    hear_result=hear_result if not isinstance(hear_result, Exception) else None,
                    rppg_result=rppg_result if not isinstance(rppg_result, Exception) else None,
                    cry_result=cry_result if not isinstance(cry_result, Exception) else None,
                    vqa_result=vga_result if not isinstance(vga_result, Exception) else None,
                    patient_age=request.patient_age,
                    patient_sex=request.patient_sex,
                    parent_notes=request.parent_notes
                )
                logger.info("Triage report generated: %s", triage_report.priority.value)
            except Exception as e:
                logger.error("FusionEngine error: %s", e, exc_info=True)

        # Build response
        processing_time_ms = (time.time() - start_time) * 1000

        response = {
            "rppg": format_pipeline_result(rppg_result) if not isinstance(rppg_result, Exception) else {"error": str(rppg_result)},
            "cry": format_pipeline_result(cry_result) if not isinstance(cry_result, Exception) else {"error": str(cry_result)},
            "hear": format_pipeline_result(hear_result) if not isinstance(hear_result, Exception) else {"error": str(hear_result)},
            "vga": format_pipeline_result(vga_result) if not isinstance(vga_result, Exception) else {"error": str(vga_result)},
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now().isoformat()
        }

        if triage_report:
            response["triage"] = {
                "priority_level": triage_report.priority.value,
                "critical_alerts": triage_report.critical_alerts,
                "recommendations": triage_report.recommendations,
                "parent_message": triage_report.parent_message,
                "specialist_message": triage_report.specialist_message,
                "confidence_score": triage_report.confidence,
                "timestamp": triage_report.timestamp.isoformat(),
                "guidelines_used": triage_report.metadata.get('guidelines_retrieved', []),
            }

        logger.info("Processing complete in %.0fms", processing_time_ms)

        return response

    except Exception as e:
        logger.error("Error processing video: %s", e, exc_info=True)
        return {
            "error": str(e),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "timestamp": datetime.now().isoformat()
        }


def process_rppg(frames: np.ndarray) -> Optional[Dict[str, Any]]:
    """Process video frames with rPPG pipeline"""
    global rppg_pipeline
    try:
        if not rppg_pipeline or not rppg_pipeline.is_initialized:
            logger.warning("rPPG pipeline not initialized")
            return None

        logger.info("Processing rPPG: frames shape=%s, dtype=%s", frames.shape, frames.dtype)
        result = rppg_pipeline.process(frames)
        if hasattr(result, 'data') and result.data:
            logger.info("rPPG result: %s", {k: v for k, v in result.data.items() if k != 'embedding'})
        return result
    except Exception as e:
        logger.error("rPPG error: %s", e, exc_info=True)
        return None


def process_cry(audio: np.ndarray) -> Optional[Dict[str, Any]]:
    """Process audio with Cry pipeline"""
    global cry_pipeline
    try:
        if not cry_pipeline or not cry_pipeline.is_initialized:
            logger.warning("Cry pipeline not initialized")
            return None

        result = cry_pipeline.process(audio)
        return result
    except Exception as e:
        logger.error("Cry error: %s", e)
        return None


def process_hear(audio: np.ndarray) -> Optional[Dict[str, Any]]:
    """Process audio with HeAR pipeline"""
    global hear_pipeline
    try:
        if not hear_pipeline or not hear_pipeline.is_initialized:
            logger.warning("HeAR pipeline not initialized")
            return None

        result = hear_pipeline.process(audio)
        return result
    except Exception as e:
        logger.error("HeAR error: %s", e)
        return None


def process_vga(screenshots: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Process screenshots with VGA pipeline"""
    global vga_pipeline
    try:
        if not vga_pipeline or not vga_pipeline.is_initialized:
            logger.warning("VGA pipeline not initialized")
            return None

        result = vga_pipeline.process_batch(screenshots)
        return result
    except Exception as e:
        logger.error("VGA error: %s", e)
        return None


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to Python lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    else:
        return obj


def format_pipeline_result(result) -> Dict[str, Any]:
    """Format PipelineResult object to dictionary with JSON-serializable values"""
    if result is None:
        return {"error": "Pipeline not available"}

    if hasattr(result, 'data'):
        return convert_numpy_to_list(result.data)
    elif isinstance(result, dict):
        return convert_numpy_to_list(result)
    else:
        return {"error": "Invalid result format"}


def process_frames_batch(frames: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Process a batch of frames with rPPG pipeline
    Runs in background thread to avoid blocking
    """
    global rppg_pipeline

    try:
        if not rppg_pipeline or not rppg_pipeline.is_initialized:
            logger.warning("Pipeline not ready for batch processing")
            return None

        result = rppg_pipeline.process(frames)

        if result and result.data:
            response = {
                "heart_rate": result.data.get('heart_rate'),
                "signal_quality": result.data.get('signal_quality', 0.0),
                "respiratory_rate": result.data.get('respiratory_rate'),
                "timestamp": datetime.now().isoformat()
            }

            logger.debug("Built response: HR=%s, SQI=%s", response.get('heart_rate'), response.get('signal_quality'))

            hrv = result.data.get('heart_rate_variability', {})
            if hrv:
                response["hrv"] = {
                    "sdnn": hrv.get('sdnn', 0),
                    "rmssd": hrv.get('rmssd', 0),
                    "lf_hf": hrv.get('LF/HF', 0)
                }

            return response

        logger.warning("No result data returned from rPPG")
        return None

    except Exception as e:
        logger.error("Error in batch processing: %s", e, exc_info=True)
        return None


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Guardian API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
