"""
FastAPI Backend for AI Guardian - Simplified Version
Uses batch frame processing to avoid threading issues
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.rppg.pipeline import RPPGPipeline
from src.pipelines.cry.pipeline import CryPipeline
from src.pipelines.hear.pipeline import HeARPipeline
from src.pipelines.vga.pipeline import VGAPipeline
from src.core.config import config
from src.core.fusion_engine import FusionEngine, TriageReport
import io
import wave

app = FastAPI(title="AI Guardian API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
    print("🚀 Starting AI Guardian API...")
    print("="*60)

    # Create thread pool executor (increased for parallel pipeline processing)
    executor = ThreadPoolExecutor(max_workers=4)
    print("✓ Thread pool ready (4 workers)")

    # Initialize rPPG pipeline
    print("\n📊 Initializing rPPG pipeline...")
    try:
        rppg_pipeline = RPPGPipeline(config.rppg_config)
        rppg_pipeline.initialize()
        print(f"✓ rPPG Pipeline ready (model: {config.rppg_config['model_name']})")
    except Exception as e:
        print(f"❌ Failed to initialize rPPG pipeline: {e}")
        rppg_pipeline = None

    # Initialize Cry pipeline
    print("\n👶 Initializing Cry pipeline...")
    try:
        cry_pipeline = CryPipeline(config.get_pipeline_config('cry'))
        cry_pipeline.initialize()
        print(f"✓ Cry Pipeline ready (backend: {config.cry_config['backend']})")
    except Exception as e:
        print(f"❌ Failed to initialize Cry pipeline: {e}")
        cry_pipeline = None

    # Initialize HeAR pipeline
    print("\n🫁 Initializing HeAR pipeline...")
    try:
        hear_pipeline = HeARPipeline(config.get_pipeline_config('hear'))
        hear_pipeline.initialize()
        print(f"✓ HeAR Pipeline ready")
    except Exception as e:
        print(f"❌ Failed to initialize HeAR pipeline: {e}")
        hear_pipeline = None

    # Initialize VGA pipeline (placeholder stub)
    print("\n👁️ Initializing VGA pipeline...")
    try:
        vga_pipeline = VGAPipeline(config.get_pipeline_config('vga'))
        vga_pipeline.initialize()
        print(f"✓ VGA Pipeline ready (placeholder stub)")
    except Exception as e:
        print(f"❌ Failed to initialize VGA pipeline: {e}")
        vga_pipeline = None

    # Initialize FusionEngine (MedGemma-powered AI reasoning)
    print("\n🧠 Initializing FusionEngine (MedGemma 4B)...")
    try:
        fusion_engine = FusionEngine()
        fusion_engine.initialize()
        print(f"✓ FusionEngine ready (Medical AI Reasoning)")
    except Exception as e:
        print(f"❌ Failed to initialize FusionEngine: {e}")
        fusion_engine = None

    print("\n" + "="*60)
    print("🎉 AI Guardian API Ready!")
    print("   - rPPG:", "✓" if rppg_pipeline else "✗")
    print("   - Cry:", "✓" if cry_pipeline else "✗")
    print("   - HeAR:", "✓" if hear_pipeline else "✗")
    print("   - VGA:", "✓" if vga_pipeline else "✗")
    print("   - FusionEngine:", "✓" if fusion_engine else "✗")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global executor
    if executor:
        executor.shutdown(wait=False)
    print("👋 AI Guardian API shutdown")


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


@app.post("/api/process-frames")
async def process_frames_endpoint(request: FrameBatchRequest):
    """
    Process a batch of frames collected by the frontend
    Frontend collects frames locally at 30 FPS, sends all at once
    """
    try:
        if not rppg_pipeline or not rppg_pipeline.is_initialized:
            return {
                "error": "Pipeline not ready",
                "heart_rate": None,
                "signal_quality": 0.0
            }
        
        # Decode all frames
        frames_list: List[np.ndarray] = []
        for i, frame_str in enumerate(request.frames):
            try:
                # Remove data URL prefix if present
                if "," in frame_str:
                    frame_str = frame_str.split(",", 1)[1]
                
                frame_data = base64.b64decode(frame_str)
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frames_list.append(frame)
            except Exception as e:
                print(f"Error decoding frame {i}: {e}")
                continue
        
        if len(frames_list) == 0:
            return {
                "error": "No valid frames decoded",
                "heart_rate": None,
                "signal_quality": 0.0
            }
        
        print(f"📊 Processing {len(frames_list)} frames via batch endpoint...")
        
        # Convert to numpy array
        frames_array = np.array(frames_list)
        print(f"🔍 Frame array shape: {frames_array.shape}, dtype: {frames_array.dtype}")
        
        # Process in background thread
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            process_frames_batch,
            frames_array
        )
        
        if result:
            print(f"✓ HR: {result.get('heart_rate', 'N/A')} BPM, SQI: {result.get('signal_quality', 0):.2f}")
            return result
        else:
            return {
                "error": "Processing failed",
                "heart_rate": None,
                "signal_quality": 0.0
            }
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "heart_rate": None,
            "signal_quality": 0.0
        }


@app.websocket("/ws/rppg")
async def websocket_rppg_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time rPPG processing
    Collects frames in buffer and processes in batches
    """
    await websocket.accept()
    print("📹 Client connected to rPPG WebSocket")
    
    frame_buffer: List[np.ndarray] = []
    batch_frame_count = 0  # Frames in current batch (resets after processing)
    batch_start_time = None  # Track when current batch started
    last_progress_time = None  # Track when we last sent progress update
    frames_since_overlap = 0  # Track NEW frames added (excluding overlap)
    BUFFER_SIZE = 300  # 10 seconds at 30 FPS (minimum for good rPPG signal)
    OVERLAP_SIZE = 30  # Frames kept from previous batch
    EXPECTED_DURATION_SECONDS = 30  # 300 frames at 10 FPS from frontend
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_json()
            
            if "frame" not in data:
                continue
            
            # Decode base64 frame
            try:
                frame_str = data["frame"]
                if "," in frame_str:
                    frame_str = frame_str.split(",", 1)[1]
                
                frame_data = base64.b64decode(frame_str)
                if len(frame_data) == 0:
                    continue
                
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
            except Exception as e:
                print(f"Error decoding frame: {e}")
                continue
            
            # Initialize batch timer on first frame of new batch
            if batch_start_time is None:
                batch_start_time = time.time()
                last_progress_time = batch_start_time
                frames_since_overlap = 0  # Reset counter for new batch
            
            batch_frame_count += 1
            frames_since_overlap += 1
            frame_buffer.append(frame)
            
            if batch_frame_count == 1:
                print(f"✓ First frame received: {frame.shape}")
            
            # Process when buffer is full
            if len(frame_buffer) >= BUFFER_SIZE:
                print(f"📊 Processing {len(frame_buffer)} frames...")
                
                try:
                    # Convert to numpy array (frames already in BGR from cv2.imdecode)
                    frames_array = np.array(frame_buffer)
                    
                    print(f"🔍 Frame array shape: {frames_array.shape}, dtype: {frames_array.dtype}")
                    
                    # Process in background thread
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        executor,
                        process_frames_batch,
                        frames_array
                    )
                    
                    if result:
                        print(f"✓ HR: {result.get('heart_rate', 'N/A')} BPM, SQI: {result.get('signal_quality', 0):.2f}")
                        await websocket.send_json(result)
                    else:
                        await websocket.send_json({
                            "signal_quality": 0.0,
                            "message": "Processing...",
                            "frame_count": batch_frame_count
                        })
                    
                    # Keep last 30 frames for overlap and reset counters
                    frame_buffer = frame_buffer[-OVERLAP_SIZE:]
                    batch_frame_count = OVERLAP_SIZE  # Reset to overlap frame count
                    batch_start_time = None  # Reset timer for next batch
                    last_progress_time = None
                    frames_since_overlap = 0
                    
                except Exception as e:
                    print(f"❌ Error processing buffer: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Send progress update every 1 second (based on elapsed time, not frame count)
            else:
                current_time = time.time()
                if last_progress_time and (current_time - last_progress_time) >= 1.0:
                    seconds_elapsed = int(current_time - batch_start_time)
                    # We need BUFFER_SIZE - OVERLAP_SIZE new frames for a complete batch
                    frames_needed = BUFFER_SIZE - OVERLAP_SIZE
                    # Only show progress while collecting (don't go past expected duration)
                    if seconds_elapsed <= EXPECTED_DURATION_SECONDS:
                        await websocket.send_json({
                            "signal_quality": 0.0,
                            "message": f"Collecting data... {seconds_elapsed}/{EXPECTED_DURATION_SECONDS}s",
                            "frame_count": batch_frame_count
                        })
                        last_progress_time = current_time
            
    except WebSocketDisconnect:
        print("📹 Client disconnected from rPPG WebSocket")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()


# Helper Functions for Video Processing

def decode_frames(frame_strings: List[str]) -> Optional[np.ndarray]:
    """Decode base64 frames to numpy array"""
    try:
        frames_list = []
        for i, frame_str in enumerate(frame_strings):
            try:
                # Remove data URL prefix if present
                if "," in frame_str:
                    frame_str = frame_str.split(",", 1)[1]

                frame_data = base64.b64decode(frame_str)
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    frames_list.append(frame)
            except Exception as e:
                print(f"⚠️ Error decoding frame {i}: {e}")
                continue

        if len(frames_list) == 0:
            return None

        return np.array(frames_list)
    except Exception as e:
        print(f"❌ Error in decode_frames: {e}")
        return None


def decode_audio(audio_string: str) -> Optional[np.ndarray]:
    """Decode base64 audio WAV to numpy array"""
    try:
        # Remove data URL prefix if present
        if "," in audio_string:
            audio_string = audio_string.split(",", 1)[1]

        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_string)

        # Read WAV file from bytes
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
            # Get audio parameters
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            print(f"📻 Audio: {channels}ch, {framerate}Hz, {sample_width*8}bit, {n_frames} frames")

            # Read audio data
            audio_data = wav_file.readframes(n_frames)

            # Convert to numpy array
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            else:
                dtype = np.int32

            audio_array = np.frombuffer(audio_data, dtype=dtype)

            # Convert to float32 and normalize to [-1, 1]
            if dtype == np.uint8:
                audio_array = (audio_array.astype(np.float32) - 128) / 128.0
            else:
                audio_array = audio_array.astype(np.float32) / float(np.iinfo(dtype).max)

            # Convert stereo to mono if needed
            if channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1)

            return audio_array

    except Exception as e:
        print(f"❌ Error decoding audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def decode_images(image_strings: List[str]) -> Optional[List[np.ndarray]]:
    """Decode base64 images to list of numpy arrays"""
    try:
        images = []
        for i, img_str in enumerate(image_strings):
            try:
                # Remove data URL prefix if present
                if "," in img_str:
                    img_str = img_str.split(",", 1)[1]

                img_data = base64.b64decode(img_str)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    images.append(img)
            except Exception as e:
                print(f"⚠️ Error decoding image {i}: {e}")
                continue

        if len(images) == 0:
            return None

        return images
    except Exception as e:
        print(f"❌ Error in decode_images: {e}")
        return None


@app.post("/api/process-video")
async def process_video_endpoint(request: VideoProcessRequest):
    """
    Process uploaded video with all pipelines (rPPG, Cry, HeAR, VGA)
    Accepts video frames, audio, and screenshots extracted by frontend
    """
    start_time = time.time()

    try:
        print("\n" + "="*60)
        print("📹 New video processing request received")
        print(f"   Frames: {len(request.video_frames)}")
        print(f"   Screenshots: {len(request.screenshots)}")
        print(f"   Metadata: {request.metadata}")
        print("="*60)

        # Decode all inputs
        print("\n🔄 Decoding inputs...")
        frames_array = decode_frames(request.video_frames)
        audio_array = decode_audio(request.audio_data)
        screenshots_list = decode_images(request.screenshots)

        if frames_array is None:
            return {"error": "Failed to decode video frames"}
        if audio_array is None:
            return {"error": "Failed to decode audio"}
        if screenshots_list is None:
            return {"error": "Failed to decode screenshots"}

        print(f"✓ Decoded: {frames_array.shape} frames, {audio_array.shape} audio, {len(screenshots_list)} screenshots")

        # Process all pipelines in parallel
        print("\n⚡ Processing with all pipelines in parallel...")
        loop = asyncio.get_running_loop()

        # Run all pipelines concurrently
        results = await asyncio.gather(
            # rPPG: Video frames
            loop.run_in_executor(executor, process_rppg, frames_array),
            # Cry: Audio
            loop.run_in_executor(executor, process_cry, audio_array),
            # HeAR: Audio
            loop.run_in_executor(executor, process_hear, audio_array),
            # VGA: Screenshots
            loop.run_in_executor(executor, process_vga, screenshots_list),
            return_exceptions=True  # Don't fail if one pipeline errors
        )

        rppg_result, cry_result, hear_result, vga_result = results

        # Generate AI-powered triage report using FusionEngine
        triage_report = None
        if fusion_engine and fusion_engine.is_initialized:
            print("\n🧠 Generating AI triage report with FusionEngine...")
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
                print(f"✓ Triage report generated: {triage_report.priority_level}")
            except Exception as e:
                print(f"⚠️ FusionEngine error: {e}")
                import traceback
                traceback.print_exc()

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

        # Add triage report if available
        if triage_report:
            response["triage"] = {
                "priority_level": triage_report.priority_level,
                "critical_alerts": triage_report.critical_alerts,
                "recommendations": triage_report.recommendations,
                "medical_interpretation": triage_report.medical_interpretation,
                "confidence_score": triage_report.confidence_score,
                "timestamp": triage_report.timestamp.isoformat()
            }

        print(f"\n✅ Processing complete in {processing_time_ms:.0f}ms")
        print("="*60 + "\n")

        return response

    except Exception as e:
        print(f"\n❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
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
            print("⚠️ rPPG pipeline not initialized")
            return None

        print(f"📊 Processing rPPG with frames shape: {frames.shape}, dtype: {frames.dtype}")
        result = rppg_pipeline.process(frames)
        print(f"📊 rPPG result type: {type(result)}")
        if hasattr(result, 'data'):
            print(f"📊 rPPG result.data keys: {result.data.keys() if result.data else 'empty'}")
            print(f"📊 rPPG result.data: {result.data}")
        return result
    except Exception as e:
        print(f"❌ rPPG error: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_cry(audio: np.ndarray) -> Optional[Dict[str, Any]]:
    """Process audio with Cry pipeline"""
    global cry_pipeline
    try:
        if not cry_pipeline or not cry_pipeline.is_initialized:
            return None

        result = cry_pipeline.process(audio)
        return result
    except Exception as e:
        print(f"❌ Cry error: {e}")
        return None


def process_hear(audio: np.ndarray) -> Optional[Dict[str, Any]]:
    """Process audio with HeAR pipeline"""
    global hear_pipeline
    try:
        if not hear_pipeline or not hear_pipeline.is_initialized:
            return None

        result = hear_pipeline.process(audio)
        return result
    except Exception as e:
        print(f"❌ HeAR error: {e}")
        return None


def process_vga(screenshots: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Process screenshots with VGA pipeline"""
    global vga_pipeline
    try:
        if not vga_pipeline or not vga_pipeline.is_initialized:
            return None

        result = vga_pipeline.process_batch(screenshots)
        return result
    except Exception as e:
        print(f"❌ VGA error: {e}")
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
        # It's a PipelineResult object - convert numpy arrays to lists
        return convert_numpy_to_list(result.data)
    elif isinstance(result, dict):
        # Already a dictionary - still need to convert numpy arrays
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
            print("❌ Pipeline not ready")
            return None
        
        # Process frames with pre-initialized pipeline
        result = rppg_pipeline.process(frames)
        
        print(f"📋 Pipeline result: {result}")
        print(f"📋 Result data: {result.data if result else 'None'}")
        
        if result and result.data:
            response = {
                "heart_rate": result.data.get('heart_rate'),
                "signal_quality": result.data.get('signal_quality', 0.0),
                "respiratory_rate": result.data.get('respiratory_rate'),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✓ Built response: HR={response.get('heart_rate')}, SQI={response.get('signal_quality')}")
            
            # Add HRV if available
            hrv = result.data.get('heart_rate_variability', {})
            if hrv:
                response["hrv"] = {
                    "sdnn": hrv.get('sdnn', 0),
                    "rmssd": hrv.get('rmssd', 0),
                    "lf_hf": hrv.get('LF/HF', 0)
                }
            
            return response
        
        print("⚠️ No result data returned")
        return None
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🩺 AI GUARDIAN API SERVER")
    print("="*60)
    print("\nStarting on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws/rppg")
    print("Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
