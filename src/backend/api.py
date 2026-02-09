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
from src.core.config import config

app = FastAPI(title="AI Guardian API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global thread pool executor and rPPG pipeline
executor: Optional[ThreadPoolExecutor] = None
rppg_pipeline: Optional[RPPGPipeline] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global executor, rppg_pipeline
    print("🚀 Starting AI Guardian API...")
    
    # Create thread pool executor
    executor = ThreadPoolExecutor(max_workers=2)
    print("✓ Thread pool ready")
    
    # Initialize rPPG pipeline (this takes a few seconds)
    print("📊 Initializing rPPG pipeline...")
    try:
        rppg_pipeline = RPPGPipeline(config.rppg_config)
        rppg_pipeline.initialize()
        print(f"✓ rPPG Pipeline ready (model: {config.rppg_config['model_name']})")
    except Exception as e:
        print(f"❌ Failed to initialize rPPG pipeline: {e}")
        rppg_pipeline = None


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
