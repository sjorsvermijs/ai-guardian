"""
FastAPI Backend for AI Guardian
Provides real-time rPPG vital signs processing from webcam frames
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime
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
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global rPPG pipeline instance
rppg_pipeline: Optional[RPPGPipeline] = None
executor: Optional[ThreadPoolExecutor] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    rppg_ready: bool


class VitalSignsResponse(BaseModel):
    heart_rate: Optional[float] = None
    respiratory_rate: Optional[float] = None
    signal_quality: float
    hrv: Optional[Dict[str, Any]] = None
    timestamp: str
    frame_count: int


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global executor
    print("🚀 Starting AI Guardian API...")
    
    # Create thread pool executor
    executor = ThreadPoolExecutor(max_workers=4)
    print("✓ Thread pool ready")


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
        timestamp=datetime.now().isoformat(),
        rppg_ready=rppg_pipeline is not None and rppg_pipeline.is_initialized
    )


@app.websocket("/ws/rppg")
async def websocket_rppg_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time rPPG processing
    
    Client sends: { "frame": "base64_encoded_image" }
    Server sends: { "heart_rate": 72.0, "signal_quality": 0.85, ... }
    """
    await websocket.accept()
    
    print("📹 Client connected to rPPG WebSocket")
    
    # Initialize rPPG pipeline once
    connection_pipeline = None
    try:
        print("🔧 Initializing rPPG model for this connection...")
        connection_pipeline = RPPGPipeline(config.rppg_config)
        connection_pipeline.initialize()
        print("✓ Connection-specific rPPG model ready")
    except Exception as e:
        print(f"❌ Failed to initialize connection pipeline: {e}")
        await websocket.send_json({
            "error": f"Failed to initialize rPPG: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        await websocket.close()
        return
    
    # Frame buffer for batch processing
    frame_buffer = []
    frame_count = 0
    BUFFER_SIZE = 150  # 5 seconds at 30 FPS
    TARGET_FPS = 30
    
    try:
        while True:
            # Receive frame from client
            print(f"⏳ Waiting for frame {frame_count + 1}...")
            data = await websocket.receive_json()
            print(f"✓ Received data for frame {frame_count + 1}")
            
            if "frame" not in data:
                await websocket.send_json({"error": "No frame data"})
                continue
            
            # Decode base64 frame
            try:
                frame_str = data["frame"]
                # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
                if "," in frame_str:
                    frame_str = frame_str.split(",", 1)[1]
                
                frame_data = base64.b64decode(frame_str)
                
                if len(frame_data) == 0:
                    print("Warning: Empty frame data received")
                    continue
                
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"Warning: Failed to decode frame (data length: {len(frame_data)})")
                    continue
            except Exception as e:
                print(f"Error decoding frame: {e}")
                continue
            
            frame_count += 1
            
            # Debug logging
            if frame_count == 1:
                print(f"✓ First frame received: {frame.shape}")
            elif frame_count % 30 == 0:
                print(f"📊 Processing frame {frame_count}...")
            
            # Feed frame to rPPG model (run in executor to avoid blocking)
            timestamp = frame_count / 30.0  # Assume 30 FPS
            
            try:
                # Run synchronous rPPG operations in dedicated thread pool
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    executor,
                    connection_pipeline.model.update_frame, 
                    frame, 
                    timestamp
                )
                
                # Get current results (every 30 frames ≈ 1 second)
                if frame_count % 30 == 0:
                    print(f"🔍 Getting HR results for frame {frame_count}...")
                    result = await loop.run_in_executor(
                        executor,
                        lambda: rppg_pipeline.model.hr(start=-5, end=None, return_hrv=True)
                    )
                    
                    print(f"Result: {result}")
                    
                    if result and result.get('hr'):
                        response_data = {
                            "heart_rate": result['hr'],
                            "signal_quality": result['SQI'],
                            "timestamp": datetime.now().isoformat(),
                            "frame_count": frame_count
                        }
                        
                        # Add HRV if available
                        hrv = result.get('hrv', {})
                        if hrv and 'breathingrate' in hrv:
                            response_data["respiratory_rate"] = hrv['breathingrate'] * 60
                            response_data["hrv"] = {
                                "sdnn": hrv.get('sdnn', 0),
                                "rmssd": hrv.get('rmssd', 0),
                                "lf_hf": hrv.get('LF/HF', 0)
                            }
                        
                        await websocket.send_json(response_data)
                    else:
                        # Send quality indicator even if no HR yet
                        await websocket.send_json({
                            "signal_quality": 0.0,
                            "message": "Collecting data...",
                            "frame_count": frame_count
                        })
            except Exception as e:
                print(f"❌ Error processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                if frame_count % 30 == 0:
                    await websocket.send_json({
                        "error": str(e),
                        "frame_count": frame_count
                    })
            
    except WebSocketDisconnect:
        print("📹 Client disconnected from rPPG WebSocket")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Cleanup connection-specific pipeline
        if connection_pipeline and connection_pipeline.model:
            try:
                print("🧹 Cleaning up connection pipeline...")
                connection_pipeline.model.stop()
            except Exception as e:
                print(f"Warning: Error stopping connection pipeline: {e}")


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
