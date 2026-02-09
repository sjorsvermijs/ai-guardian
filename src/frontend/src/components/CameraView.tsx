import { useEffect, useRef, useState, useCallback } from 'react';
import './CameraView.css';

interface VitalSigns {
  heart_rate?: number;
  respiratory_rate?: number;
  signal_quality: number;
  hrv?: {
    sdnn: number;
    rmssd: number;
    lf_hf: number;
  };
  error?: string;
}

interface CameraViewProps {
  onError?: (error: string) => void;
}

const API_URL = 'http://localhost:8000/api/process-frames';
const READY_URL = 'http://localhost:8000/ready';
const FRAMES_NEEDED = 300;
const TARGET_FPS = 30;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

export function CameraView({ onError }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [vitalSigns, setVitalSigns] = useState<VitalSigns | null>(null);
  const [backendReady, setBackendReady] = useState(false);
  const [collectionProgress, setCollectionProgress] = useState<string>('');
  const [progressPercent, setProgressPercent] = useState(0);
  const [qualityLevel, setQualityLevel] = useState<'poor' | 'fair' | 'good' | null>(null);
  const [hasStarted, setHasStarted] = useState(false);
  
  const framesRef = useRef<string[]>([]);
  const captureIntervalRef = useRef<number | null>(null);
  const collectionStartRef = useRef<number | null>(null);

  const checkBackendReady = useCallback(() => {
    fetch(READY_URL)
      .then(res => res.json())
      .then(data => {
        if (data.ready) {
          console.log('✓ Backend is ready');
          setBackendReady(true);
        } else {
          console.log('⏳ Backend not ready yet...');
          setTimeout(checkBackendReady, 2000);
        }
      })
      .catch(err => {
        console.error('Failed to check backend status:', err);
        setTimeout(checkBackendReady, 2000);
      });
  }, []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    const frameData = canvas.toDataURL('image/png');
    framesRef.current.push(frameData);
    
    if (collectionStartRef.current) {
      const elapsed = Math.floor((Date.now() - collectionStartRef.current) / 1000);
      const percent = Math.min(100, Math.round((framesRef.current.length / FRAMES_NEEDED) * 100));
      
      setProgressPercent(percent);
      setCollectionProgress(`${elapsed}s`);
      
      if (framesRef.current.length >= FRAMES_NEEDED) {
        console.log(`📊 Collected ${framesRef.current.length} frames, sending to backend...`);
        sendFrameBatch();
        return;
      }
    }
  }, []);

  const sendFrameBatch = useCallback(async () => {
    if (framesRef.current.length === 0) {
      console.error('No frames to send');
      return;
    }
    
    if (captureIntervalRef.current !== null) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    
    setIsProcessing(true);
    setCollectionProgress('Processing...');
    
    try {
      console.log(`📤 Sending ${framesRef.current.length} frames to backend...`);
      
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          frames: framesRef.current
        })
      });
      
      const result = await response.json();
      console.log('✓ Backend response:', result);
      
      if (result.error) {
        console.error('Backend error:', result.error);
        setError(result.error);
      } else if (result.heart_rate !== undefined) {
        const sqi = result.signal_quality || 0;
        let quality: 'poor' | 'fair' | 'good';
        
        if (sqi < 0.5) {
          quality = 'poor';
        } else if (sqi < 0.7) {
          quality = 'fair';
        } else {
          quality = 'good';
        }
        
        console.log('✅ Vitals received:', result, 'Quality:', quality);
        setVitalSigns(result);
        setQualityLevel(quality);
        setCollectionProgress('');
        setProgressPercent(0);
      }
    } catch (err) {
      console.error('Failed to send frames:', err);
      setError(`Failed to send frames: ${err}`);
    } finally {
      setIsProcessing(false);
      framesRef.current = [];
      collectionStartRef.current = null;
    }
  }, []);

  const startCollecting = useCallback(() => {
    if (!backendReady) {
      console.log('Backend not ready, waiting...');
      return;
    }
    
    console.log('📹 Starting frame collection...');
    framesRef.current = [];
    collectionStartRef.current = Date.now();
    setCollectionProgress('Collecting data... 0/10s');
    setQualityLevel(null);
    
    captureIntervalRef.current = window.setInterval(captureFrame, FRAME_INTERVAL_MS);
  }, [backendReady, captureFrame]);

  const handleRetry = useCallback(() => {
    setVitalSigns(null);
    setQualityLevel(null);
    startCollecting();
  }, [startCollecting]);

  const handleContinue = useCallback(() => {
    setVitalSigns(null);
    setQualityLevel(null);
  }, []);

  const handleStartCollection = useCallback(() => {
    console.log('🔴 User pressed start button');
    setHasStarted(true);
    setCollectionProgress('Starting camera...');
    setTimeout(startCollecting, 500);
  }, [startCollecting]);

  useEffect(() => {
    let stream: MediaStream | null = null;
    
    const startCamera = async () => {
      try {
        setCollectionProgress('Initializing AI Guardian...');
        checkBackendReady();
        
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 }
          },
          audio: false
        });
        
        stream = mediaStream;
        
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          setHasPermission(true);
          
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
          };
        }
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error accessing camera';
        setError(errorMessage);
        setIsLoading(false);
        onError?.(errorMessage);
        console.error('Camera access error:', err);
      }
    };
    
    startCamera();
    
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (captureIntervalRef.current !== null) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, [checkBackendReady, onError]);

  useEffect(() => {
    if (backendReady && hasPermission) {
      setIsLoading(false);
    }
  }, [backendReady, hasPermission]);

  return (
    <div className="camera-view">
      {isLoading && (
        <div className="camera-loading">
          <div className="spinner"></div>
          {backendReady ? (
            <>
              <p>Starting camera...</p>
              <p className="camera-hint">Please allow camera access when prompted</p>
            </>
          ) : (
            <>
              <p>Initializing AI Guardian...</p>
              <p className="camera-hint">Loading rPPG model, please wait...</p>
            </>
          )}
        </div>
      )}

      {error && (
        <div className="camera-error">
          <div className="error-icon">📷</div>
          <h3>Error</h3>
          <p>{error}</p>
          <div className="error-help">
            <p><strong>Please:</strong></p>
            <ul>
              <li>Allow camera permissions in your browser</li>
              <li>Make sure no other app is using the camera</li>
              <li>Try refreshing the page</li>
            </ul>
          </div>
        </div>
      )}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`camera-video ${hasPermission ? 'visible' : 'hidden'} ${!hasStarted ? 'inactive' : ''} ${vitalSigns ? 'dimmed' : ''}`}
      />

      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {hasPermission && !hasStarted && (
        <div className="start-overlay">
          <div className="start-content">
            <h2>Ready to Measure?</h2>
            <p>Position your face in the center of the camera</p>
            <button className="btn btn-primary start-button" onClick={handleStartCollection}>
              Start Measurement
            </button>
          </div>
        </div>
      )}

      {hasPermission && (
        <>
          {!vitalSigns && !isProcessing && (
            <div className="camera-overlay">
              <div className="face-guide">
                <div className="face-oval"></div>
              </div>
            </div>
          )}

          <div className={`vitals-panel ${!vitalSigns ? 'collecting' : ''}`}>
            {!vitalSigns ? (
              <div className="collection-phase">
                <div className="status-header">
                  <h3>Collecting Vital Signs</h3>
                  <p className="status-subtitle">Keep your child's face centered in the oval</p>
                </div>
                
                <div className="progress-container">
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${progressPercent}%` }}></div>
                  </div>
                  <div className="progress-text">
                    <span className="progress-time">{collectionProgress}</span>
                    <span className="progress-total">/ 10s</span>
                  </div>
                </div>
                
                {isProcessing && (
                  <div className="processing-status">
                    <div className="spinner-small"></div>
                    <span>Analyzing data...</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="vitals-display-section">
                <div className={`reading-complete ${qualityLevel}`}>
                  {qualityLevel === 'poor' 
                    ? '⚠️ Low Signal Quality' 
                    : qualityLevel === 'fair'
                    ? '◐ Fair Signal Quality'
                    : '✓ Good Reading'}
                </div>
                <div className="vital-item">
                  <span className="vital-label">Signal Quality</span>
                  <span className={`vital-value ${vitalSigns.signal_quality > 0.5 ? 'good' : 'poor'}`}>
                    {(vitalSigns.signal_quality * 100).toFixed(0)}%
                  </span>
                </div>

                {vitalSigns.heart_rate && (
                  <div className="vital-item primary">
                    <span className="vital-label">Heart Rate</span>
                    <span className="vital-value">
                      {vitalSigns.heart_rate.toFixed(0)} <span className="unit">BPM</span>
                    </span>
                  </div>
                )}

                {vitalSigns.respiratory_rate && (
                  <div className="vital-item">
                    <span className="vital-label">Respiratory Rate</span>
                    <span className="vital-value">
                      {vitalSigns.respiratory_rate.toFixed(1)} <span className="unit">br/min</span>
                    </span>
                  </div>
                )}

                {qualityLevel === 'poor' && (
                  <div className="quality-warning">
                    <p className="warning-title">⚠️ Signal Quality Too Low</p>
                    <p className="warning-text">Please ensure:</p>
                    <ul className="tips-list">
                      <li>Face is well-lit (good lighting is essential)</li>
                      <li>Keep your child's face still in the oval</li>
                      <li>Try moving slightly closer to the camera</li>
                    </ul>
                    <button className="btn btn-primary" onClick={handleRetry}>Try Again</button>
                  </div>
                )}

                {qualityLevel === 'fair' && (
                  <div className="quality-decision">
                    <p className="decision-text">Signal quality is acceptable but could be better.</p>
                    <div className="button-group">
                      <button className="btn btn-secondary" onClick={handleRetry}>Take Another Reading</button>
                      <button className="btn btn-primary" onClick={handleContinue}>Continue</button>
                    </div>
                  </div>
                )}

                {qualityLevel === 'good' && (
                  <div className="quality-success">
                    <p className="success-text">✓ Reading is complete and ready to use.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
  