import { useState, useRef, useEffect } from 'react';
import { processVideo } from '../utils/videoProcessor';
import type { ProcessedVideoData } from '../utils/videoProcessor';
import './VideoUploadView.css';

const API_BASE = import.meta.env.VITE_API_URL?.replace('/api/process-video', '') || 'http://localhost:8000';

interface VitalSigns {
  heart_rate?: number;
  respiratory_rate?: number;
  signal_quality: number;
  hrv?: {
    sdnn: number;
    rmssd: number;
    lf_hf: number;
  };
}

interface CryResult {
  embedding?: number[];
  embedding_dim?: number;
  prediction?: string;
  confidence?: number;
  probabilities?: Record<string, number>;
}

interface HeARResult {
  embeddings?: number[][];
  num_chunks?: number;
  embedding_dim?: number;
  binary_classification?: {
    prediction: string;
    confidence: number;
    probabilities: Record<string, number>;
  };
  multiclass_classification?: {
    prediction: string;
    confidence: number;
    probabilities: Record<string, number>;
    note?: string;
  };
}

interface VGAResult {
  status?: string;
  message?: string;
}

interface TriageReport {
  priority_level: string;
  critical_alerts: string[];
  recommendations: string[];
  parent_message: string;
  specialist_message: string;
  confidence_score: number;
  timestamp: string;
}

interface ProcessingResults {
  rppg?: VitalSigns;
  cry?: CryResult;
  hear?: HeARResult;
  vga?: VGAResult;
  triage?: TriageReport;
  processing_time_ms?: number;
  timestamp?: string;
  error?: string;
}

interface VideoUploadViewProps {
  onError?: (error: string) => void;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/process-video';

export function VideoUploadView({ onError }: VideoUploadViewProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoPreviewRef = useRef<HTMLVideoElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [extractionProgress, setExtractionProgress] = useState<string>('');
  const [progressPercent, setProgressPercent] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ProcessingResults | null>(null);

  // Patient context fields
  const [patientAge, setPatientAge] = useState<string>('');
  const [patientSex, setPatientSex] = useState<string>('');
  const [parentNotes, setParentNotes] = useState<string>('');

  // Backend readiness
  const [backendReady, setBackendReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok && !cancelled) setBackendReady(true);
      } catch {
        if (!cancelled) setTimeout(checkBackend, 2000);
      }
    };
    checkBackend();
    return () => { cancelled = true; };
  }, []);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['video/mp4', 'video/quicktime', 'video/webm', 'video/x-msvideo'];
    if (!validTypes.includes(file.type)) {
      setError('Invalid file type. Please upload MP4, MOV, WebM, or AVI files.');
      return;
    }

    // Validate file size (max 500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File too large. Please upload a video smaller than 500MB.');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResults(null);

    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleProcessVideo = async () => {
    if (!selectedFile) return;

    setIsExtracting(true);
    setError(null);
    setExtractionProgress('Starting...');
    setProgressPercent(0);

    try {
      // Extract frames, audio, and screenshots
      const processedData: ProcessedVideoData = await processVideo(
        selectedFile,
        (stage, percent) => {
          setExtractionProgress(stage);
          setProgressPercent(percent);
        }
      );

      setExtractionProgress('Uploading to backend...');
      setProgressPercent(100);
      setIsExtracting(false);
      setIsUploading(true);

      // Send to backend with patient context
      const requestBody = {
        ...processedData,
        patient_age: patientAge ? parseInt(patientAge) : null,
        patient_sex: patientSex || null,
        parent_notes: parentNotes || null
      };

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      const result = await response.json();

      if (result.error) {
        setError(result.error);
        onError?.(result.error);
      } else {
        setResults(result);
        console.log('✅ Processing complete:', result);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to process video';
      setError(errorMessage);
      onError?.(errorMessage);
      console.error('Processing error:', err);
    } finally {
      setIsExtracting(false);
      setIsUploading(false);
      setExtractionProgress('');
      setProgressPercent(0);
    }
  };

  const handleReset = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    setExtractionProgress('');
    setProgressPercent(0);
    setPatientAge('');
    setPatientSex('');
    setParentNotes('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="video-upload-view">
      <div className="upload-container">
        {!selectedFile && !results && (
          <div className="upload-prompt">
            <div className="upload-icon">📹</div>
            <h2>Upload Video for Analysis</h2>
            <p className="upload-subtitle">
              AI Guardian will analyze the first 10 seconds of your video
            </p>
            <div className="upload-info">
              <p><strong>What we analyze:</strong></p>
              <ul>
                <li>❤️ Heart rate and vital signs (rPPG)</li>
                <li>🔊 Audio for cry detection</li>
                <li>🫁 Respiratory sounds (HeAR)</li>
                <li>👁️ Visual assessment (VGA)</li>
              </ul>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,.mp4,.mov,.webm,.avi"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            <button
              className="btn btn-primary upload-button"
              onClick={() => fileInputRef.current?.click()}
            >
              Select Video File
            </button>
            <p className="file-hint">Supported formats: MP4, MOV, WebM, AVI (max 500MB)</p>
          </div>
        )}

        {selectedFile && !results && (
          <div className="video-preview-section">
            <div className="preview-header">
              <h3>Selected Video: {selectedFile.name}</h3>
              <button className="btn btn-secondary" onClick={handleReset}>
                Cancel
              </button>
            </div>

            {previewUrl && (
              <video
                ref={videoPreviewRef}
                src={previewUrl}
                controls
                className="video-preview"
              />
            )}

            {!isExtracting && !isUploading && (
              <div className="action-section">
                <div className="patient-info-section">
                  <h4>Patient Information (Recommended)</h4>
                  <p className="info-subtitle">Help our AI provide better guidance</p>

                  <div className="patient-inputs">
                    <div className="input-group">
                      <label htmlFor="patient-age">Age in months</label>
                      <input
                        id="patient-age"
                        type="number"
                        min="0"
                        max="120"
                        placeholder="e.g., 6"
                        value={patientAge}
                        onChange={(e) => setPatientAge(e.target.value)}
                        className="patient-input"
                      />
                    </div>

                    <div className="input-group">
                      <label htmlFor="patient-sex">Sex</label>
                      <select
                        id="patient-sex"
                        value={patientSex}
                        onChange={(e) => setPatientSex(e.target.value)}
                        className="patient-input"
                      >
                        <option value="">Select...</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                      </select>
                    </div>
                  </div>

                  <div className="input-group">
                    <label htmlFor="parent-notes">Additional Notes</label>
                    <textarea
                      id="parent-notes"
                      placeholder="Any observations or concerns? (e.g., 'coughing for 2 days', 'seems fussy after feeding')"
                      value={parentNotes}
                      onChange={(e) => setParentNotes(e.target.value)}
                      className="patient-input notes-input"
                      rows={3}
                    />
                  </div>
                </div>

                <p className="processing-info">
                  Note: Only the first <strong>10 seconds</strong> will be analyzed
                </p>
                <button
                  className="btn btn-primary process-button"
                  onClick={handleProcessVideo}
                  disabled={isExtracting || isUploading || !backendReady}
                >
                  {backendReady ? 'Process Video' : 'Backend loading models...'}
                </button>
              </div>
            )}

            {(isExtracting || isUploading) && (
              <div className="extraction-progress">
                <div className="progress-header">
                  <h3>{isUploading ? 'Uploading...' : 'Extracting Data...'}</h3>
                  <p className="progress-subtitle">{extractionProgress}</p>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progressPercent}%` }}></div>
                </div>
                <div className="progress-text">
                  <span className="progress-percent">{progressPercent}%</span>
                </div>
                {isExtracting && (
                  <div className="extraction-details">
                    <div className="spinner-small"></div>
                    <span>This may take up to 30 seconds...</span>
                  </div>
                )}
                {isUploading && (
                  <div className="extraction-details">
                    <div className="spinner-small"></div>
                    <span>Sending to AI Guardian</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {results && (
          <div className="results-section">
            <div className="results-header">
              <h2>✅ Analysis Complete</h2>
              <button className="btn btn-secondary" onClick={handleReset}>
                Analyze Another Video
              </button>
            </div>

            {/* AI Triage Report (Main View) */}
            {results.triage && (
              <div className="triage-report-main">
                <div className="triage-header">
                  <h2>AI Medical Assessment</h2>
                  <span className={`priority-badge priority-${results.triage.priority_level.toLowerCase()}`}>
                    {results.triage.priority_level}
                  </span>
                </div>

                {results.triage.critical_alerts && results.triage.critical_alerts.length > 0 && (
                  <div className="critical-alerts">
                    <h3>Important Alerts</h3>
                    <ul>
                      {results.triage.critical_alerts.map((alert, index) => (
                        <li key={index}>{alert}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Parent-friendly message */}
                {results.triage.parent_message && (
                  <div className="parent-message">
                    <h3>For Parents</h3>
                    <div className="message-text">
                      {results.triage.parent_message.split('\n').map((line, index) => (
                        line.trim() ? <p key={index}>{line}</p> : null
                      ))}
                    </div>
                  </div>
                )}

                {results.triage.recommendations && results.triage.recommendations.length > 0 && (
                  <div className="recommendations">
                    <h3>Next Steps</h3>
                    <ul>
                      {results.triage.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Specialist clinical note */}
                {results.triage.specialist_message && (
                  <details className="specialist-details">
                    <summary className="specialist-summary">
                      Clinical Note (For Healthcare Providers)
                    </summary>
                    <div className="specialist-message">
                      {results.triage.specialist_message.split('\n').map((line, index) => (
                        line.trim() ? <p key={index}>{line}</p> : null
                      ))}
                    </div>
                  </details>
                )}

                <div className="confidence-info">
                  <span>AI Confidence: {(results.triage.confidence_score * 100).toFixed(0)}%</span>
                </div>
              </div>
            )}

            {/* Detailed Pipeline Results (Expandable) */}
            <details className="pipeline-details">
              <summary className="details-summary">
                <h3>📊 View Detailed Measurements</h3>
              </summary>

            <div className="results-grid">
              {/* rPPG Results */}
              <div className="result-card rppg-card">
                <h3>💓 Vital Signs (rPPG)</h3>
                {results.rppg ? (
                  <>
                    <div className="vital-item primary">
                      <span className="vital-label">Heart Rate</span>
                      <span className="vital-value">
                        {results.rppg.heart_rate?.toFixed(0) || 'N/A'}{' '}
                        <span className="unit">BPM</span>
                      </span>
                    </div>
                    <div className="vital-item">
                      <span className="vital-label">Respiratory Rate</span>
                      <span className="vital-value">
                        {results.rppg.respiratory_rate?.toFixed(1) || 'N/A'}{' '}
                        <span className="unit">br/min</span>
                      </span>
                    </div>
                    <div className="vital-item">
                      <span className="vital-label">Signal Quality</span>
                      <span className={`vital-value ${results.rppg.signal_quality > 0.5 ? 'good' : 'poor'}`}>
                        {(results.rppg.signal_quality * 100).toFixed(0)}%
                      </span>
                    </div>
                  </>
                ) : (
                  <p className="no-data">No rPPG data available</p>
                )}
              </div>

              {/* Cry Results */}
              <div className="result-card cry-card">
                <h3>👶 Cry Analysis</h3>
                {results.cry ? (
                  <>
                    {results.cry.prediction ? (
                      <>
                        <div className="vital-item primary">
                          <span className="vital-label">Cry Type</span>
                          <span className="vital-value">{results.cry.prediction}</span>
                        </div>
                        <div className="info-item">
                          <span className="info-label">Confidence</span>
                          <span className="info-value success">
                            {((results.cry.confidence || 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                        {results.cry.probabilities && (
                          <div style={{ marginTop: '12px', fontSize: '12px', color: 'rgba(255,255,255,0.7)' }}>
                            <div>All probabilities:</div>
                            {Object.entries(results.cry.probabilities)
                              .sort(([,a], [,b]) => b - a)
                              .map(([label, prob]) => (
                                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                                  <span>{label}:</span>
                                  <span>{(prob * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <>
                        <div className="info-item">
                          <span className="info-label">Embedding Dimension</span>
                          <span className="info-value">{results.cry.embedding_dim || 768}</span>
                        </div>
                        <div className="info-item">
                          <span className="info-label">Status</span>
                          <span className="info-value success">✓ Processed</span>
                        </div>
                      </>
                    )}
                  </>
                ) : (
                  <p className="no-data">No cry data available</p>
                )}
              </div>

              {/* HeAR Results */}
              <div className="result-card hear-card">
                <h3>🫁 Respiratory Audio (HeAR)</h3>
                {results.hear ? (
                  <>
                    {results.hear.binary_classification ? (
                      <>
                        <div className="vital-item primary">
                          <span className="vital-label">Assessment</span>
                          <span className={`vital-value ${results.hear.binary_classification.prediction === 'Normal' ? 'good' : 'poor'}`}>
                            {results.hear.binary_classification.prediction}
                          </span>
                        </div>
                        <div className="info-item">
                          <span className="info-label">Confidence</span>
                          <span className="info-value">
                            {(results.hear.binary_classification.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        {results.hear.multiclass_classification && (
                          <>
                            <div className="info-item">
                              <span className="info-label">Sound Type</span>
                              <span className="info-value">{results.hear.multiclass_classification.prediction}</span>
                            </div>
                            {results.hear.multiclass_classification.note && (
                              <p style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginTop: '8px' }}>
                                {results.hear.multiclass_classification.note}
                              </p>
                            )}
                          </>
                        )}
                        <div className="info-item">
                          <span className="info-label">Audio Chunks</span>
                          <span className="info-value">{results.hear.num_chunks || 0}</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="info-item">
                          <span className="info-label">Audio Chunks</span>
                          <span className="info-value">{results.hear.num_chunks || 0}</span>
                        </div>
                        <div className="info-item">
                          <span className="info-label">Embedding Dimension</span>
                          <span className="info-value">{results.hear.embedding_dim || 512}</span>
                        </div>
                        <div className="info-item">
                          <span className="info-label">Status</span>
                          <span className="info-value success">✓ Processed</span>
                        </div>
                      </>
                    )}
                  </>
                ) : (
                  <p className="no-data">No HeAR data available</p>
                )}
              </div>

              {/* VGA Results */}
              <div className="result-card vga-card">
                <h3>👁️ Visual Assessment (VGA)</h3>
                {results.vga ? (
                  <>
                    <div className="info-item">
                      <span className="info-label">Status</span>
                      <span className="info-value warning">{results.vga.status || 'Processing'}</span>
                    </div>
                    {results.vga.message && (
                      <p className="vga-message">{results.vga.message}</p>
                    )}
                  </>
                ) : (
                  <p className="no-data">No VGA data available</p>
                )}
              </div>
            </div>
            </details>

            {results.processing_time_ms && (
              <div className="processing-stats">
                <span>⏱️ Processing time: {(results.processing_time_ms / 1000).toFixed(2)}s</span>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="error-display">
            <div className="error-icon">⚠️</div>
            <h3>Error</h3>
            <p>{error}</p>
            <button className="btn btn-primary" onClick={handleReset}>
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
