#!/usr/bin/env python3
"""
Live Webcam rPPG Monitor with Visual Preview
Shows real-time video feed with face detection and heart rate overlay
"""
import os
import sys
import warnings

# Set environment variable for macOS camera permissions
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')
warnings.filterwarnings('ignore', message='Class AVF')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import threading
from src.pipelines.rppg.pipeline import RPPGPipeline
from src.core.config import config
from src.core.fusion_engine import FusionEngine
from src.core.base_pipeline import PipelineResult
from datetime import datetime

# Global flag for quit signal
quit_flag = threading.Event()

def main():
    """Main live monitoring function."""
    print("=" * 80)
    print("🎥 AI GUARDIAN - LIVE WEBCAM rPPG MONITOR")
    print("=" * 80)
    print()
    print("This tool provides real-time vital signs monitoring with video preview.")
    print()
    print("How it works:")
    print("  1. Monitors signal quality in real-time")
    print("  2. Once SQI > 50%, captures 15 seconds of good data")
    print("  3. Automatically stops and shows results")
    print()
    print("Controls:")
    print("  • Press 'q' or ESC to quit early")
    print("  • Press 's' to take a snapshot measurement")
    print()
    print("Tips:")
    print("  • Ensure good lighting")
    print("  • Keep your face visible and steady")
    print()
    
    camera_index = int(input("Enter camera index (0 for default): ") or "0")
    capture_duration = float(input("Data collection duration after good signal (default 15s, min 10s): ") or "15")
    
    # Ensure minimum duration
    if capture_duration < 10:
        print(f"⚠️ Duration too short ({capture_duration}s), using 10s minimum")
        capture_duration = 10
    
    use_medgemma = input("Use MedGemma AI for clinical interpretation? (y/n, default=y): ").strip().lower() != 'n'
    
    print("\nInitializing pipeline...")
    pipeline = RPPGPipeline(config.rppg_config)
    pipeline.initialize()
    print(f"✓ rPPG ready (model: {config.rppg_config['model_name']})")
    
    # Initialize fusion engine with MedGemma if requested
    fusion = None
    if use_medgemma:
        print("\nInitializing MedGemma AI...")
        fusion = FusionEngine(config={'use_medgemma': True})
    
    print("\nStarting live preview...")
    print("Waiting for good signal quality (SQI > 50%)...\n")
    
    quit_flag.clear()  # Reset quit flag
    
    try:
        # Use open-rppg's video_capture for real-time processing
        pipeline.model.video_capture(camera_index).__enter__()
        
        last_hr_time = 0
        current_hr = None
        current_sqi = None
        current_rr = None
        should_quit = False
        frame_count = 0
        
        # Auto-stop state
        good_signal_start = None
        collecting_data = False
        
        # Iterate through preview frames
        for frame, box in pipeline.model.preview:
            if quit_flag.is_set() or should_quit:
                break
            
            frame_count += 1
                
            # Convert RGB to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Calculate HR every 2 seconds to avoid lag
            now = time.time()
            if now - last_hr_time > 2.0:
                result = pipeline.model.hr(start=-30)  # Use last 30 seconds
                if result and result['hr']:
                    current_hr = result['hr']
                    current_sqi = result['SQI']
                    hrv = result.get('hrv', {})
                    if 'breathingrate' in hrv:
                        current_rr = hrv['breathingrate'] * 60  # Convert to breaths/min
                    else:
                        current_rr = None
                    
                    # Check if we should start collecting good data
                    if current_sqi and current_sqi > 0.5:
                        if not collecting_data:
                            # Start the collection timer
                            good_signal_start = now
                            collecting_data = True
                            print(f"\n✓ Good signal detected (SQI: {current_sqi:.1%})")
                            print(f"  Collecting {capture_duration}s of data...")
                        else:
                            # Check if we've collected enough data
                            elapsed = now - good_signal_start
                            remaining = capture_duration - elapsed
                            if elapsed >= capture_duration:
                                print(f"\n✓ Data collection complete!")
                                should_quit = True
                                quit_flag.set()
                                break
                            elif int(elapsed) != int(elapsed - 2.0):  # Print every 2 seconds
                                print(f"  Collecting... {remaining:.1f}s remaining")
                    
                last_hr_time = now
            
            # Draw face detection box
            if box is not None:
                y1, y2 = box[0]
                x1, x2 = box[1]
                
                # Draw green rectangle around detected face
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display metrics on frame
                y_offset = y1 - 10
                
                if current_hr is not None:
                    # Heart Rate
                    hr_color = (0, 255, 0) if 60 <= current_hr <= 100 else (0, 165, 255)
                    cv2.putText(display_frame, f"HR: {current_hr:.1f} BPM", 
                               (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, hr_color, 2)
                    y_offset -= 30
                    
                    # Signal Quality
                    if current_sqi is not None:
                        sqi_color = (0, 255, 0) if current_sqi > 0.5 else (0, 165, 255)
                        cv2.putText(display_frame, f"SQI: {current_sqi:.1%}", 
                                   (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, sqi_color, 2)
                        y_offset -= 25
                    
                    # Respiratory Rate
                    if current_rr is not None:
                        rr_color = (0, 255, 0) if 12 <= current_rr <= 20 else (0, 165, 255)
                        cv2.putText(display_frame, f"RR: {current_rr:.1f} br/min", 
                                   (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, rr_color, 2)
            else:
                # No face detected warning
                cv2.putText(display_frame, "No face detected", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 2)
            
            # Add status bar at bottom
            if collecting_data and good_signal_start:
                elapsed = time.time() - good_signal_start
                remaining = max(0, capture_duration - elapsed)
                status_text = f"Collecting data: {remaining:.1f}s remaining | SQI: {current_sqi:.1%}" if current_sqi else f"Collecting: {remaining:.1f}s"
            elif current_sqi and current_sqi > 0.3:
                status_text = f"SQI: {current_sqi:.1%} - Waiting for > 50%..."
            else:
                status_text = f"Detecting face and signal..."
            
            cv2.putText(display_frame, status_text, 
                       (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("AI Guardian - rPPG Monitor", display_frame)
            
            # Handle keyboard input - use very short wait for responsiveness
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                should_quit = True
                quit_flag.set()
                print("\nStopping...")
                break  # Immediate break
            elif key == ord('s'):
                print("\n📸 Snapshot measurement:")
                if current_hr:
                    print(f"  HR:  {current_hr:.1f} BPM")
                    print(f"  SQI: {current_sqi:.1%}")
                    if current_rr:
                        print(f"  RR:  {current_rr:.1f} breaths/min")
                else:
                    print("  No data available yet, wait longer...")
        
        print(f"\nProcessed {frame_count} frames")
        
        # Get final comprehensive results if we collected data
        if collecting_data:
            print("\nAnalyzing collected data...")
            
            # Get results from the full collection window
            collection_time = capture_duration if good_signal_start else 30
            
            # Try different time windows if the requested one fails
            for time_window in [collection_time, 10, 15, 20, 30]:
                try:
                    final_result = pipeline.model.hr(start=-time_window, end=None, return_hrv=True)
                    if final_result and final_result.get('hr'):
                        if time_window != collection_time:
                            print(f"Note: Using {time_window}s window (requested {collection_time}s was too short)")
                        break
                except:
                    continue
            else:
                final_result = None
            
            if final_result and final_result.get('hr'):
                hr = final_result['hr']
                sqi = final_result['SQI']
                hrv = final_result.get('hrv', {})
                rr = hrv['breathingrate'] * 60 if hrv and 'breathingrate' in hrv else None
                
                # Display basic vital signs
                print("\n" + "=" * 80)
                print("💓 FINAL RESULTS")
                print("=" * 80)
                print(f"\nAnalysis period: {collection_time:.1f} seconds")
                print(f"\n📊 Vital Signs:")
                
                hr_status = "✅" if 60 <= hr <= 100 else "⚠️"
                print(f"  {hr_status} Heart Rate:       {hr:.1f} BPM")
                
                if rr:
                    rr_status = "✅" if 12 <= rr <= 20 else "⚠️"
                    print(f"  {rr_status} Respiratory Rate: {rr:.1f} breaths/min")
                
                sqi_status = "✅" if sqi > 0.7 else ("⚠️" if sqi > 0.5 else "❌")
                print(f"  {sqi_status} Signal Quality:   {sqi:.1%}")
                
                # Show HRV if available
                if hrv and 'sdnn' in hrv:
                    print(f"\n💓 Heart Rate Variability:")
                    print(f"  SDNN:      {hrv.get('sdnn', 0):.1f} ms")
                    print(f"  RMSSD:     {hrv.get('rmssd', 0):.1f} ms")
                    if 'LF/HF' in hrv:
                        lf_hf = hrv['LF/HF']
                        print(f"  LF/HF:     {lf_hf:.2f}")
                        if lf_hf > 2.5:
                            print(f"  Status:    Sympathetic dominant (stressed)")
                        elif lf_hf < 1.5:
                            print(f"  Status:    Parasympathetic dominant (relaxed)")
                        else:
                            print(f"  Status:    Balanced")
                
                print("\n" + "=" * 80)
                
                # Generate MedGemma clinical interpretation if enabled
                if fusion and fusion.use_medgemma:
                    print("\n🤖 Generating clinical interpretation with MedGemma AI...")
                    print("(This may take a few seconds...)\n")
                    
                    # Create PipelineResult for fusion engine
                    vital_signs_data = {
                        'heart_rate': hr,
                        'signal_quality': sqi
                    }
                    if rr:
                        vital_signs_data['respiratory_rate'] = rr
                    if hrv:
                        vital_signs_data['hrv'] = hrv
                    
                    rppg_result = PipelineResult(
                        pipeline_name="rPPG",
                        timestamp=datetime.now(),
                        confidence=sqi,
                        data=vital_signs_data,
                        warnings=[],
                        errors=[],
                        metadata={'model': config.rppg_config['model_name'], 'duration': collection_time}
                    )
                    
                    # Generate triage report with MedGemma interpretation
                    triage_report = fusion.fuse(rppg_result=rppg_result)
                    
                    print("\n" + "=" * 80)
                    print("🩺 CLINICAL INTERPRETATION (MedGemma AI)")
                    print("=" * 80)
                    print(f"\n🚦 Triage Priority: {triage_report.priority.value}")
                    print(f"📊 Confidence: {triage_report.confidence:.1%}")
                    
                    if triage_report.critical_alerts:
                        print(f"\n🚨 Critical Alerts:")
                        for alert in triage_report.critical_alerts:
                            print(f"  • {alert}")
                    
                    if triage_report.recommendations:
                        print(f"\n💡 Recommendations:")
                        for rec in triage_report.recommendations:
                            print(f"  • {rec}")
                    
                    # Extract just the MedGemma interpretation from the summary
                    if "--- MedGemma Clinical Interpretation ---" in triage_report.summary:
                        medgemma_part = triage_report.summary.split("--- MedGemma Clinical Interpretation ---")[1]
                        print(f"\n{medgemma_part}")
                    else:
                        print(f"\n{triage_report.summary}")
                    
                    print("\n" + "=" * 80)
                
            else:
                print("\n⚠️ Unable to generate final results")
                print("Note: Collection window may be too short. Try 10-15 seconds minimum.")
                print("      Or the signal quality dropped during collection.")
        
        # Proper cleanup - avoid threading errors
        import sys
        from io import StringIO
        
        # Suppress stderr during cleanup to hide threading errors
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        
        try:
            pipeline.model.alive = False  # Signal thread to stop
            pipeline.model.stop()
        except RuntimeError as e:
            # Suppress "cannot join current thread" error from open-rppg
            if "cannot join current thread" not in str(e):
                sys.stderr = old_stderr
                raise
        except:
            pass
        finally:
            sys.stderr = old_stderr
        
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except RuntimeError as e:
        # Suppress "cannot join current thread" from open-rppg cleanup
        if "cannot join current thread" not in str(e):
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import sys
        from io import StringIO
        
        # Suppress stderr during cleanup
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        
        try:
            pipeline.model.stop()
        except RuntimeError as e:
            # Suppress "cannot join current thread" from open-rppg cleanup
            if "cannot join current thread" not in str(e):
                sys.stderr = old_stderr
                raise
        except:
            pass
        finally:
            sys.stderr = old_stderr
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        pipeline.model = None
        pipeline.is_initialized = False
    
    print("\nMonitoring stopped.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        try:
            cv2.destroyAllWindows()
        except:
            pass
        sys.exit(0)
