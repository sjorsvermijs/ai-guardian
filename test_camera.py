#!/usr/bin/env python3
"""
Test camera access and display basic info
"""
import os
import sys
import warnings

# Set environment for macOS
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np

def test_camera(index=0):
    """Test camera access and display info."""
    print(f"Testing camera {index}...")
    
    try:
        cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera {index}")
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Camera {index} opened successfully")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame from camera {index}")
            cap.release()
            return False
        
        print(f"✓ Successfully captured frame: {frame.shape}")
        
        # Display test frame for 2 seconds
        print(f"\nDisplaying test frame for 2 seconds...")
        cv2.imshow(f"Camera {index} Test", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("CAMERA ACCESS TEST")
    print("=" * 60)
    print()
    
    # Test default camera
    if test_camera(0):
        print("\n✅ Default camera (index 0) works!")
    else:
        print("\n❌ Default camera (index 0) failed")
        print("\nTroubleshooting:")
        print("  1. Check if another app is using the camera")
        print("  2. Grant camera permissions:")
        print("     System Preferences → Security & Privacy → Camera")
        print("  3. Try running as: OPENCV_AVFOUNDATION_SKIP_AUTH=1 python test_camera.py")
        return
    
    # Try to test camera 1
    print("\n" + "=" * 60)
    test_input = input("\nTest camera 1 (external)? (y/n): ")
    if test_input.lower() == 'y':
        test_camera(1)
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        cv2.destroyAllWindows()
