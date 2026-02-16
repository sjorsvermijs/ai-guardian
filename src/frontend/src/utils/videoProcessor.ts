/**
 * Video Processing Utilities
 *
 * Extracts frames, audio, and screenshots from video files for AI Guardian pipelines.
 * - Video frames (300 @ 30fps) for rPPG pipeline
 * - Audio (16kHz mono) for Cry and HeAR pipelines
 * - Screenshots (10 evenly distributed) for VGA pipeline
 */

const TARGET_FPS = 30;
const TARGET_DURATION = 10; // seconds
const SCREENSHOT_COUNT = 10;
const AUDIO_SAMPLE_RATE = 16000;

/**
 * Extract first 10 seconds of video as a Blob
 */
export async function extractFirst10Seconds(file: File): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';

    video.onloadedmetadata = () => {
      // If video is already <= 10s, return as-is
      if (video.duration <= TARGET_DURATION) {
        resolve(file);
        URL.revokeObjectURL(video.src);
        return;
      }

      // Otherwise, we'll just work with the first 10s during frame extraction
      // For now, return the original file and handle trimming during extraction
      resolve(file);
      URL.revokeObjectURL(video.src);
    };

    video.onerror = () => {
      reject(new Error('Failed to load video metadata'));
      URL.revokeObjectURL(video.src);
    };

    video.src = URL.createObjectURL(file);
  });
}

/**
 * Extract 300 frames at 30 FPS from video (first 10 seconds)
 * Returns base64-encoded JPEG images
 */
export async function extractFrames(
  videoBlob: Blob,
  fps: number = TARGET_FPS,
  duration: number = TARGET_DURATION
): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      reject(new Error('Failed to get canvas context'));
      return;
    }

    const frames: string[] = [];
    let currentFrame = 0;
    const totalFrames = fps * duration;
    const frameInterval = 1 / fps; // seconds between frames

    video.preload = 'auto';
    video.muted = true;

    video.onloadedmetadata = async () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const actualDuration = Math.min(video.duration, duration);

      try {
        for (let i = 0; i < totalFrames && currentFrame / fps < actualDuration; i++) {
          const timestamp = i * frameInterval;

          // Seek to timestamp
          video.currentTime = timestamp;

          // Wait for seek to complete
          await new Promise<void>((seekResolve) => {
            const seekHandler = () => {
              video.removeEventListener('seeked', seekHandler);
              seekResolve();
            };
            video.addEventListener('seeked', seekHandler);
          });

          // Draw frame to canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Convert to base64 JPEG (quality 0.85 for good balance)
          const frameData = canvas.toDataURL('image/jpeg', 0.85);
          frames.push(frameData);

          currentFrame++;
        }

        URL.revokeObjectURL(video.src);
        resolve(frames);
      } catch (error) {
        URL.revokeObjectURL(video.src);
        reject(error);
      }
    };

    video.onerror = () => {
      URL.revokeObjectURL(video.src);
      reject(new Error('Failed to load video for frame extraction'));
    };

    video.src = URL.createObjectURL(videoBlob);
  });
}

/**
 * Extract 10 evenly-distributed screenshots from video
 * Returns base64-encoded JPEG images
 */
export async function extractScreenshots(
  videoBlob: Blob,
  count: number = SCREENSHOT_COUNT
): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      reject(new Error('Failed to get canvas context'));
      return;
    }

    const screenshots: string[] = [];

    video.preload = 'auto';
    video.muted = true;

    video.onloadedmetadata = async () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const actualDuration = Math.min(video.duration, TARGET_DURATION);
      const interval = actualDuration / count;

      try {
        for (let i = 0; i < count; i++) {
          // Take screenshots at evenly distributed timestamps
          const timestamp = i * interval + interval / 2; // Center of each segment

          video.currentTime = Math.min(timestamp, actualDuration - 0.1);

          // Wait for seek to complete
          await new Promise<void>((seekResolve) => {
            const seekHandler = () => {
              video.removeEventListener('seeked', seekHandler);
              seekResolve();
            };
            video.addEventListener('seeked', seekHandler);
          });

          // Draw screenshot to canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Convert to base64 JPEG
          const screenshotData = canvas.toDataURL('image/jpeg', 0.9);
          screenshots.push(screenshotData);
        }

        URL.revokeObjectURL(video.src);
        resolve(screenshots);
      } catch (error) {
        URL.revokeObjectURL(video.src);
        reject(error);
      }
    };

    video.onerror = () => {
      URL.revokeObjectURL(video.src);
      reject(new Error('Failed to load video for screenshot extraction'));
    };

    video.src = URL.createObjectURL(videoBlob);
  });
}

/**
 * Convert AudioBuffer to WAV format
 */
function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const length = buffer.length * buffer.numberOfChannels * 2 + 44;
  const arrayBuffer = new ArrayBuffer(length);
  const view = new DataView(arrayBuffer);
  const channels = [];
  let offset = 0;
  let pos = 0;

  // Write WAV header
  const setUint16 = (data: number) => {
    view.setUint16(pos, data, true);
    pos += 2;
  };

  const setUint32 = (data: number) => {
    view.setUint32(pos, data, true);
    pos += 4;
  };

  // RIFF identifier
  setUint32(0x46464952);
  // File length
  setUint32(length - 8);
  // RIFF type
  setUint32(0x45564157);
  // Format chunk identifier
  setUint32(0x20746d66);
  // Format chunk length
  setUint32(16);
  // Sample format (PCM)
  setUint16(1);
  // Channel count
  setUint16(buffer.numberOfChannels);
  // Sample rate
  setUint32(buffer.sampleRate);
  // Byte rate
  setUint32(buffer.sampleRate * 2 * buffer.numberOfChannels);
  // Block align
  setUint16(buffer.numberOfChannels * 2);
  // Bits per sample
  setUint16(16);
  // Data chunk identifier
  setUint32(0x61746164);
  // Data chunk length
  setUint32(length - pos - 4);

  // Write interleaved PCM data
  for (let i = 0; i < buffer.numberOfChannels; i++) {
    channels.push(buffer.getChannelData(i));
  }

  while (pos < length) {
    for (let i = 0; i < buffer.numberOfChannels; i++) {
      const sample = Math.max(-1, Math.min(1, channels[i][offset]));
      view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      pos += 2;
    }
    offset++;
  }

  return arrayBuffer;
}

/**
 * Convert ArrayBuffer to base64 string
 */
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Extract and resample audio from video to 16kHz mono
 * Returns base64-encoded WAV file
 */
export async function extractAndResampleAudio(videoBlob: Blob): Promise<string> {
  try {
    // Convert video blob to array buffer
    const arrayBuffer = await videoBlob.arrayBuffer();

    // Create audio context for decoding
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

    // Decode audio from video
    let audioBuffer: AudioBuffer;
    try {
      audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    } catch (error) {
      throw new Error('No audio track found in video or unsupported audio format');
    }

    // Calculate target duration (first 10 seconds or less)
    const targetDuration = Math.min(audioBuffer.duration, TARGET_DURATION);
    const targetLength = Math.floor(targetDuration * AUDIO_SAMPLE_RATE);

    // Create offline context for resampling to 16kHz mono
    const offlineContext = new OfflineAudioContext(1, targetLength, AUDIO_SAMPLE_RATE);

    // Create buffer source
    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;

    // Connect to destination
    source.connect(offlineContext.destination);

    // Start and render
    source.start(0, 0, targetDuration);
    const resampledBuffer = await offlineContext.startRendering();

    // Convert to WAV format
    const wavBuffer = audioBufferToWav(resampledBuffer);

    // Convert to base64
    const base64Audio = arrayBufferToBase64(wavBuffer);

    // Close audio context
    await audioContext.close();

    return `data:audio/wav;base64,${base64Audio}`;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to extract audio from video');
  }
}

/**
 * Process entire video: extract frames, screenshots, and audio
 * Returns all data needed for backend API
 */
export interface ProcessedVideoData {
  video_frames: string[];
  audio_data: string;
  screenshots: string[];
  metadata: {
    fps: number;
    duration: number;
    frame_count: number;
    screenshot_count: number;
    original_filename: string;
  };
}

export async function processVideo(
  file: File,
  onProgress?: (stage: string, percent: number) => void
): Promise<ProcessedVideoData> {
  try {
    // Stage 1: Extract first 10 seconds
    onProgress?.('Preparing video', 0);
    const videoBlob = await extractFirst10Seconds(file);

    // Stage 2: Extract frames (slowest operation)
    onProgress?.('Extracting frames', 10);
    const frames = await extractFrames(videoBlob);

    // Stage 3: Extract screenshots
    onProgress?.('Extracting screenshots', 60);
    const screenshots = await extractScreenshots(videoBlob);

    // Stage 4: Extract audio
    onProgress?.('Extracting audio', 80);
    const audio = await extractAndResampleAudio(videoBlob);

    onProgress?.('Complete', 100);

    return {
      video_frames: frames,
      audio_data: audio,
      screenshots: screenshots,
      metadata: {
        fps: TARGET_FPS,
        duration: TARGET_DURATION,
        frame_count: frames.length,
        screenshot_count: screenshots.length,
        original_filename: file.name
      }
    };
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to process video');
  }
}
