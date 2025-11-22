import cv2
import subprocess
import numpy as np
import time

def capture_with_ffmpeg(device_index=2, width=640, height=480, fps=30):
    """
    Uses FFmpeg to capture video frames, bypassing cv2.VideoCapture.
    This is a generator that yields frames.
    """
    # Command to run ffmpeg
    # It will output raw video frames in BGR format to the stdout pipe
    command = [
        'ffmpeg',
        '-f', 'v4l2',
        '-input_format', 'mjpeg',
        '-framerate', str(fps),
        '-video_size', f'{width}x{height}',
        '-i', f'/dev/video{device_index}',
        '-f', 'rawvideo',      # Output format
        '-pix_fmt', 'bgr24',     # Output pixel format (what OpenCV uses)
        'pipe:1'                # Output to stdout
    ]

    # Start the ffmpeg process
    # stderr is redirected to DEVNULL to hide ffmpeg's status messages
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    # Calculate the size of one frame in bytes
    frame_size = width * height * 3
    
    try:
        while True:
            # Read exactly one frame from the pipe
            raw_frame = proc.stdout.read(frame_size)
            
            if len(raw_frame) != frame_size:
                print("Error: Did not read a full frame. Exiting.")
                break
            
            # Convert the raw bytes to a NumPy array
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            
            yield frame
            
    finally:
        # Ensure the process is terminated when the generator is closed
        proc.kill()


if __name__ == "__main__":
    print("Starting capture with FFmpeg... Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    # Use the generator
    for frame in capture_with_ffmpeg(device_index=6, width=640, height=480, fps=30):
        
        frame_count += 1
        cv2.imshow("FFmpeg Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    cv2.destroyAllWindows()

    elapsed = end_time - start_time
    if elapsed > 0:
        measured_fps = frame_count / elapsed
        print("\n================== RESULT ==================")
        print(f"Captured {frame_count} frames in {elapsed:.2f} seconds.")
        print(f"Measured FPS: {measured_fps:.2f}")
        print("==========================================")