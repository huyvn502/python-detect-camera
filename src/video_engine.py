"""
Video processing engine for extracting frames from security footage.

Handles video file enumeration, frame extraction with sampling,
and memory-efficient processing using generators.
"""

import cv2
import os
from pathlib import Path
from typing import Generator, Tuple, List
import tempfile
from PIL import Image
import numpy as np


class VideoEngine:
    """Handles video file processing and frame extraction."""
    
    def __init__(self, sample_rate: float = 0.5):
        """
        Initialize the video engine.
        
        Args:
            sample_rate: Number of frames to extract per second (e.g., 0.5 = 1 frame every 2 seconds)
        """
        self.sample_rate = sample_rate
        
        # Initialize OCR reader for timestamp extraction (lazy loading)
        self._ocr_reader = None
    
    def _get_ocr_reader(self):
        """Lazy load OCR reader (only when needed)."""
        if self._ocr_reader is None:
            try:
                import easyocr
                print("[OCR] Initializing text recognition...")
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("[OCR] Text recognition ready")
            except ImportError:
                print("[OCR] Warning: easyocr not installed. Camera timestamps will not be extracted.")
                self._ocr_reader = False  # Mark as unavailable
        return self._ocr_reader if self._ocr_reader is not False else None
    
    def extract_camera_timestamp(self, frame: np.ndarray) -> str:
        """
        Extract camera timestamp from frame overlay using OCR.
        Optimized to join split text blocks and handle common OCR misreads.
        """
        reader = self._get_ocr_reader()
        if reader is None:
            return ""
        
        try:
            height, width = frame.shape[:2]
            
            # Check top 20% and bottom 20%
            regions = [
                ("TOP", frame[0:int(height * 0.20), :]),
                ("BOTTOM", frame[int(height * 0.80):height, :])
            ]
            
            import re
            # Much more flexible pattern to handle misreads
            # [Date] [Time]
            # Date can be XX-XX-XXXX or XXXX-XX-XX
            # Time can be XX:XX:XX
            # We allow almost any symbol for separators to be safe
            date_part = r'[0-9O]{2,4}[^0-9a-zA-Z]{1,2}[0-9O]{2}[^0-9a-zA-Z]{1,2}[0-9O]{2,4}'
            time_part = r'[0-9O]{2}[^0-9a-zA-Z]{1,2}[0-9O]{2}[^0-9a-zA-Z]{1,2}[0-9O]{2}'
            ts_pattern = f'({date_part})\\s+({time_part})'
            
            for region_name, region in regions:
                # Pre-process: Adaptive thresholding is better for varying light
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                # Apply a slight blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                
                # Multi-stage check: Original, adaptive threshold, and fixed threshold
                adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                _, fixed = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)
                for img in [region, adaptive, fixed]:
                    results = reader.readtext(img, detail=0)
                    if not results:
                        continue
                        
                    combined_text = " ".join(results)
                    
                    match = re.search(ts_pattern, combined_text)
                    if match:
                        found_date = match.group(1)
                        found_time = match.group(2)
                        
                        # Clean up and normalize
                        found_date = found_date.replace('O', '0').replace('o', '0')
                        found_time = found_time.replace('O', '0').replace('o', '0')
                        
                        clean_time = re.sub(r'[^0-9]', ':', found_time)
                        result_ts = f"{found_date} {clean_time}"
                        return result_ts
                    elif results:
                        # Fallback for very messy reads: just try to find something that looks like numbers
                        digit_only = re.sub(r'[^0-9]', '', combined_text)
                        if len(digit_only) >= 12:
                            return f"RAW: {combined_text[:30]}"
            
            return ""

        except Exception as e:
            return ""



    
    @staticmethod
    def get_video_files(directory: str) -> List[Path]:
        """
        Get all video files from the specified directory.
        
        Args:
            directory: Path to the directory containing video files
            
        Returns:
            List of Path objects for video files
            
        Raises:
            ValueError: If directory doesn't exist or contains no video files
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        # Find all video files (multiple formats supported)
        video_extensions = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv", "*.MKV"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(dir_path.glob(ext))
        
        if not video_files:
            raise ValueError(f"No video files found in directory: {directory}")
        
        return sorted(video_files)
    
    def extract_frames(self, video_path: Path) -> Generator[Tuple[np.ndarray, str, float], None, None]:
        """
        Extract frames from a video at the specified sample rate.
        
        Uses a generator pattern to avoid loading all frames into memory.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple of (frame_array, timestamp_string, timestamp_seconds)
            
        Raises:
            ValueError: If video file cannot be opened
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval based on sample rate
            frame_interval = int(fps / self.sample_rate) if fps > 0 else 1
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Only process frames at the specified interval
                if frame_count % frame_interval == 0:
                    # Calculate timestamp
                    timestamp_seconds = frame_count / fps if fps > 0 else 0
                    timestamp_str = self._format_timestamp(timestamp_seconds)
                    
                    # Extract camera timestamp from frame overlay
                    camera_timestamp = self.extract_camera_timestamp(frame)
                    
                    yield frame, timestamp_str, timestamp_seconds, camera_timestamp

                
                frame_count += 1
        
        finally:
            cap.release()
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Format timestamp in HH:MM:SS format.
        
        Args:
            seconds: Timestamp in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def save_frame_temp(frame: np.ndarray) -> str:
        """
        Save a frame to a temporary file for upload to Gemini.
        
        Args:
            frame: Frame array from OpenCV
            
        Returns:
            Path to the temporary file
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=".jpg",
            prefix="frame_"
        )
        
        pil_image.save(temp_file.name, "JPEG", quality=85)
        
        return temp_file.name
    
    def get_video_info(self, video_path: Path) -> dict:
        """
        Get metadata about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video metadata
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            return {
                "filename": video_path.name,
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": self._format_timestamp(duration_seconds),
                "duration_seconds": duration_seconds,
            }
        finally:
            cap.release()
