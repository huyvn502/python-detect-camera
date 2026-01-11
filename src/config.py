"""
Configuration management for Silhouette-Match Video Processor.

Handles environment variables and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for application settings."""
    
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
    # Frame Sampling Configuration
    # How many frames to extract per second (e.g., 0.5 = 1 frame every 2 seconds)
    FRAME_SAMPLE_RATE = float(os.getenv("FRAME_SAMPLE_RATE", "0.5"))
    
    # Processing Configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
    
    # API Retry Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
    
    # Confidence threshold for reporting matches (0-100)
    CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "50"))
    
    # Telegram Notification Configuration (Optional)
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    
    @classmethod
    def validate(cls):
        """
        Validate that all required configuration is present.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required. Please set it in your .env file.\n"
                "You can get an API key from: https://makersuite.google.com/app/apikey"
            )
        
        if cls.FRAME_SAMPLE_RATE <= 0:
            raise ValueError("FRAME_SAMPLE_RATE must be greater than 0")
        
        if cls.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be greater than 0")
    
    @classmethod
    def get_summary(cls):
        """Return a summary of current configuration settings."""
        return {
            "model": cls.GEMINI_MODEL,
            "frame_sample_rate": cls.FRAME_SAMPLE_RATE,
            "batch_size": cls.BATCH_SIZE,
            "max_retries": cls.MAX_RETRIES,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
        }
