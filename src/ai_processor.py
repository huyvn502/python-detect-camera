"""
AI Processor for silhouette matching using Google Gemini API.

Handles image/frame analysis, person detection, and comparison against reference silhouette.
"""

from google import genai
from google.genai import types
from pathlib import Path
import time
import json
from typing import Optional, Dict, Any
import os


class AIProcessor:
    """Manages interaction with Gemini API for silhouette matching."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro", max_retries: int = 3, retry_delay: int = 2):
        """
        Initialize the AI processor.
        
        Args:
            api_key: Google Gemini API key
            model_name: Name of the Gemini model to use
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Initial delay in seconds between retries (uses exponential backoff)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.reference_image = None
        
        # Initialize new GenAI client
        self.client = genai.Client(api_key=api_key)

    
    def upload_reference_image(self, image_path: str) -> bool:
        """
        Upload and store the reference silhouette image.
        
        Args:
            image_path: Path to the reference image
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Reference image not found: {image_path}")
            
            # Upload file to Gemini using new API
            import mimetypes
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = 'image/png'  # Default fallback
            
            with open(image_path, 'rb') as f:
                self.reference_image = self.client.files.upload(file=f, config={'mime_type': mime_type})
            
            print(f"[AI] Reference image uploaded: {Path(image_path).name}")
            return True
            
        except Exception as e:
            print(f"[AI] Error uploading reference image: {e}")
            return False
    
    def analyze_frame(self, frame_path: str, video_filename: str, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a video frame against the reference silhouette, or detect any human if no reference.
        
        Args:
            frame_path: Path to the frame image file
            video_filename: Name of the source video file
            timestamp: Timestamp of the frame in the video
            
        Returns:
            Dictionary with detection results, or None if analysis fails
        """
        try:
            # Upload the frame using new API
            import mimetypes
            mime_type, _ = mimetypes.guess_type(frame_path)
            if not mime_type:
                mime_type = 'image/jpeg'  # Default for frames
            
            with open(frame_path, 'rb') as f:
                frame_file = self.client.files.upload(file=f, config={'mime_type': mime_type})
            
            # Choose prompt based on whether we have a reference image
            if self.reference_image:
                # Mode 1: Compare against reference silhouette
                prompt = self._build_analysis_prompt()
                response = self._call_api_with_retry(
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=[
                            types.Part(text=prompt),
                            types.Part(file_data=types.FileData(file_uri=self.reference_image.uri, mime_type=self.reference_image.mime_type)),
                            types.Part(file_data=types.FileData(file_uri=frame_file.uri, mime_type=frame_file.mime_type))
                        ]
                    )
                )
            else:
                # Mode 2: General human detection (no reference)
                prompt = self._build_detection_prompt()
                response = self._call_api_with_retry(
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=[
                            types.Part(text=prompt),
                            types.Part(file_data=types.FileData(file_uri=frame_file.uri, mime_type=frame_file.mime_type))
                        ]
                    )
                )

            
            if not response:
                return None
            
            # Parse the response
            result = self._parse_response(response.text, video_filename, timestamp)
            
            # Clean up uploaded frame file
            try:
                self.client.files.delete(name=frame_file.name)
            except:
                pass  # Ignore cleanup errors
            
            # Note: Do NOT delete the temp frame file here - it's needed for notifications
            # The caller (main.py) is responsible for cleanup after notifications are sent
            
            return result
            
        except Exception as e:
            print(f"[AI] Error analyzing frame: {e}")
            return None
    
    def _build_analysis_prompt(self) -> str:
        """
        Build the prompt for silhouette analysis.
        
        Returns:
            Formatted prompt string
        """
        return """You are analyzing security footage to detect a SPECIFIC PERSON by their shadow/silhouette.

CRITICAL: The reference image shows a HUMAN SHADOW/SILHOUETTE in low light - NOT objects, foliage, or items.

Task: Compare the HUMAN SHADOW in the reference image to HUMAN FIGURES in the video frame.

What to FOCUS ON (Human Features Only):
- Head shape and position
- Shoulder width and slope
- Torso length and shape
- Arm position and length
- Leg stance and proportions
- Overall body height-to-width ratio
- Posture (standing, sitting, crouching, etc.)

What to IGNORE:
- Background objects (foliage, debris, items)
- Clothing details (focus on body outline only)
- Objects the person might be carrying
- Environmental elements

Analysis Rules:
1. FOCUS ON CENTER: Prioritize human figures in the central 60% of the frame (the middle 6/10 region).
2. HIGH PRIORITY LANDMARK: Pay special attention to a **bright/glowing plant or bush** visible in the frame. Detections of a human figure standing near or behind this plant are of the highest interest.
3. IGNORE DISTANT/SMALL: Ignore figures that are too far away or appear very small (less than 10% of frame height).
4. ONLY match if you see a clear HUMAN FIGURE/SHADOW with similar body proportions.
5. The reference is a SHADOW - look for similar human shadows in the video.
6. Do NOT match objects, plants, or non-human shapes.
7. Focus on skeletal structure: head → shoulders → torso → legs.
8. Low-light/infrared footage shows humans as dark silhouettes.

Response Format (JSON only):
{
  "match": true or false,
  "confidence": 0-100 (integer percentage),
  "reasoning": "brief explanation focusing ONLY on human body features (in English)",
  "reasoning_vi": "Vietnamese translation of the reasoning above"
}

Important:
- Only return true if you see a HUMAN FIGURE with matching body proportions
- Confidence above 70% means you clearly see a matching HUMAN shadow
- Provide BOTH English (reasoning) and Vietnamese (reasoning_vi) explanations
- Return ONLY valid JSON, no other text"""
    
    def _build_detection_prompt(self) -> str:
        """
        Build the prompt for general human detection (no reference comparison).
        
        Returns:
            Formatted prompt string
        """
        return """You are analyzing infrared security footage from a house entrance camera.

SCENE DESCRIPTION:
This camera overlooks the front of a house. Key landmarks:
- LEFT SIDE: A narrow passage/corridor between a house wall and a metal mesh fence
- CENTER-LEFT: A **bright glowing plant/bush** (appears white in infrared) - this is the KEY LANDMARK
- RIGHT SIDE: Metal mesh fence and open area with some storage items

TARGET DETECTION ZONE (HIGH PRIORITY):
You are specifically monitoring the **narrow passage on the LEFT side of the frame**, which runs:
- From the top of the frame down to where the bright plant is located
- Between the house wall (left edge) and the metal fence (center)
- This is where intruders would walk to approach the house entrance

What to DETECT:
- ANY human figure, silhouette, or shadow in or near the narrow passage
- Humans appearing **around, next to, behind, or near the bright glowing plant**
- Dark human-shaped silhouettes (typical in infrared/night vision)
- Human body parts: head, shoulders, torso, limbs
- Humans in any posture: standing, walking, crouching, hiding

What to IGNORE:
- The bright glowing plant itself (it's vegetation, not human)
- Animals (cats, dogs, birds)
- Static objects (furniture, containers, wires, poles)
- Vague shadows without clear human form
- Activity on the far right side of frame (outside the main path)

Detection Rules:
1. PRIORITY: Focus analysis on the narrow passage (left portion) and area around the bright plant
2. A human silhouette typically shows: rounded head, shoulder line, vertical torso
3. Confidence 80%+ = Clear human shape visible
4. Confidence 60-79% = Possible human, shape is ambiguous
5. Confidence < 60% = Unlikely to be human

Response Format (JSON only):
{
  "match": true or false,
  "confidence": 0-100 (integer),
  "reasoning": "describe what you see - location relative to the bright plant, human features detected (in English)",
  "reasoning_vi": "Vietnamese translation of the reasoning"
}

Important:
- Return ONLY valid JSON
- Provide BOTH English and Vietnamese reasoning
- Be specific about WHERE in the frame you see the human (e.g., "left of the bright plant", "in the narrow passage")"""

    
    def _parse_response(self, response_text: str, video_filename: str, timestamp: str) -> Dict[str, Any]:
        """
        Parse Gemini API response into structured format.
        
        Args:
            response_text: Raw response from Gemini
            video_filename: Source video filename
            timestamp: Frame timestamp
            
        Returns:
            Structured detection result
        """
        try:
            # Try to extract JSON from response
            # Sometimes the model might add extra text, so we look for JSON block
            response_text = response_text.strip()
            
            # Find JSON content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Validate required fields
            if "match" not in data or "confidence" not in data:
                raise ValueError("Response missing required fields")
            
            # Add metadata
            data["video_filename"] = video_filename
            data["timestamp"] = timestamp
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"[AI] Failed to parse JSON response: {e}")
            print(f"[AI] Raw response: {response_text}")
            
            # Return a default structure
            return {
                "match": False,
                "confidence": 0,
                "reasoning": f"Failed to parse response: {str(e)}",
                "video_filename": video_filename,
                "timestamp": timestamp,
                "error": True
            }
        except Exception as e:
            print(f"[AI] Error parsing response: {e}")
            return {
                "match": False,
                "confidence": 0,
                "reasoning": str(e),
                "video_filename": video_filename,
                "timestamp": timestamp,
                "error": True
            }
    
    def _call_api_with_retry(self, api_call):
        """
        Call Gemini API without retry - fails immediately on error.
        
        Args:
            api_call: Callable function that makes the API request
            
        Returns:
            API response or None if API call fails
        """
        try:
            return api_call()
        except Exception as e:
            error_msg = str(e)
            print(f"[AI] API call failed: {error_msg}")
            print(f"[AI] No retries configured - failing immediately")
            return None
        
        return None
    
    def cleanup(self):
        """Clean up uploaded files from Gemini."""
        try:
            if self.reference_image:
                try:
                    self.client.files.delete(name=self.reference_image.name)
                    print("[AI] Cleaned up reference image")
                except KeyboardInterrupt:
                    raise  # Re-raise keyboard interrupt
                except Exception as e:
                    # Silently ignore cleanup errors (file may already be deleted)
                    pass
        except KeyboardInterrupt:
            # Don't suppress keyboard interrupts
            pass
        except Exception as e:
            # Catch any other unexpected errors
            pass
