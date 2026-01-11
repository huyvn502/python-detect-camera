# Project Brief: Silhouette-Match Video Processor

## 1. Context & Objective
The goal is to build a Python-based CLI tool that processes 24 hours of EZVIZ security footage (approx. 100+ video clips). The app must identify when a specific person—represented by a "target" silhouette/screenshot—appears in these videos. 

**Challenge:** The footage is dark/infrared. The "target" is a shape, not a clear face.
**Solution:** Use Gemini 1.5/3 Pro's multimodal capabilities to "reason" about human shapes in low light.

## 2. Technical Requirements
- **Language:** Python 3.10+
- **Input:** - A directory path containing multiple `.mp4` files.
    - A path to a `reference_image.jpg` (the silhouette to match).
- **Core Engine:** Google Generative AI (Gemini API). 
- **Processing Logic:** - Implement frame sampling (e.g., 1 frame per 1-2 seconds) to optimize token usage and speed.
    - Use Gemini's File API to upload videos/images for context.
- **Output:** - Console output in the format: `[Filename] | [Timestamp] | [Detection Summary]`.

## 3. Architecture & Modularity
The assistant must implement a modular structure to allow for future cloud deployment (Google Cloud Run/Functions) and new notification channels (Telegram/APIs).

### Required Modules:
1. `config.py`: Handles environment variables (API keys, directory paths).
2. `video_engine.py`: Uses OpenCV or MoviePy to handle file reading and sampling.
3. `ai_processor.py`: Interfaces with the Gemini API. 
    - **Prompt Logic:** "Compare the attached reference image silhouette to this video. Identify if the same individual appears. Note timestamps even if visibility is low."
4. `notifier.py`: A base class for reporting results. Currently only prints to console, but must have a placeholder for `TelegramNotifier`.
5. `main.py`: The entry point orchestrating the workflow.

## 4. Implementation Steps for the AI
1. **Initialize Project:** Create `requirements.txt` with `google-generativeai`, `opencv-python`, and `python-dotenv`.
2. **File Handling:** Create a utility to batch process files (don't load 100 videos into memory at once).
3. **AI Integration:** - Use the `GenerativeModel.generate_content` method.
    - Implement error handling for API rate limits (exponential backoff).
4. **Result Aggregation:** Collect all timestamps and print a final summary table to the console.

## 5. Constraints
- **No UI:** Purely a CLI/Background process.
- **Efficiency:** The system must skip frames to save processing time; 24 hours of video at 30fps is too heavy for direct processing.
- **Extensibility:** Ensure the `ai_processor` returns a structured JSON or Dictionary so the `notifier` can easily parse it.

---
**Instruction to AI Assistant:** "Please generate the project structure and the code for the modules mentioned above. Prioritize clean, documented code and handle the video sampling logic efficiently."