# Silhouette-Match Video Processor

A Python CLI tool that uses **Google Gemini AI** to detect specific persons in security camera footage by comparing video frames against a reference silhouette image. Optimized for dark/infrared footage where traditional face recognition fails.

## 🎯 Use Case

Process 24 hours of EZVIZ security footage (100+ video clips) to identify when a specific person appears, even in low-light or infrared conditions. The system focuses on body shape, posture, and proportions rather than facial features.

## ✨ Features

- **AI-Powered Detection**: Uses Gemini 1.5/2.0 Pro's multimodal vision capabilities
- **Efficient Frame Sampling**: Configurable frame extraction rate (default: 1 frame per 2 seconds)
- **Memory Efficient**: Processes videos incrementally without loading everything into memory
- **Extensible Architecture**: Modular design ready for cloud deployment and additional notification channels
- **Retry Logic**: Exponential backoff for API rate limit handling
- **Detailed Reporting**: Console output with match confidence scores and reasoning

## 📋 Requirements

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- Video files in `.mp4` format
- Reference silhouette image (`.jpg`, `.png`)

## 🚀 Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /Users/huyvn/Codes/personal-projects/python-detect-camera
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## 🎮 Usage

### Basic Usage

```bash
source venv/bin/activate
python src/main.py --videos videos --reference "ref/file-ref.png"
```

### With Options

```bash
python src/main.py \
  --videos ./footage \
  --reference ./target_person.jpg \
  --verbose \
  --confidence-threshold 60
```

### Command-Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--videos` | `-v` | Yes | Directory containing `.mp4` video files |
| `--reference` | `-r` | Yes | Path to reference silhouette image |
| `--verbose` | | No | Print detailed info for all frames (not just matches) |
| `--confidence-threshold` | | No | Minimum confidence % for reporting matches (default: 50) |

## 📊 Example Output

### Processing Output
```
================================================================================
                    SILHOUETTE-MATCH VIDEO PROCESSOR                    
                        Powered by Google Gemini AI                     
================================================================================

⚙️  Configuration:
   model: gemini-1.5-pro
   frame_sample_rate: 0.5
   batch_size: 5
   max_retries: 3
   confidence_threshold: 50

📂 Found 3 video files in ./footage

📸 Uploading reference image: ./target_person.jpg
[AI] Reference image uploaded: target_person.jpg

🎬 Processing 3 video files...

📹 Video 1/3: camera_front_20240110.mp4
   Duration: 01:30:45 | Resolution: 1920x1080
   ✓ Analyzed 2715 frames from this video

✓ MATCH (75% confidence) | camera_front_20240110.mp4 | 00:12:34
  └─ Similar shoulder profile and height-to-width ratio

✓ MATCH (82% confidence) | camera_back_20240110.mp4 | 01:45:12
  └─ Body proportions and posture align with reference
```

### Summary Report
```
================================================================================
                              DETECTION SUMMARY                              
================================================================================
Total Videos Processed:  3
Total Frames Analyzed:   8145
Total Matches Found:     12
Processing Time:         45m 23s
================================================================================

DETECTIONS:
--------------------------------------------------------------------------------
 1. [camera_front_20240110.mp4              ] | 00:12:34 | 75% confidence
    └─ Similar shoulder profile and height-to-width ratio
 2. [camera_back_20240110.mp4               ] | 01:45:12 | 82% confidence
    └─ Body proportions and posture align with reference
...
--------------------------------------------------------------------------------
```

## 🏗️ Project Structure

```
python-detect-camera/
├── src/
│   ├── __init__.py         # Package initializer
│   ├── config.py           # Configuration management
│   ├── video_engine.py     # Video processing & frame extraction
│   ├── ai_processor.py     # Gemini API integration
│   ├── notifier.py         # Notification system
│   └── main.py             # CLI entry point
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── requirement.md          # Original project specification
└── README.md              # This file
```

## ⚙️ Configuration

All settings can be configured via `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *Required* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Gemini model to use |
| `FRAME_SAMPLE_RATE` | `0.5` | Frames per second to extract (0.5 = 1 every 2s) |
| `BATCH_SIZE` | `5` | Videos to process concurrently |
| `MAX_RETRIES` | `3` | API retry attempts |
| `RETRY_DELAY` | `2` | Initial retry delay in seconds |
| `CONFIDENCE_THRESHOLD` | `50` | Minimum % confidence for matches |

## 🔧 Architecture

### Modular Design

The project uses a modular architecture for maintainability and future extensibility:

1. **`config.py`**: Centralized configuration with validation
2. **`video_engine.py`**: Video file handling and frame extraction using OpenCV
3. **`ai_processor.py`**: Gemini API integration with retry logic
4. **`notifier.py`**: Extensible notification system (currently console, prepared for Telegram)
5. **`main.py`**: Orchestration and CLI interface

### Frame Sampling Strategy

To optimize API usage and processing speed:
- Extract frames at configurable rate (default: 0.5 fps = 1 frame every 2 seconds)
- 1 hour of video → ~1,800 frames instead of 108,000 frames (at 30fps)
- Significant cost and time savings while maintaining detection accuracy

### AI Prompt Engineering

The system uses a carefully crafted prompt that:
- Focuses on body shape, posture, and proportions
- Accounts for low-light/infrared conditions
- Returns structured JSON with match status, confidence, and reasoning

## 🚀 Future Enhancements

### Cloud Deployment
Ready for deployment to:
- Google Cloud Run
- Google Cloud Functions
- AWS Lambda

### Additional Notification Channels
The `BaseNotifier` class enables easy addition of:
- ✅ Console output (implemented)
- 📱 Telegram notifications (placeholder ready)
- 📧 Email notifications
- 🔗 Webhook/API callbacks

To add Telegram support:
1. Install `python-telegram-bot`
2. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env`
3. Implement the `TelegramNotifier` class methods

### Potential Improvements
- Real-time processing with video stream input
- Web dashboard for results visualization
- Database storage for historical analysis
- Multi-person tracking support
- GPU acceleration for faster frame processing

## 🐛 Troubleshooting

### "GEMINI_API_KEY is required"
- Make sure you've created a `.env` file (copy from `.env.example`)
- Add your actual API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### "No video files found"
- Ensure your videos are in `.mp4` format
- Check the directory path is correct
- Videos must be directly in the specified directory (not subdirectories)

### Rate Limit Errors
- The system automatically retries with exponential backoff
- If persistent, increase `RETRY_DELAY` or `MAX_RETRIES` in `.env`
- Consider reducing `FRAME_SAMPLE_RATE` to process fewer frames

### Low Detection Accuracy
- Try adjusting `CONFIDENCE_THRESHOLD` (lower = more sensitive)
- Ensure reference image shows clear body silhouette
- Consider using `--verbose` to see all analysis results

## 📝 License

This project is for personal use. Modify as needed for your requirements.

## 🙏 Acknowledgments

- Google Gemini AI for multimodal vision capabilities
- OpenCV for video processing
- Python community for excellent libraries

---

**Need help?** Check the `requirement.md` file for the original project specification.
