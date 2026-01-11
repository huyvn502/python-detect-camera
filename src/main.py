"""
Main entry point for Silhouette-Match Video Processor.

Orchestrates the workflow: video processing, AI analysis, and result notification.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Suppress PyTorch/MPS warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


from config import Config
from video_engine import VideoEngine
from ai_processor import AIProcessor
from notifier import ConsoleNotifier, TelegramNotifier, MultiNotifier



def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Silhouette-Match Video Processor - Detect specific persons in security footage using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --videos ./footage --reference ./target_person.jpg
  python src/main.py -v ./videos -r ./silhouette.jpg --verbose
        """
    )
    
    parser.add_argument(
        "-v", "--videos",
        required=True,
        help="Directory containing video files (.mp4)"
    )
    
    parser.add_argument(
        "-r", "--reference",
        required=False,
        default=None,
        help="Path to reference silhouette image (optional - if not provided, will detect any human)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for all frames (not just matches)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=None,
        help="Minimum confidence threshold for reporting matches (0-100)"
    )
    
    return parser.parse_args()


def validate_inputs(videos_dir: str, reference_image: str = None):
    """
    Validate input paths.
    
    Args:
        videos_dir: Path to videos directory
        reference_image: Path to reference image (optional)
        
    Raises:
        SystemExit: If validation fails
    """
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        print(f"❌ Error: Videos directory not found: {videos_dir}")
        sys.exit(1)
    
    if not videos_path.is_dir():
        print(f"❌ Error: Videos path is not a directory: {videos_dir}")
        sys.exit(1)
    
    # Validate reference image only if provided
    if reference_image:
        reference_path = Path(reference_image)
        if not reference_path.exists():
            print(f"❌ Error: Reference image not found: {reference_image}")
            sys.exit(1)
        
        if not reference_path.is_file():
            print(f"❌ Error: Reference path is not a file: {reference_image}")
            sys.exit(1)


def process_videos(
    video_files: List[Path],
    reference_image: str,
    ai_processor: AIProcessor,
    video_engine: VideoEngine,
    notifier: ConsoleNotifier,
    confidence_threshold: int
) -> Dict[str, Any]:
    """
    Process all video files and detect matches.
    
    Args:
        video_files: List of video file paths
        reference_image: Path to reference image
        ai_processor: Initialized AI processor
        video_engine: Video processing engine
        notifier: Notification handler
        confidence_threshold: Minimum confidence for matches
        
    Returns:
        Summary statistics dictionary
    """
    start_time = time.time()
    
    total_frames_analyzed = 0
    matches = []
    
    print(f"\n🎬 Processing {len(video_files)} video files...\n")
    
    # Process each video file
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\n📹 Video {video_idx}/{len(video_files)}: {video_path.name}")
        
        # Get video info for progress bar
        video_info = video_engine.get_video_info(video_path)
        print(f"   Duration: {video_info.get('duration', 'Unknown')} | "
              f"Resolution: {video_info.get('width')}x{video_info.get('height')}")
        
        # Extract and analyze frames
        frame_count = 0
        
        try:
            for frame, timestamp, _, camera_timestamp in video_engine.extract_frames(video_path):
                frame_count += 1
                total_frames_analyzed += 1
                
                # Save frame temporarily
                temp_frame_path = video_engine.save_frame_temp(frame)
                
                # Analyze frame
                result = ai_processor.analyze_frame(
                    temp_frame_path,
                    video_path.name,
                    timestamp
                )
                
                # Add camera timestamp to result if available
                if result and camera_timestamp:
                    result["camera_timestamp"] = camera_timestamp
                
                if result:
                    # Check if it's a match above threshold
                    if result.get("match") and result.get("confidence", 0) >= confidence_threshold:
                        matches.append(result)
                    
                    # Notify
                    notifier.send(result)
        
        except Exception as e:
            print(f"   ⚠️  Error processing video: {e}")
            continue
        
        print(f"   ✓ Analyzed {frame_count} frames from this video")
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Prepare summary
    summary = {
        "total_videos": len(video_files),
        "total_frames_analyzed": total_frames_analyzed,
        "total_matches": len(matches),
        "processing_time_seconds": processing_time,
        "matches": matches
    }
    
    return summary


def main():
    """Main execution function."""
    
    # Parse arguments first
    args = parse_arguments()
    
    # Print banner based on mode
    print("\n" + "=" * 80)
    if args.reference:
        print("SILHOUETTE-MATCH VIDEO PROCESSOR".center(80))
    else:
        print("HUMAN DETECTION VIDEO PROCESSOR".center(80))
    print("Powered by Google Gemini AI".center(80))
    print("=" * 80 + "\n")

    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    
    # Set confidence threshold
    confidence_threshold = args.confidence_threshold or Config.CONFIDENCE_THRESHOLD
    
    # Display configuration
    print("⚙️  Configuration:")
    config_summary = Config.get_summary()
    for key, value in config_summary.items():
        print(f"   {key}: {value}")
    print(f"   confidence_threshold: {confidence_threshold}")
    print()
    
    # Validate inputs
    validate_inputs(args.videos, args.reference)
    
    # Initialize components
    print("🔧 Initializing components...")
    
    try:
        video_engine = VideoEngine(sample_rate=Config.FRAME_SAMPLE_RATE)
        ai_processor = AIProcessor(
            api_key=Config.GEMINI_API_KEY,
            model_name=Config.GEMINI_MODEL,
            max_retries=Config.MAX_RETRIES,
            retry_delay=Config.RETRY_DELAY
        )
        
        # Initialize notifiers
        notifiers = [ConsoleNotifier(verbose=args.verbose)]
        
        # Add Telegram notifier if configured
        if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
            try:
                telegram_notifier = TelegramNotifier(
                    bot_token=Config.TELEGRAM_BOT_TOKEN,
                    chat_id=Config.TELEGRAM_CHAT_ID
                )
                notifiers.append(telegram_notifier)
                print("   ✓ Telegram notifications enabled")
            except Exception as e:
                print(f"   ⚠️  Telegram setup failed: {e}")
        
        # Use MultiNotifier to send to all channels
        notifier = MultiNotifier(notifiers)

        
        print("   ✓ Video engine initialized")
        print("   ✓ AI processor initialized")
        print("   ✓ Notifier initialized")
        
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        sys.exit(1)
    
    # Get video files
    try:
        video_files = video_engine.get_video_files(args.videos)
        print(f"\n📂 Found {len(video_files)} video files in {args.videos}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Upload reference image if provided
    if args.reference:
        print(f"\n📸 Uploading reference image: {args.reference}")
        if not ai_processor.upload_reference_image(args.reference):
            print("❌ Failed to upload reference image")
            sys.exit(1)
    else:
        print("\n🔍 Mode: General human detection (no reference image)")
    
    # Process videos
    try:
        summary = process_videos(
            video_files=video_files,
            reference_image=args.reference,
            ai_processor=ai_processor,
            video_engine=video_engine,
            notifier=notifier,
            confidence_threshold=confidence_threshold
        )
        
        # Send summary
        notifier.send_summary(summary)
        
        # Cleanup
        ai_processor.cleanup()
        
        print("✅ Processing complete!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        ai_processor.cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        ai_processor.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
