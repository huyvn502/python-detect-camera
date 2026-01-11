"""
Notification system for reporting detection results.

Provides extensible notification via base class pattern.
Currently implements console output with future support for Telegram and other channels.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime


class BaseNotifier(ABC):
    """Abstract base class for notification systems."""
    
    @abstractmethod
    def send(self, detection_result: Dict[str, Any]):
        """
        Send a single detection notification.
        
        Args:
            detection_result: Dictionary containing detection information
        """
        pass
    
    @abstractmethod
    def send_summary(self, summary: Dict[str, Any]):
        """
        Send a summary notification.
        
        Args:
            summary: Dictionary containing summary statistics
        """
        pass


class ConsoleNotifier(BaseNotifier):
    """Console-based notification implementation."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console notifier.
        
        Args:
            verbose: If True, print detailed information for each detection
        """
        self.verbose = verbose
        self.detections = []
    
    def send(self, detection_result: Dict[str, Any]):
        """
        Print detection result to console.
        
        Args:
            detection_result: Detection information dictionary
        """
        # Store for summary
        self.detections.append(detection_result)
        
        # Only print matches or if verbose mode
        is_match = detection_result.get("match", False)
        has_error = detection_result.get("error", False)
        
        if is_match or self.verbose or has_error:
            filename = detection_result.get("video_filename", "Unknown")
            timestamp = detection_result.get("timestamp", "00:00:00")
            camera_timestamp = detection_result.get("camera_timestamp", "")
            confidence = detection_result.get("confidence", 0)
            reasoning = detection_result.get("reasoning", "No reasoning provided")
            
            # Format output
            if is_match:
                status = f"✓ MATCH ({confidence}% confidence)"
                color = "\033[92m"  # Green
                reset = "\033[0m"
            elif has_error:
                status = "⚠ ERROR"
                color = "\033[93m"  # Yellow
                reset = "\033[0m"
            else:
                status = f"✗ No match ({confidence}% confidence)"
                color = "\033[90m"  # Gray
                reset = "\033[0m"
            
            if camera_timestamp:
                print(f"{color}[{filename:40}] | {timestamp} | 📷 {camera_timestamp} | {status}{reset}")
            else:
                print(f"{color}[{filename:40}] | {timestamp} | {status}{reset}")
            
            # Always show reasoning for matches
            if is_match or self.verbose:
                reasoning_vi = detection_result.get("reasoning_vi", "")
                print(f"  └─ {reasoning}")
                if reasoning_vi:
                    print(f"  └─ 🇻🇳 {reasoning_vi}")
    
    def send_summary(self, summary: Dict[str, Any]):
        """
        Print summary table to console.
        
        Args:
            summary: Summary statistics dictionary
        """
        total_videos = summary.get("total_videos", 0)
        total_frames = summary.get("total_frames_analyzed", 0)
        total_matches = summary.get("total_matches", 0)
        processing_time = summary.get("processing_time_seconds", 0)
        match_list = summary.get("matches", [])
        
        print("\n" + "=" * 80)
        print("DETECTION SUMMARY".center(80))
        print("=" * 80)
        print(f"Total Videos Processed:  {total_videos}")
        print(f"Total Frames Analyzed:   {total_frames}")
        print(f"Total Matches Found:     {total_matches}")
        print(f"Processing Time:         {self._format_duration(processing_time)}")
        print("=" * 80)
        
        if match_list:
            print("\nDETECTIONS:")
            print("-" * 80)
            for i, match in enumerate(match_list, 1):
                filename = match.get("video_filename", "Unknown")
                timestamp = match.get("timestamp", "00:00:00")
                camera_timestamp = match.get("camera_timestamp", "")
                confidence = match.get("confidence", 0)
                
                if camera_timestamp:
                    print(f"{i:2d}. [{filename:40}] | {timestamp} | 📷 {camera_timestamp} | {confidence}% confidence")
                else:
                    print(f"{i:2d}. [{filename:40}] | {timestamp} | {confidence}% confidence")
            print("-" * 80)
        else:
            print("\nNo matches found.")

        
        print()
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class TelegramNotifier(BaseNotifier):
    """
    Telegram notification implementation.
    
    Sends detection results and summaries to a Telegram chat.
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot API token
            chat_id: Telegram chat ID to send notifications to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        
        try:
            import telegram
            import asyncio
            self.telegram = telegram
            self.asyncio = asyncio
            # Initialize bot (synchronous wrapper)
            self.bot = telegram.Bot(token=bot_token)
        except ImportError:
            raise ImportError("python-telegram-bot is not installed. Run: pip install python-telegram-bot")
    
    def _send_message_sync(self, text: str):
        """Send message synchronously (wrapper for async method)."""
        try:
            # Create event loop if doesn't exist
            try:
                loop = self.asyncio.get_event_loop()
            except RuntimeError:
                loop = self.asyncio.new_event_loop()
                self.asyncio.set_event_loop(loop)
            
            # Send message
            loop.run_until_complete(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode='Markdown'
                )
            )
        except Exception as e:
            print(f"[Telegram] Error sending message: {e}")
    
    def send(self, detection_result: Dict[str, Any]):
        """Send a real-time detection alert with bilingual reasoning via Telegram."""
        if not detection_result.get("match", False):
            return

        filename = detection_result.get("video_filename", "Unknown")
        timestamp = detection_result.get("timestamp", "00:00:00")
        camera_time = detection_result.get("camera_timestamp", "")
        confidence = detection_result.get("confidence", 0)
        reasoning = detection_result.get("reasoning", "No reasoning provided")
        reasoning_vi = detection_result.get("reasoning_vi", "")

        message = "🎯 *DETECTION ALERT*\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += f"📹 *Video:* `{filename}`\n"
        
        time_info = timestamp
        if camera_time:
            time_info = f"{timestamp} (📷 {camera_time})"
        
        message += f"⏰ *Time:* `{time_info}`\n"
        message += f"✅ *Confidence:* `{confidence}%` \n\n"
        message += f"💡 *Details:* {reasoning}\n"
        if reasoning_vi:
            message += f"🇻🇳 *Chi tiết:* {reasoning_vi}\n"
        
        self._send_message_sync(message)
    
    def send_summary(self, summary: Dict[str, Any]):
        """Send a concise summary of all detections via Telegram (no reasoning)."""
        total_videos = summary.get("total_videos", 0)
        total_frames = summary.get("total_frames_analyzed", 0)
        total_matches = summary.get("total_matches", 0)
        processing_time = summary.get("processing_time_seconds", 0)
        match_list = summary.get("matches", [])
        
        # Build Header
        message = "📊 *DETECTION SUMMARY*\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += f"📂 *Videos:* `{total_videos}`\n"
        message += f"🎞 *Frames:* `{total_frames}`\n"
        message += f"✅ *Matches:* `{total_matches}`\n"
        message += f"⏱ *Time:* `{self._format_duration(processing_time)}` \n"
        
        if match_list:
            message += "\n📋 *LIST OF DETECTIONS:*\n"
            message += "━━━━━━━━━━━━━━━━━━━━\n"
            
            # Simple list without reasoning to keep summary short
            current_video = ""
            for i, match in enumerate(match_list, 1):
                filename = match.get("video_filename", "Unknown")
                timestamp = match.get("timestamp", "00:00:00")
                camera_time = match.get("camera_timestamp", "")
                confidence = match.get("confidence", 0)
                
                if filename != current_video:
                    message += f"\n🎬 `{filename}`\n"
                    current_video = filename
                
                time_info = timestamp
                if camera_time:
                    time_info = f"{timestamp} (📷 {camera_time})"
                
                message += f"• {i}. `{time_info}` | `{confidence}%` \n"
                
                # Length protection
                if len(message) > 3800:
                    message += "\n⚠️ *... (Truncated)*"
                    break
        else:
            message += "\n❌ *No human figures detected.*"
        
        self._send_message_sync(message)


    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"



class MultiNotifier(BaseNotifier):
    """
    Composite notifier that sends to multiple notification channels.
    
    Example:
        notifier = MultiNotifier([
            ConsoleNotifier(verbose=True),
            TelegramNotifier(bot_token, chat_id)
        ])
    """
    
    def __init__(self, notifiers: List[BaseNotifier]):
        """
        Initialize multi-channel notifier.
        
        Args:
            notifiers: List of notifier instances
        """
        self.notifiers = notifiers
    
    def send(self, detection_result: Dict[str, Any]):
        """Send detection to all configured notifiers."""
        for notifier in self.notifiers:
            try:
                notifier.send(detection_result)
            except Exception as e:
                print(f"[Notifier] Error in {notifier.__class__.__name__}: {e}")
    
    def send_summary(self, summary: Dict[str, Any]):
        """Send summary to all configured notifiers."""
        for notifier in self.notifiers:
            try:
                notifier.send_summary(summary)
            except Exception as e:
                print(f"[Notifier] Error in {notifier.__class__.__name__}: {e}")
