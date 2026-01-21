"""
Telegram Notification Module
Sends alerts to Telegram groups/channels
"""
import os
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Telegram configuration from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("bot_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("chat_id")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Optional: Disable Telegram notifications via environment variable
TELEGRAM_DISABLED = os.getenv("TELEGRAM_DISABLED", "").lower() in ("true", "1", "yes")


class TelegramNotifier:
    """Centralized Telegram notification handler"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat/group ID (defaults to TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.enabled = bool(self.bot_token and self.chat_id) and not TELEGRAM_DISABLED
        
        if not self.enabled:
            if not self.bot_token:
                logger.warning("⚠️ Telegram notifications DISABLED: TELEGRAM_BOT_TOKEN not set")
            elif not self.chat_id:
                logger.warning("⚠️ Telegram notifications DISABLED: TELEGRAM_CHAT_ID not set")
            elif TELEGRAM_DISABLED:
                logger.info("ℹ️ Telegram notifications DISABLED: TELEGRAM_DISABLED=true")
            else:
                logger.warning("⚠️ Telegram notifications DISABLED: Unknown reason")
        else:
            logger.info(f"✅ Telegram notifier initialized successfully (chat_id: {self.chat_id[:10]}...)")
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message to Telegram
        
        Args:
            text: Message text (supports HTML formatting)
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            resp = requests.post(url, data=data, timeout=10)
            
            if resp.status_code == 200:
                logger.info("✅ Telegram message sent successfully")
                return True
            else:
                logger.error(f"❌ Telegram API error: {resp.status_code} - {resp.text}")
                # Log response details for debugging
                try:
                    error_data = resp.json()
                    logger.error(f"   Error details: {error_data}")
                except:
                    pass
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        Send a photo to Telegram
        
        Args:
            photo_path: Path to image file
            caption: Optional caption text
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Check if file exists
            if not os.path.exists(photo_path):
                logger.warning(f"Photo file not found: {photo_path}")
                return False
            
            # Check file size (Telegram limit: 10MB for photos)
            file_size = os.path.getsize(photo_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"Photo file too large ({file_size / 1024 / 1024:.1f}MB), sending as document instead")
                return self.send_document(photo_path, caption=caption, parse_mode=parse_mode)
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption[:1024] if len(caption) > 1024 else caption,  # Telegram caption limit: 1024 chars
                    "parse_mode": parse_mode
                }
                resp = requests.post(url, files=files, data=data, timeout=30)
            
            if resp.status_code == 200:
                logger.info(f"✅ Telegram photo sent successfully: {os.path.basename(photo_path)} ({file_size / 1024:.1f}KB)")
                return True
            else:
                logger.warning(f"Telegram API error: {resp.status_code} - {resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram photo: {e}", exc_info=True)
            return False
    
    def send_document(self, document_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        Send a document (GIF, video, etc.) to Telegram
        
        Args:
            document_path: Path to document file
            caption: Optional caption text
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Check if file exists
            if not os.path.exists(document_path):
                logger.warning(f"Document file not found: {document_path}")
                return False
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"
            
            with open(document_path, 'rb') as doc:
                files = {'document': doc}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": parse_mode
                }
                resp = requests.post(url, files=files, data=data, timeout=30)
            
            if resp.status_code == 200:
                file_size = os.path.getsize(document_path)
                logger.info(f"✅ Telegram document sent successfully: {os.path.basename(document_path)} ({file_size / 1024:.1f}KB)")
                return True
            else:
                logger.warning(f"Telegram API error: {resp.status_code} - {resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram document: {e}", exc_info=True)
            return False
    
    def send_alert(
        self,
        channel_id: str,
        alert_type: str,
        alert_message: str,
        image_path: Optional[str] = None,
        alert_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a formatted alert to Telegram
        
        Args:
            channel_id: Channel identifier
            alert_type: Type of alert (e.g., 'fall_alert', 'smoking_alert')
            alert_message: Alert message text
            image_path: Optional path to image/GIF file
            alert_data: Optional additional alert data
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Format alert message
            alert_emoji = self._get_alert_emoji(alert_type)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build message
            message_parts = [
                f"{alert_emoji} <b>Alert: {self._format_alert_type(alert_type)}</b>",
                f"📍 <b>Channel:</b> {channel_id}",
                f"⏰ <b>Time:</b> {timestamp}",
                f"",
                f"📝 <b>Message:</b> {alert_message}"
            ]
            
            # Add additional data if available
            if alert_data:
                if isinstance(alert_data, dict):
                    # Add relevant fields from alert_data
                    if 'detection_count' in alert_data:
                        message_parts.append(f"🔢 <b>Detections:</b> {alert_data['detection_count']}")
                    if 'crowd_count' in alert_data:
                        message_parts.append(f"👥 <b>Crowd Count:</b> {alert_data['crowd_count']}")
                        if 'threshold' in alert_data:
                            message_parts.append(f"📊 <b>Threshold:</b> {alert_data['threshold']}")
                    if 'raw_count' in alert_data:
                        message_parts.append(f"📈 <b>Raw Count:</b> {alert_data['raw_count']}")
                    if 'long_stay_count' in alert_data:
                        message_parts.append(f"⏱️ <b>Long Stay Count:</b> {alert_data['long_stay_count']}")
                    if 'duration' in alert_data:
                        duration = alert_data['duration']
                        if isinstance(duration, (int, float)):
                            message_parts.append(f"⏱️ <b>Duration:</b> {duration:.0f}s")
                    if 'violations' in alert_data:
                        violations = alert_data['violations']
                        if isinstance(violations, list):
                            violations = ", ".join(violations)
                        message_parts.append(f"⚠️ <b>Violations:</b> {violations}")
                    if 'waiting_time' in alert_data:
                        wait_min = alert_data['waiting_time'] / 60
                        message_parts.append(f"⏱️ <b>Wait Time:</b> {wait_min:.1f} min")
                    if 'order_wait_time' in alert_data and alert_data['order_wait_time'] is not None:
                        order_wait_min = alert_data['order_wait_time'] / 60
                        message_parts.append(f"📝 <b>Order Wait:</b> {order_wait_min:.1f} min")
                    if 'service_wait_time' in alert_data and alert_data['service_wait_time'] is not None:
                        service_wait_min = alert_data['service_wait_time'] / 60
                        message_parts.append(f"🍽️ <b>Service Wait:</b> {service_wait_min:.1f} min")
                    if 'table_id' in alert_data:
                        message_parts.append(f"🪑 <b>Table:</b> {alert_data['table_id']}")
                    if 'violation_type' in alert_data:
                        violation_type = alert_data['violation_type']
                        if violation_type == 'order_wait':
                            message_parts.append(f"⚠️ <b>Type:</b> Order Wait Violation")
                        elif violation_type == 'service_wait':
                            message_parts.append(f"⚠️ <b>Type:</b> Service Wait Violation")
            
            message = "\n".join(message_parts)
            
            # Send with image if available
            if image_path:
                # Try to resolve the actual file path
                resolved_path = self._resolve_image_path(image_path)
                
                if resolved_path and os.path.exists(resolved_path):
                    logger.info(f"Sending Telegram alert with image: {resolved_path}")
                    # Determine if it's a GIF or image
                    file_ext = Path(resolved_path).suffix.lower()
                    if file_ext == '.gif':
                        return self.send_document(resolved_path, caption=message)
                    else:
                        # Send as photo for better display (jpg, png, etc.)
                        return self.send_photo(resolved_path, caption=message)
                else:
                    logger.warning(f"Image file not found for Telegram alert: {image_path} (resolved: {resolved_path})")
                    # Send text only if image not found
                    return self.send_message(message)
            else:
                # Send text only
                logger.debug("Sending Telegram alert without image")
                return self.send_message(message)
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            return False
    
    def _get_alert_emoji(self, alert_type: str) -> str:
        """Get emoji for alert type"""
        emoji_map = {
            'fall_alert': '🚨',
            'smoking_alert': '🚬',
            'fire_smoke_alert': '🔥',
            'person_smoking_alert': '🚬',
            'unauthorized_entry_alert': '🚫',
            'queue_alert': '👥',
            'people_alert': '👥',
            'crowd_alert': '👥',
            'cash_alert': '💰',
            'dresscode_alert': '👔',
            'ppe_alert': '🦺',
            'grooming_alert': '💇',
            'queue_violation': '⚠️',
            'table_service_violation': '🍽️',
            'table_cleanliness_violation': '🧹',
            'material_theft_alert': '📦',
            'phone_alert': '📱',
            'mopping_alert': '🧹',
            'restricted_area_alert': '🚧'
        }
        return emoji_map.get(alert_type, '⚠️')
    
    def _format_alert_type(self, alert_type: str) -> str:
        """Format alert type for display"""
        type_map = {
            'fall_alert': 'Fall Detection',
            'smoking_alert': 'Smoking Detection',
            'fire_smoke_alert': 'Fire/Smoke Detection',
            'person_smoking_alert': 'Person Smoking',
            'unauthorized_entry_alert': 'Unauthorized Entry',
            'queue_alert': 'Queue Alert',
            'people_alert': 'People Alert',
            'crowd_alert': 'Crowd Detection',
            'cash_alert': 'Cash Detection',
            'dresscode_alert': 'Dress Code Violation',
            'ppe_alert': 'PPE Violation',
            'grooming_alert': 'Grooming Violation',
            'queue_violation': 'Queue Violation',
            'table_service_violation': 'Table Service Violation',
            'table_cleanliness_violation': 'Table Cleanliness Violation',
            'material_theft_alert': 'Material Theft',
            'phone_alert': 'Phone Usage',
            'mopping_alert': 'Mopping Detection',
            'restricted_area_alert': 'Restricted Area Violation'
        }
        return type_map.get(alert_type, alert_type.replace('_', ' ').title())
    
    def _resolve_image_path(self, image_path: str) -> Optional[str]:
        """
        Resolve image path by trying multiple possible locations
        
        Args:
            image_path: Original image path (can be relative or absolute)
            
        Returns:
            Resolved absolute path if file exists, None otherwise
        """
        if not image_path:
            return None
        
        # If already absolute and exists, return it
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        
        # Try different path variations
        possible_paths = [
            image_path,  # Original path
            os.path.join("static", image_path),  # Relative to static
            os.path.abspath(image_path),  # Absolute from current dir
            os.path.abspath(os.path.join("static", image_path)),  # Absolute static path
        ]
        
        # Also try with just the filename in common directories
        filename = os.path.basename(image_path)
        static_dirs = [
            "static/alerts",
            "static/fall_snapshots",
            "static/cash_snapshots",
            "static/service_discipline",
            "static/smoking_snapshots",
            "static/phone_snapshots",
            "static/dresscode_snapshots",
            "static/ppe_snapshots",
            "static/grooming_snapshots",
            "static/table_service_violations",
            "static/table_cleanliness_violations",
            "static/mopping_snapshots",
            "static/restricted_area_snapshots",
        ]
        
        for static_dir in static_dirs:
            possible_paths.append(os.path.join(static_dir, filename))
            possible_paths.append(os.path.abspath(os.path.join(static_dir, filename)))
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                logger.debug(f"Resolved image path: {image_path} -> {path}")
                return path
        
        logger.warning(f"Could not resolve image path: {image_path}")
        return None


# Global instance
_telegram_notifier = None

def get_telegram_notifier() -> TelegramNotifier:
    """Get or create global Telegram notifier instance"""
    global _telegram_notifier
    if _telegram_notifier is None:
        _telegram_notifier = TelegramNotifier()
    return _telegram_notifier

