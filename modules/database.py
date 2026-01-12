"""
Database Models for Sakshi.AI
"""
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from zoneinfo import ZoneInfo
import json
import logging
import os

logger = logging.getLogger(__name__)

def _send_telegram_alert(channel_id, alert_type, alert_message, snapshot_path=None, alert_data=None):
    """Helper function to send Telegram alert notification"""
    try:
        from modules.telegram_notifier import get_telegram_notifier
        notifier = get_telegram_notifier()
        
        # The notifier will handle path resolution internally
        # Just pass the snapshot_path as-is (can be relative or absolute)
        notifier.send_alert(
            channel_id=channel_id,
            alert_type=alert_type,
            alert_message=alert_message or f"Alert from {channel_id}",
            image_path=snapshot_path,  # Pass as-is, let notifier resolve
            alert_data=alert_data
        )
    except Exception as tg_error:
        logger.warning(f"Failed to send Telegram notification: {tg_error}", exc_info=True)

# IST timezone helper function
def get_ist_now():
    """Get current time in IST timezone"""
    return datetime.now(ZoneInfo("Asia/Kolkata"))

class DatabaseManager:
    def __init__(self, db):
        self.db = db
        self.define_models()
    
    def define_models(self):
        """Define database models"""
        
        # User authentication
        class User(self.db.Model):
            __tablename__ = 'users'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            username = self.db.Column(self.db.String(50), unique=True, nullable=False)
            password_hash = self.db.Column(self.db.String(255), nullable=False)
            role = self.db.Column(self.db.String(20), nullable=False)  # 'admin' or 'user'
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
            last_login = self.db.Column(self.db.DateTime)
        
        # Footfall tracking for People Counter
        class DailyFootfall(self.db.Model):
            __tablename__ = 'daily_footfall'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            report_date = self.db.Column(self.db.Date, nullable=False)
            in_count = self.db.Column(self.db.Integer, default=0)
            out_count = self.db.Column(self.db.Integer, default=0)
            
            __table_args__ = (self.db.UniqueConstraint('channel_id', 'report_date'),)
        
        class HourlyFootfall(self.db.Model):
            __tablename__ = 'hourly_footfall'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            report_date = self.db.Column(self.db.Date, nullable=False)
            hour = self.db.Column(self.db.Integer, nullable=False)  # 0-23
            in_count = self.db.Column(self.db.Integer, default=0)
            out_count = self.db.Column(self.db.Integer, default=0)
            
            __table_args__ = (self.db.UniqueConstraint('channel_id', 'report_date', 'hour'),)
        
        # Queue analytics for Queue Monitor
        class QueueAnalytics(self.db.Model):
            __tablename__ = 'queue_analytics'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            timestamp = self.db.Column(self.db.DateTime, default=get_ist_now)
            queue_count = self.db.Column(self.db.Integer, default=0)
            counter_count = self.db.Column(self.db.Integer, default=0)
            alert_triggered = self.db.Column(self.db.Boolean, default=False)
            alert_message = self.db.Column(self.db.Text)
        
        # Configuration storage
        class ChannelConfig(self.db.Model):
            __tablename__ = 'channel_config'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            app_name = self.db.Column(self.db.String(50), nullable=False)
            config_type = self.db.Column(self.db.String(50), nullable=False)  # 'roi', 'counting_line', 'settings'
            config_data = self.db.Column(self.db.Text)  # JSON data
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
            updated_at = self.db.Column(self.db.DateTime, default=get_ist_now, onupdate=get_ist_now)
            
            __table_args__ = (self.db.UniqueConstraint('channel_id', 'app_name', 'config_type'),)
        
        # Detection events for audit trail
        class DetectionEvent(self.db.Model):
            __tablename__ = 'detection_events'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            app_name = self.db.Column(self.db.String(50), nullable=False)
            event_type = self.db.Column(self.db.String(50), nullable=False)  # 'person_in', 'person_out', 'queue_alert'
            event_data = self.db.Column(self.db.Text)  # JSON data
            confidence = self.db.Column(self.db.Float)
            timestamp = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # RTSP channels configuration
        class RTSPChannel(self.db.Model):
            __tablename__ = 'rtsp_channels'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), unique=True, nullable=False)
            name = self.db.Column(self.db.String(100), nullable=False)
            rtsp_url = self.db.Column(self.db.String(500), nullable=False)
            description = self.db.Column(self.db.Text)
            is_active = self.db.Column(self.db.Boolean, default=True)
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
            updated_at = self.db.Column(self.db.DateTime, default=get_ist_now, onupdate=get_ist_now)
        
        # Alert GIF recordings
        class AlertGif(self.db.Model):
            __tablename__ = 'alert_gifs'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            alert_type = self.db.Column(self.db.String(50), nullable=False)  # 'queue_alert', 'people_alert', etc.
            gif_filename = self.db.Column(self.db.String(255), nullable=False)
            gif_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with alert details
            frame_count = self.db.Column(self.db.Integer)
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            duration_seconds = self.db.Column(self.db.Float)
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Heatmap snapshots
        class HeatmapSnapshot(self.db.Model):
            __tablename__ = 'heatmap_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            hotspot_count = self.db.Column(self.db.Integer, default=0)
            hotspots_data = self.db.Column(self.db.Text)  # JSON data with hotspot locations
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Cash detection snapshots
        class CashSnapshot(self.db.Model):
            __tablename__ = 'cash_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with detection details
            detection_count = self.db.Column(self.db.Integer, default=0)
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Fall detection snapshots
        class FallSnapshot(self.db.Model):
            __tablename__ = 'fall_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with fall detection details
            fall_duration = self.db.Column(self.db.Float)  # Duration person was fallen
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Grooming standards detection snapshots
        class GroomingSnapshot(self.db.Model):
            __tablename__ = 'grooming_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with violation details
            violation_type = self.db.Column(self.db.String(100))  # 'missing_required', 'prohibited_item'
            violation_item = self.db.Column(self.db.String(100))  # 'uniform', 'name_tag', 'long_hair', etc.
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Dress code monitoring snapshots
        class DressCodeAlert(self.db.Model):
            __tablename__ = 'dresscode_alerts'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            employee_id = self.db.Column(self.db.String(50))  # Employee identifier
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            violations = self.db.Column(self.db.Text)  # Comma-separated list of violations
            uniform_color = self.db.Column(self.db.String(50))  # grey, black, beige, blue, red
            alert_data = self.db.Column(self.db.Text)  # JSON data with detailed violation info
            is_compliant = self.db.Column(self.db.Boolean, default=False)
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # PPE (Personal Protective Equipment) monitoring violations
        class PPEAlert(self.db.Model):
            __tablename__ = 'ppe_alerts'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            employee_id = self.db.Column(self.db.String(50))  # Employee identifier
            snapshot_filename = self.db.Column(self.db.String(255))  # Optional snapshot
            snapshot_path = self.db.Column(self.db.String(500))  # Optional snapshot path
            violations = self.db.Column(self.db.Text, nullable=False)  # Comma-separated list: "No apron", "No gloves", "No hairnet"
            violation_types = self.db.Column(self.db.Text)  # JSON array of violation types
            alert_data = self.db.Column(self.db.Text)  # JSON data with detailed violation info
            is_compliant = self.db.Column(self.db.Boolean, default=False)
            file_size = self.db.Column(self.db.Integer)  # File size in bytes (if snapshot exists)
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Queue monitoring violations
        class QueueViolation(self.db.Model):
            __tablename__ = 'queue_violations'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=True)
            snapshot_path = self.db.Column(self.db.String(500), nullable=True)
            violation_type = self.db.Column(self.db.String(100), nullable=False)  # 'queue_too_long', 'wait_time_exceeded', 'no_counter_staff'
            violation_message = self.db.Column(self.db.Text, nullable=False)
            queue_count = self.db.Column(self.db.Integer, default=0)
            counter_count = self.db.Column(self.db.Integer, default=0)
            wait_time_seconds = self.db.Column(self.db.Float, default=0.0)
            alert_data = self.db.Column(self.db.Text)  # JSON data with detailed violation info
            file_size = self.db.Column(self.db.Integer)  # File size in bytes (if snapshot exists)
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Mopping detection snapshots
        class MoppingSnapshot(self.db.Model):
            __tablename__ = 'mopping_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with detection details
            detection_count = self.db.Column(self.db.Integer, default=0)
            detection_time = self.db.Column(self.db.DateTime)  # When mopping was detected
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Smoking detection snapshots
        class SmokingSnapshot(self.db.Model):
            __tablename__ = 'smoking_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with detection details
            detection_count = self.db.Column(self.db.Integer, default=0)
            detection_time = self.db.Column(self.db.DateTime)  # When smoking was detected
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        class PhoneSnapshot(self.db.Model):
            __tablename__ = 'phone_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with detection details
            detection_count = self.db.Column(self.db.Integer, default=0)
            detection_time = self.db.Column(self.db.DateTime)  # When phone usage was detected
            file_size = self.db.Column(self.db.Integer)  # File size in bytes
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Restricted Area Monitor snapshots
        class RestrictedAreaSnapshot(self.db.Model):
            __tablename__ = 'restricted_area_snapshots'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            snapshot_filename = self.db.Column(self.db.String(255), nullable=False)
            snapshot_path = self.db.Column(self.db.String(500), nullable=False)
            alert_message = self.db.Column(self.db.Text)
            alert_data = self.db.Column(self.db.Text)  # JSON data with violation details
            violation_count = self.db.Column(self.db.Integer, default=0)
            detection_time = self.db.Column(self.db.DateTime)
            file_size = self.db.Column(self.db.Integer)
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Table Service Monitor violations
        class TableServiceViolation(self.db.Model):
            __tablename__ = 'table_service_violations'
            
            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            table_id = self.db.Column(self.db.String(50), nullable=False)
            waiting_time = self.db.Column(self.db.Float, nullable=False)  # Waiting time in seconds (legacy/primary)
            order_wait_time = self.db.Column(self.db.Float, nullable=True)  # Time from customer sits to waiter takes order
            service_wait_time = self.db.Column(self.db.Float, nullable=True)  # Time from waiter leaves to food served
            snapshot_filename = self.db.Column(self.db.String(255), nullable=True)
            snapshot_path = self.db.Column(self.db.String(500), nullable=True)
            alert_data = self.db.Column(self.db.Text)  # JSON data with detailed violation info
            file_size = self.db.Column(self.db.Integer)  # File size in bytes (if snapshot exists)
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)

        # Table Cleanliness violations (unclean tables / slow reset only)
        class TableCleanlinessViolation(self.db.Model):
            __tablename__ = 'table_cleanliness_violations'

            id = self.db.Column(self.db.Integer, primary_key=True)
            channel_id = self.db.Column(self.db.String(50), nullable=False)
            table_id = self.db.Column(self.db.String(50), nullable=False)
            violation_type = self.db.Column(self.db.String(50), nullable=False)  # 'unclean_table' | 'slow_reset'
            snapshot_filename = self.db.Column(self.db.String(255), nullable=True)
            snapshot_path = self.db.Column(self.db.String(500), nullable=True)
            alert_data = self.db.Column(self.db.Text)  # JSON data with detailed violation info
            file_size = self.db.Column(self.db.Integer)  # File size in bytes (if snapshot exists)
            created_at = self.db.Column(self.db.DateTime, default=get_ist_now)
        
        # Store model classes for access
        self.User = User
        self.DailyFootfall = DailyFootfall
        self.HourlyFootfall = HourlyFootfall
        self.QueueAnalytics = QueueAnalytics
        self.ChannelConfig = ChannelConfig
        self.DetectionEvent = DetectionEvent
        self.RTSPChannel = RTSPChannel
        self.AlertGif = AlertGif
        self.HeatmapSnapshot = HeatmapSnapshot
        self.TableServiceViolation = TableServiceViolation
        self.TableCleanlinessViolation = TableCleanlinessViolation
        self.CashSnapshot = CashSnapshot
        self.FallSnapshot = FallSnapshot
        self.GroomingSnapshot = GroomingSnapshot
        self.DressCodeAlert = DressCodeAlert
        self.PPEAlert = PPEAlert
        self.QueueViolation = QueueViolation
        self.MoppingSnapshot = MoppingSnapshot
        self.SmokingSnapshot = SmokingSnapshot
        self.PhoneSnapshot = PhoneSnapshot
        self.RestrictedAreaSnapshot = RestrictedAreaSnapshot

    def add_table_cleanliness_violation(
        self,
        channel_id,
        table_id,
        violation_type,
        snapshot_path=None,
        timestamp=None,
        alert_data=None,
    ):
        """Add a table cleanliness violation (unclean/slow reset) to database"""
        import os
        import json

        try:
            # Get file size if snapshot exists
            file_size = 0
            snapshot_filename = None
            if snapshot_path:
                full_path = os.path.join("static", snapshot_path)
                if os.path.exists(full_path):
                    file_size = os.path.getsize(full_path)
                    snapshot_filename = os.path.basename(snapshot_path)

            # Convert alert_data to JSON string if provided
            alert_data_str = None
            if alert_data:
                if isinstance(alert_data, dict):
                    alert_data_str = json.dumps(alert_data)
                else:
                    alert_data_str = str(alert_data)

            violation = self.TableCleanlinessViolation(
                channel_id=channel_id,
                table_id=table_id,
                violation_type=violation_type,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_data=alert_data_str,
                file_size=file_size,
                created_at=timestamp if timestamp else get_ist_now(),
            )

            self.db.session.add(violation)
            self.db.session.commit()

            logger.info(
                f"Table cleanliness violation saved: type={violation_type}, table={table_id}, channel={channel_id}"
            )
            
            # Send Telegram notification
            violation_msg = "Unclean table" if violation_type == 'unclean_table' else "Slow table reset"
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='table_cleanliness_violation',
                alert_message=f"Table {table_id}: {violation_msg}",
                snapshot_path=snapshot_path,
                alert_data={'table_id': table_id, 'violation_type': violation_type}
            )
            
            return violation.id

        except Exception as e:
            logger.error(f"Error adding table cleanliness violation: {e}")
            self.db.session.rollback()
            return None

    def get_table_cleanliness_violations(self, channel_id=None, table_id=None, limit=50, days=None):
        """Get recent table cleanliness violations"""
        import json
        from datetime import datetime, timedelta

        try:
            query = self.TableCleanlinessViolation.query

            if days is not None and isinstance(days, (int, float)) and days > 0:
                date_threshold = datetime.now() - timedelta(days=days)
                query = query.filter(self.TableCleanlinessViolation.created_at >= date_threshold)

            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            if table_id:
                query = query.filter_by(table_id=table_id)

            violations = query.order_by(self.TableCleanlinessViolation.created_at.desc()).limit(limit).all()

            result = []
            for v in violations:
                alert_data = None
                if v.alert_data:
                    try:
                        alert_data = json.loads(v.alert_data)
                    except Exception:
                        alert_data = v.alert_data

                result.append(
                    {
                        "id": v.id,
                        "channel_id": v.channel_id,
                        "table_id": v.table_id,
                        "violation_type": v.violation_type,
                        "snapshot_filename": v.snapshot_filename,
                        "snapshot_path": v.snapshot_path,
                        "alert_data": alert_data,
                        "file_size": v.file_size,
                        "created_at": v.created_at.isoformat() if v.created_at else None,
                        "timestamp": v.created_at.isoformat() if v.created_at else None,
                    }
                )

            return result
        except Exception as e:
            logger.error(f"Error getting table cleanliness violations: {e}")
            return []

    def delete_table_cleanliness_violation(self, violation_id):
        """Delete a table cleanliness violation"""
        import os
        from pathlib import Path

        try:
            violation = self.TableCleanlinessViolation.query.get(violation_id)
            if not violation:
                logger.warning(f"Table cleanliness violation not found: {violation_id}")
                return False

            # Delete snapshot file if present - handle both relative and absolute paths
            if violation.snapshot_path:
                # Try absolute path first
                if os.path.isabs(violation.snapshot_path):
                    full_path = violation.snapshot_path
                else:
                    # Try relative to static directory
                    full_path = os.path.join("static", violation.snapshot_path)
                    if not os.path.exists(full_path):
                        # Try with just the filename in static/table_service_violations
                        filename = os.path.basename(violation.snapshot_path)
                        full_path = os.path.join("static", "table_service_violations", filename)
                
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                        logger.info(f"Deleted snapshot file: {full_path}")
                    except PermissionError as pe:
                        logger.error(f"Permission denied deleting file {full_path}: {pe}")
                        # Continue with DB deletion even if file deletion fails
                    except Exception as file_err:
                        logger.warning(f"Could not delete file {full_path}: {file_err}")
                        # Continue with DB deletion even if file deletion fails
                else:
                    logger.warning(f"Snapshot file not found: {full_path} (violation_id: {violation_id})")

            self.db.session.delete(violation)
            self.db.session.commit()
            logger.info(f"Deleted table cleanliness violation: {violation_id}")
            return True

        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting table cleanliness violation {violation_id}: {e}", exc_info=True)
            return False
    
    def update_footfall_count(self, channel_id, direction, count=1):
        """Update footfall counts for People Counter"""
        now = datetime.now()
        today = now.date()
        current_hour = now.hour
        
        # Update daily count
        daily_record = self.DailyFootfall.query.filter_by(
            channel_id=channel_id, 
            report_date=today
        ).first()
        
        if not daily_record:
            daily_record = self.DailyFootfall(
                channel_id=channel_id,
                report_date=today,
                in_count=0,
                out_count=0
            )
            self.db.session.add(daily_record)
        
        # Ensure counts are not None
        if daily_record.in_count is None:
            daily_record.in_count = 0
        if daily_record.out_count is None:
            daily_record.out_count = 0
        
        if direction == 'in':
            daily_record.in_count += count
        elif direction == 'out':
            daily_record.out_count += count
        
        # Update hourly count
        hourly_record = self.HourlyFootfall.query.filter_by(
            channel_id=channel_id,
            report_date=today,
            hour=current_hour
        ).first()
        
        if not hourly_record:
            hourly_record = self.HourlyFootfall(
                channel_id=channel_id,
                report_date=today,
                hour=current_hour,
                in_count=0,
                out_count=0
            )
            self.db.session.add(hourly_record)
        
        # Ensure counts are not None
        if hourly_record.in_count is None:
            hourly_record.in_count = 0
        if hourly_record.out_count is None:
            hourly_record.out_count = 0
        
        if direction == 'in':
            hourly_record.in_count += count
        elif direction == 'out':
            hourly_record.out_count += count
        
        try:
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def get_today_footfall_count(self, channel_id=None):
        """Get today's footfall count from database"""
        try:
            today = datetime.now().date()
            
            if channel_id:
                # Get count for specific channel
                daily_record = self.DailyFootfall.query.filter_by(
                    channel_id=channel_id,
                    report_date=today
                ).first()
                
                if daily_record:
                    return {
                        'channel_id': channel_id,
                        'in_count': daily_record.in_count or 0,
                        'out_count': daily_record.out_count or 0,
                        'date': today.isoformat()
                    }
                else:
                    return {
                        'channel_id': channel_id,
                        'in_count': 0,
                        'out_count': 0,
                        'date': today.isoformat()
                    }
            else:
                # Get count for all channels
                daily_records = self.DailyFootfall.query.filter_by(
                    report_date=today
                ).all()
                
                total_in = 0
                total_out = 0
                channels_data = []
                
                for record in daily_records:
                    total_in += record.in_count or 0
                    total_out += record.out_count or 0
                    channels_data.append({
                        'channel_id': record.channel_id,
                        'in_count': record.in_count or 0,
                        'out_count': record.out_count or 0
                    })
                
                return {
                    'total_in': total_in,
                    'total_out': total_out,
                    'date': today.isoformat(),
                    'channels': channels_data
                }
        except Exception as e:
            logger.error(f"Error getting today's footfall count: {e}")
            if channel_id:
                return {'channel_id': channel_id, 'in_count': 0, 'out_count': 0, 'date': today.isoformat()}
            else:
                return {'total_in': 0, 'total_out': 0, 'date': today.isoformat(), 'channels': []}
    
    def get_people_counter_peak_hour(self, channel_id=None):
        """Get peak hour analytics for people counter"""
        try:
            from sqlalchemy import func
            today = datetime.now().date()
            
            if channel_id:
                # Get peak hour for specific channel
                peak_hour_data = self.db.session.query(
                    self.HourlyFootfall.hour,
                    func.sum(self.HourlyFootfall.in_count + self.HourlyFootfall.out_count).label('total_traffic')
                ).filter_by(
                    channel_id=channel_id,
                    report_date=today
                ).group_by(
                    self.HourlyFootfall.hour
                ).order_by(
                    func.sum(self.HourlyFootfall.in_count + self.HourlyFootfall.out_count).desc()
                ).first()
                
                if peak_hour_data and peak_hour_data.total_traffic > 0:
                    hour_24 = peak_hour_data.hour
                    # Convert to 12-hour format
                    if hour_24 == 0:
                        hour_12 = "12 AM"
                    elif hour_24 < 12:
                        hour_12 = f"{hour_24} AM"
                    elif hour_24 == 12:
                        hour_12 = "12 PM"
                    else:
                        hour_12 = f"{hour_24 - 12} PM"
                    
                    return {
                        'peak_hour': hour_12,
                        'peak_hour_24': hour_24,
                        'traffic_count': int(peak_hour_data.total_traffic)
                    }
                else:
                    return {'peak_hour': 'N/A', 'peak_hour_24': None, 'traffic_count': 0}
            else:
                # Get peak hour across all channels
                peak_hour_data = self.db.session.query(
                    self.HourlyFootfall.hour,
                    func.sum(self.HourlyFootfall.in_count + self.HourlyFootfall.out_count).label('total_traffic')
                ).filter_by(
                    report_date=today
                ).group_by(
                    self.HourlyFootfall.hour
                ).order_by(
                    func.sum(self.HourlyFootfall.in_count + self.HourlyFootfall.out_count).desc()
                ).first()
                
                if peak_hour_data and peak_hour_data.total_traffic > 0:
                    hour_24 = peak_hour_data.hour
                    # Convert to 12-hour format
                    if hour_24 == 0:
                        hour_12 = "12 AM"
                    elif hour_24 < 12:
                        hour_12 = f"{hour_24} AM"
                    elif hour_24 == 12:
                        hour_12 = "12 PM"
                    else:
                        hour_12 = f"{hour_24 - 12} PM"
                    
                    return {
                        'peak_hour': hour_12,
                        'peak_hour_24': hour_24,
                        'traffic_count': int(peak_hour_data.total_traffic)
                    }
                else:
                    return {'peak_hour': 'N/A', 'peak_hour_24': None, 'traffic_count': 0}
                    
        except Exception as e:
            logger.error(f"Error getting peak hour for people counter: {e}")
            return {'peak_hour': 'N/A', 'peak_hour_24': None, 'traffic_count': 0}
    
    def get_fall_detection_analytics(self, days=7):
        """Get comprehensive fall detection analytics"""
        try:
            from sqlalchemy import func
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            today = end_date.date()
            
            # Total fall incidents in the period
            total_falls = self.FallSnapshot.query.filter(
                self.FallSnapshot.created_at >= start_date
            ).count()
            
            # Today's fall incidents
            today_falls = self.FallSnapshot.query.filter(
                func.date(self.FallSnapshot.created_at) == today
            ).count()
            
            # Average fall duration
            avg_duration_result = self.db.session.query(
                func.avg(self.FallSnapshot.fall_duration)
            ).filter(
                self.FallSnapshot.created_at >= start_date,
                self.FallSnapshot.fall_duration.isnot(None)
            ).scalar()
            
            avg_fall_duration = round(float(avg_duration_result), 1) if avg_duration_result else 0
            
            # Longest fall duration
            max_duration_result = self.db.session.query(
                func.max(self.FallSnapshot.fall_duration)
            ).filter(
                self.FallSnapshot.created_at >= start_date,
                self.FallSnapshot.fall_duration.isnot(None)
            ).scalar()
            
            max_fall_duration = round(float(max_duration_result), 1) if max_duration_result else 0
            
            # Channel-wise fall count
            channel_stats = self.db.session.query(
                self.FallSnapshot.channel_id,
                func.count(self.FallSnapshot.id).label('fall_count'),
                func.avg(self.FallSnapshot.fall_duration).label('avg_duration')
            ).filter(
                self.FallSnapshot.created_at >= start_date
            ).group_by(
                self.FallSnapshot.channel_id
            ).order_by(
                func.count(self.FallSnapshot.id).desc()
            ).all()
            
            channels_data = []
            most_incidents_channel = None
            
            for channel_id, fall_count, avg_dur in channel_stats:
                channel_info = {
                    'channel_id': channel_id,
                    'fall_count': fall_count,
                    'avg_duration': round(float(avg_dur), 1) if avg_dur else 0,
                    'severity': 'High' if fall_count > 5 else ('Medium' if fall_count > 2 else 'Low')
                }
                channels_data.append(channel_info)
                
                if most_incidents_channel is None:
                    most_incidents_channel = channel_info
            
            # Daily trend
            daily_falls = self.db.session.query(
                func.date(self.FallSnapshot.created_at).label('date'),
                func.count(self.FallSnapshot.id).label('count')
            ).filter(
                self.FallSnapshot.created_at >= start_date
            ).group_by(
                func.date(self.FallSnapshot.created_at)
            ).order_by(
                func.date(self.FallSnapshot.created_at)
            ).all()
            
            daily_trend = []
            for day, count in daily_falls:
                try:
                    # Handle both date objects and strings
                    if isinstance(day, str):
                        date_str = day
                    else:
                        date_str = day.isoformat() if hasattr(day, 'isoformat') else str(day)
                    daily_trend.append({
                        'date': date_str,
                        'count': count
                    })
                except Exception as e:
                    logger.warning(f"Error formatting fall detection date: {e}, type: {type(day)}, value: {day}")
                    daily_trend.append({
                        'date': str(day) if day else None,
                        'count': count
                    })
            
            # Peak hour analysis
            hourly_falls = self.db.session.query(
                func.extract('hour', self.FallSnapshot.created_at).label('hour'),
                func.count(self.FallSnapshot.id).label('count')
            ).filter(
                self.FallSnapshot.created_at >= start_date
            ).group_by(
                func.extract('hour', self.FallSnapshot.created_at)
            ).order_by(
                func.count(self.FallSnapshot.id).desc()
            ).first()
            
            peak_hour = 'N/A'
            peak_hour_count = 0
            if hourly_falls:
                hour_24 = int(hourly_falls.hour)
                peak_hour_count = hourly_falls.count
                
                # Convert to 12-hour format
                if hour_24 == 0:
                    peak_hour = "12 AM"
                elif hour_24 < 12:
                    peak_hour = f"{hour_24} AM"
                elif hour_24 == 12:
                    peak_hour = "12 PM"
                else:
                    peak_hour = f"{hour_24 - 12} PM"
            
            # Response time categories (based on duration)
            quick_response = self.FallSnapshot.query.filter(
                self.FallSnapshot.created_at >= start_date,
                self.FallSnapshot.fall_duration.isnot(None),
                self.FallSnapshot.fall_duration < 30  # Less than 30 seconds
            ).count()
            
            delayed_response = self.FallSnapshot.query.filter(
                self.FallSnapshot.created_at >= start_date,
                self.FallSnapshot.fall_duration.isnot(None),
                self.FallSnapshot.fall_duration >= 30,
                self.FallSnapshot.fall_duration < 60  # 30-60 seconds
            ).count()
            
            critical_response = self.FallSnapshot.query.filter(
                self.FallSnapshot.created_at >= start_date,
                self.FallSnapshot.fall_duration.isnot(None),
                self.FallSnapshot.fall_duration >= 60  # More than 60 seconds
            ).count()
            
            return {
                'total_falls': total_falls,
                'today_falls': today_falls,
                'avg_fall_duration': avg_fall_duration,
                'max_fall_duration': max_fall_duration,
                'peak_hour': peak_hour,
                'peak_hour_count': peak_hour_count,
                'channels': channels_data,
                'most_incidents_channel': most_incidents_channel,
                'daily_trend': daily_trend,
                'response_categories': {
                    'quick': quick_response,  # < 30s
                    'delayed': delayed_response,  # 30-60s
                    'critical': critical_response  # > 60s
                },
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting fall detection analytics: {e}")
            return {
                'total_falls': 0,
                'today_falls': 0,
                'avg_fall_duration': 0,
                'max_fall_duration': 0,
                'peak_hour': 'N/A',
                'peak_hour_count': 0,
                'channels': [],
                'most_incidents_channel': None,
                'daily_trend': [],
                'response_categories': {'quick': 0, 'delayed': 0, 'critical': 0},
                'period_days': days
            }
    
    def log_queue_analytics(self, channel_id, queue_count, counter_count, alert_triggered=False, alert_message=None):
        """Log queue analytics data"""
        analytics = self.QueueAnalytics(
            channel_id=channel_id,
            queue_count=queue_count,
            counter_count=counter_count,
            alert_triggered=alert_triggered,
            alert_message=alert_message
        )
        
        self.db.session.add(analytics)
        try:
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def save_channel_config(self, channel_id, app_name, config_type, config_data):
        """Save channel configuration"""
        config = self.ChannelConfig.query.filter_by(
            channel_id=channel_id,
            app_name=app_name,
            config_type=config_type
        ).first()
        
        if not config:
            config = self.ChannelConfig(
                channel_id=channel_id,
                app_name=app_name,
                config_type=config_type
            )
            self.db.session.add(config)
        
        config.config_data = json.dumps(config_data)
        config.updated_at = get_ist_now()
        
        try:
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def get_channel_config(self, channel_id, app_name, config_type):
        """Get channel configuration"""
        config = self.ChannelConfig.query.filter_by(
            channel_id=channel_id,
            app_name=app_name,
            config_type=config_type
        ).first()
        
        if config and config.config_data:
            return json.loads(config.config_data)
        return None
    
    def log_detection_event(self, channel_id, app_name, event_type, event_data, confidence=None):
        """Log detection event"""
        event = self.DetectionEvent(
            channel_id=channel_id,
            app_name=app_name,
            event_type=event_type,
            event_data=json.dumps(event_data),
            confidence=confidence
        )
        
        self.db.session.add(event)
        try:
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def get_footfall_report(self, channel_id, period='7days'):
        """Get footfall report for specified period"""
        end_date = datetime.now().date()
        
        if period == '24hours':
            start_date = end_date
            # Get hourly data for today
            hourly_data = self.HourlyFootfall.query.filter_by(
                channel_id=channel_id,
                report_date=start_date
            ).order_by(self.HourlyFootfall.hour).all()
            
            return {
                'period': period,
                'data': [
                    {
                        'hour': h.hour,
                        'in_count': h.in_count,
                        'out_count': h.out_count,
                        'total': h.in_count + h.out_count
                    } for h in hourly_data
                ]
            }
        
        elif period == '7days':
            start_date = end_date - timedelta(days=6)
        elif period == '30days':
            start_date = end_date - timedelta(days=29)
        else:
            start_date = end_date - timedelta(days=6)
        
        # Get daily data
        daily_data = self.DailyFootfall.query.filter(
            self.DailyFootfall.channel_id == channel_id,
            self.DailyFootfall.report_date >= start_date,
            self.DailyFootfall.report_date <= end_date
        ).order_by(self.DailyFootfall.report_date).all()
        
        return {
            'period': period,
            'data': [
                {
                    'date': d.report_date.isoformat(),
                    'in_count': d.in_count,
                    'out_count': d.out_count,
                    'total': d.in_count + d.out_count
                } for d in daily_data
            ]
        }
    
    def get_queue_report(self, channel_id, start_date=None, end_date=None):
        """Get queue analytics report"""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).date()
        if not end_date:
            end_date = datetime.now().date()
        
        # Convert to datetime for filtering
        start_datetime = datetime.combine(start_date, datetime.min.time()) if isinstance(start_date, str) else datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time()) if isinstance(end_date, str) else datetime.combine(end_date, datetime.max.time())
        
        analytics_data = self.QueueAnalytics.query.filter(
            self.QueueAnalytics.channel_id == channel_id,
            self.QueueAnalytics.timestamp >= start_datetime,
            self.QueueAnalytics.timestamp <= end_datetime
        ).order_by(self.QueueAnalytics.timestamp).all()
        
        return {
            'start_date': start_date.isoformat() if hasattr(start_date, 'isoformat') else start_date,
            'end_date': end_date.isoformat() if hasattr(end_date, 'isoformat') else end_date,
            'data': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'queue_count': a.queue_count,
                    'counter_count': a.counter_count,
                    'alert_triggered': a.alert_triggered,
                    'alert_message': a.alert_message
                } for a in analytics_data
            ]
        }
    
    def save_rtsp_channel(self, channel_id, name, rtsp_url, description=None):
        """Save RTSP channel configuration"""
        try:
            # Check if channel already exists
            existing = self.RTSPChannel.query.filter_by(channel_id=channel_id).first()
            
            if existing:
                # Update existing
                existing.name = name
                existing.rtsp_url = rtsp_url
                existing.description = description
                existing.updated_at = get_ist_now()
            else:
                # Create new
                channel = self.RTSPChannel(
                    channel_id=channel_id,
                    name=name,
                    rtsp_url=rtsp_url,
                    description=description
                )
                self.db.session.add(channel)
            
            self.db.session.commit()
            return True
            
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def get_rtsp_channels(self):
        """Get all active RTSP channels"""
        channels = self.RTSPChannel.query.filter_by(is_active=True).all()
        return [
            {
                'channel_id': c.channel_id,
                'name': c.name,
                'rtsp_url': c.rtsp_url,
                'description': c.description,
                'created_at': c.created_at.isoformat(),
                'updated_at': c.updated_at.isoformat()
            } for c in channels
        ]
    
    def get_rtsp_channel(self, channel_id):
        """Get specific RTSP channel"""
        channel = self.RTSPChannel.query.filter_by(
            channel_id=channel_id, 
            is_active=True
        ).first()
        
        if channel:
            return {
                'channel_id': channel.channel_id,
                'name': channel.name,
                'rtsp_url': channel.rtsp_url,
                'description': channel.description,
                'created_at': channel.created_at.isoformat(),
                'updated_at': channel.updated_at.isoformat()
            }
        return None
    
    def remove_rtsp_channel(self, channel_id):
        """Remove RTSP channel (soft delete)"""
        try:
            channel = self.RTSPChannel.query.filter_by(channel_id=channel_id).first()
            if channel:
                channel.is_active = False
                channel.updated_at = get_ist_now()
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            raise e
    
    def save_alert_gif(self, channel_id, alert_type, gif_info, alert_message=None, alert_data=None):
        """Save alert GIF information to database"""
        try:
            import os
            
            # Get file size if path exists
            file_size = 0
            gif_path = gif_info.get('gif_path', '')
            if os.path.exists(gif_path):
                file_size = os.path.getsize(gif_path)
            
            alert_gif = self.AlertGif(
                channel_id=channel_id,
                alert_type=alert_type,
                gif_filename=gif_info.get('gif_filename', ''),
                gif_path=gif_path,
                alert_message=alert_message,
                alert_data=json.dumps(alert_data) if alert_data else None,
                frame_count=gif_info.get('frame_count', 0),
                file_size=file_size,
                duration_seconds=gif_info.get('duration', 0.0)
            )
            
            self.db.session.add(alert_gif)
            self.db.session.commit()
            
            # Send Telegram notification
            try:
                from modules.telegram_notifier import get_telegram_notifier
                notifier = get_telegram_notifier()
                # Try to resolve full path for GIF
                full_gif_path = gif_path
                if gif_path and not os.path.isabs(gif_path):
                    # Try relative to static directory
                    static_path = os.path.join("static", gif_path)
                    if os.path.exists(static_path):
                        full_gif_path = static_path
                    elif os.path.exists(gif_path):
                        full_gif_path = gif_path
                    else:
                        full_gif_path = None
                
                if full_gif_path:
                    notifier.send_alert(
                        channel_id=channel_id,
                        alert_type=alert_type,
                        alert_message=alert_message or f"Alert from {channel_id}",
                        image_path=full_gif_path,
                        alert_data=alert_data
                    )
                else:
                    # Send text-only if GIF path not found
                    notifier.send_alert(
                        channel_id=channel_id,
                        alert_type=alert_type,
                        alert_message=alert_message or f"Alert from {channel_id}",
                        alert_data=alert_data
                    )
            except Exception as tg_error:
                logger.warning(f"Failed to send Telegram notification: {tg_error}")
            
            return alert_gif.id
            
        except Exception as e:
            self.db.session.rollback()
            # Check if it's a unique constraint violation (sequence out of sync)
            error_str = str(e)
            if 'UniqueViolation' in error_str and 'alert_gifs_pkey' in error_str:
                logger.warning(f"Sequence out of sync for alert_gifs, attempting to fix...")
                try:
                    # Get max ID and reset sequence
                    max_id_result = self.db.session.execute(
                        self.db.text("SELECT COALESCE(MAX(id), 0) FROM alert_gifs")
                    ).scalar()
                    new_seq = max_id_result + 1
                    self.db.session.execute(
                        self.db.text(f"SELECT setval('alert_gifs_id_seq', {new_seq}, false)")
                    )
                    self.db.session.commit()
                    logger.info(f"Fixed alert_gifs sequence to {new_seq}, retrying save...")
                    # Retry the save
                    return self.save_alert_gif(channel_id, alert_type, gif_info, alert_message, alert_data)
                except Exception as fix_error:
                    logger.error(f"Failed to fix sequence: {fix_error}")
            logger.error(f"Error saving alert GIF: {e}")
            raise e
    
    def log_alert(self, channel_id, alert_type, alert_message, alert_data=None):
        """Log an alert to the database (creates AlertGif entry without GIF file)"""
        try:
            alert_gif = self.AlertGif(
                channel_id=channel_id,
                alert_type=alert_type,
                gif_filename='',  # No GIF file yet
                gif_path='',  # No GIF file yet
                alert_message=alert_message,
                alert_data=json.dumps(alert_data) if alert_data else None,
                frame_count=0,
                file_size=0,
                duration_seconds=0.0
            )
            
            self.db.session.add(alert_gif)
            self.db.session.commit()
            
            # Send Telegram notification
            try:
                from modules.telegram_notifier import get_telegram_notifier
                notifier = get_telegram_notifier()
                notifier.send_alert(
                    channel_id=channel_id,
                    alert_type=alert_type,
                    alert_message=alert_message,
                    alert_data=alert_data
                )
            except Exception as tg_error:
                logger.warning(f"Failed to send Telegram notification: {tg_error}")
            
            return alert_gif.id
            
        except Exception as e:
            self.db.session.rollback()
            # Check if it's a unique constraint violation (sequence out of sync)
            error_str = str(e)
            if 'UniqueViolation' in error_str and 'alert_gifs_pkey' in error_str:
                logger.warning(f"Sequence out of sync for alert_gifs, attempting to fix...")
                try:
                    # Get max ID and reset sequence
                    max_id_result = self.db.session.execute(
                        self.db.text("SELECT COALESCE(MAX(id), 0) FROM alert_gifs")
                    ).scalar()
                    new_seq = max_id_result + 1
                    self.db.session.execute(
                        self.db.text(f"SELECT setval('alert_gifs_id_seq', {new_seq}, false)")
                    )
                    self.db.session.commit()
                    logger.info(f"Fixed alert_gifs sequence to {new_seq}, retrying save...")
                    # Retry the save
                    return self.log_alert(channel_id, alert_type, alert_message, alert_data)
                except Exception as fix_error:
                    logger.error(f"Failed to fix sequence: {fix_error}")
            logger.error(f"Error logging alert: {e}")
            raise e
    
    def get_alert_count(self, alert_type=None, days=7, channel_id=None):
        """Get count of alerts within specified days"""
        try:
            from datetime import datetime, timedelta
            
            # Build query
            query = self.AlertGif.query
            
            # Only apply date filter if days is provided and is a number
            if days is not None and isinstance(days, (int, float)) and days > 0:
                date_threshold = datetime.now() - timedelta(days=days)
                query = query.filter(self.AlertGif.created_at >= date_threshold)
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            if alert_type:
                query = query.filter_by(alert_type=alert_type)
            
            return query.count()
            
        except Exception as e:
            logger.error(f"Error getting alert count: {e}")
            return 0
    
    def get_bag_detection_analytics(self, days=7):
        """Get comprehensive bag detection analytics"""
        try:
            from datetime import datetime, timedelta
            from sqlalchemy import func, cast, Date
            
            date_threshold = datetime.now() - timedelta(days=days)
            
            # Get total alerts by channel
            channel_alerts = self.db.session.query(
                self.AlertGif.channel_id,
                func.count(self.AlertGif.id).label('alert_count'),
                func.max(self.AlertGif.created_at).label('last_alert')
            ).filter(
                self.AlertGif.alert_type == 'bag_unattended',
                self.AlertGif.created_at >= date_threshold
            ).group_by(
                self.AlertGif.channel_id
            ).all()
            
            # Get daily alert trends
            daily_alerts = self.db.session.query(
                cast(self.AlertGif.created_at, Date).label('alert_date'),
                func.count(self.AlertGif.id).label('count')
            ).filter(
                self.AlertGif.alert_type == 'bag_unattended',
                self.AlertGif.created_at >= date_threshold
            ).group_by(
                cast(self.AlertGif.created_at, Date)
            ).order_by(
                cast(self.AlertGif.created_at, Date)
            ).all()
            
            # Format results
            channels_data = []
            for ch in channel_alerts:
                channels_data.append({
                    'channel_id': ch.channel_id,
                    'alert_count': ch.alert_count,
                    'last_alert': ch.last_alert.isoformat() if ch.last_alert else None
                })
            
            daily_data = []
            for day in daily_alerts:
                daily_data.append({
                    'date': day.alert_date.isoformat(),
                    'count': day.count
                })
            
            # Calculate totals
            total_alerts = sum(c['alert_count'] for c in channels_data)
            
            # Identify most risky channel
            most_risky = max(channels_data, key=lambda x: x['alert_count']) if channels_data else None
            
            return {
                'total_alerts': total_alerts,
                'channels': channels_data,
                'daily_trend': daily_data,
                'most_risky_channel': most_risky,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting bag detection analytics: {e}")
            return {
                'total_alerts': 0,
                'channels': [],
                'daily_trend': [],
                'most_risky_channel': None,
                'period_days': days
            }
    
    def get_alert_gifs(self, channel_id=None, alert_type=None, limit=50, days=None):
        """Get alert GIFs from database"""
        try:
            from datetime import datetime, timedelta
            
            query = self.AlertGif.query
            
            # Apply date filter if days is provided
            if days is not None and isinstance(days, (int, float)) and days > 0:
                date_threshold = datetime.now() - timedelta(days=days)
                query = query.filter(self.AlertGif.created_at >= date_threshold)
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            if alert_type:
                query = query.filter_by(alert_type=alert_type)
            
            # Order by creation time (newest first)
            alert_gifs = query.order_by(self.AlertGif.created_at.desc()).limit(limit).all()
            
            return [
                {
                    'id': gif.id,
                    'channel_id': gif.channel_id,
                    'alert_type': gif.alert_type,
                    'gif_filename': gif.gif_filename,
                    'gif_path': gif.gif_path,
                    'alert_message': gif.alert_message,
                    'alert_data': json.loads(gif.alert_data) if gif.alert_data else None,
                    'frame_count': gif.frame_count,
                    'file_size': gif.file_size,
                    'file_size_mb': round(gif.file_size / (1024 * 1024), 2) if gif.file_size else 0,
                    'duration_seconds': gif.duration_seconds,
                    'created_at': gif.created_at.isoformat(),
                    'gif_url': f'/static/alerts/{gif.gif_filename}'
                } for gif in alert_gifs
            ]
            
        except Exception as e:
            logger.error(f"Error getting alert GIFs: {e}")
            return []
    
    def delete_alert_gif(self, gif_id):
        """Delete alert GIF from database"""
        try:
            import os
            from pathlib import Path
            
            alert_gif = self.AlertGif.query.get(gif_id)
            if alert_gif:
                # Delete the actual file - handle both relative and absolute paths
                if alert_gif.gif_path:
                    # Try absolute path first
                    if os.path.isabs(alert_gif.gif_path):
                        file_path = alert_gif.gif_path
                    else:
                        # Try relative to current directory
                        file_path = os.path.join(os.getcwd(), alert_gif.gif_path)
                        if not os.path.exists(file_path):
                            # Try relative to static/alerts directory
                            filename = os.path.basename(alert_gif.gif_path)
                            file_path = os.path.join("static", "alerts", filename)
                    
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted alert GIF file: {file_path}")
                        except PermissionError as pe:
                            logger.error(f"Permission denied deleting file {file_path}: {pe}")
                            # Continue with DB deletion even if file deletion fails
                        except Exception as file_err:
                            logger.warning(f"Could not delete file {file_path}: {file_err}")
                            # Continue with DB deletion even if file deletion fails
                    else:
                        logger.warning(f"Alert GIF file not found: {file_path} (gif_id: {gif_id})")
                
                self.db.session.delete(alert_gif)
                self.db.session.commit()
                logger.info(f"Deleted alert GIF record: {gif_id}")
                return True
            logger.warning(f"Alert GIF not found: {gif_id}")
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting alert GIF {gif_id}: {e}", exc_info=True)
            raise e
    
    def cleanup_old_alert_gifs(self, max_age_days=30, alert_type='all'):
        """Clean up old alert GIF records and files"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = get_ist_now() - timedelta(days=max_age_days)
            
            # Build query based on alert type
            query = self.AlertGif.query.filter(
                self.AlertGif.created_at < cutoff_date
            )
            
            # Filter by alert type if specified
            if alert_type != 'all':
                query = query.filter(self.AlertGif.alert_type == alert_type)
            
            old_gifs = query.all()
            
            removed_count = 0
            for gif in old_gifs:
                # Delete file if exists
                if os.path.exists(gif.gif_path):
                    try:
                        os.remove(gif.gif_path)
                    except:
                        pass
                
                # Delete database record
                self.db.session.delete(gif)
                removed_count += 1
            
            self.db.session.commit()
            return removed_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error cleaning up old alert GIFs: {e}")
            raise e
    
    def save_heatmap_snapshot(self, channel_id, snapshot_path, hotspot_count, hotspots):
        """Save heatmap snapshot to database"""
        try:
            import os
            
            snapshot_filename = os.path.basename(snapshot_path)
            file_size = os.path.getsize(snapshot_path) if os.path.exists(snapshot_path) else 0
            
            snapshot = self.HeatmapSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                hotspot_count=hotspot_count,
                hotspots_data=json.dumps(hotspots)
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            return snapshot.id
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error saving heatmap snapshot: {e}")
            raise e
    
    def get_heatmap_snapshots(self, channel_id=None, limit=20):
        """Get heatmap snapshots from database"""
        try:
            query = self.HeatmapSnapshot.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.order_by(
                self.HeatmapSnapshot.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': snap.id,
                    'channel_id': snap.channel_id,
                    'snapshot_filename': snap.snapshot_filename,
                    'snapshot_path': snap.snapshot_path,
                    'hotspot_count': snap.hotspot_count,
                    'hotspots': json.loads(snap.hotspots_data) if snap.hotspots_data else [],
                    'created_at': snap.created_at.isoformat(),
                    'snapshot_url': f'/static/heatmaps/{snap.snapshot_filename}'
                } for snap in snapshots
            ]
            
        except Exception as e:
            logger.error(f"Error getting heatmap snapshots: {e}")
            return []
    
    def delete_heatmap_snapshot(self, snapshot_id):
        """Delete heatmap snapshot from database"""
        try:
            snapshot = self.HeatmapSnapshot.query.get(snapshot_id)
            if snapshot:
                # Delete the actual file
                import os
                if os.path.exists(snapshot.snapshot_path):
                    os.remove(snapshot.snapshot_path)
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            raise e
    def get_heatmap_analytics(self, days=7):
        """Get comprehensive heatmap analytics across all channels"""
        try:
            from datetime import timedelta
            from sqlalchemy import func
            
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            # Get channel-wise statistics
            channel_stats = self.db.session.query(
                self.HeatmapSnapshot.channel_id,
                func.count(self.HeatmapSnapshot.id).label('snapshot_count'),
                func.sum(self.HeatmapSnapshot.hotspot_count).label('total_hotspots'),
                func.avg(self.HeatmapSnapshot.hotspot_count).label('avg_hotspots'),
                func.max(self.HeatmapSnapshot.hotspot_count).label('max_hotspots'),
                func.max(self.HeatmapSnapshot.created_at).label('last_snapshot')
            ).filter(
                self.HeatmapSnapshot.created_at >= cutoff_date
            ).group_by(
                self.HeatmapSnapshot.channel_id
            ).all()
            
            # Format channel data
            channels_data = []
            for stat in channel_stats:
                channels_data.append({
                    'channel_id': stat.channel_id,
                    'snapshot_count': stat.snapshot_count,
                    'total_hotspots': int(stat.total_hotspots or 0),
                    'avg_hotspots': round(float(stat.avg_hotspots or 0), 2),
                    'max_hotspots': int(stat.max_hotspots or 0),
                    'last_snapshot': stat.last_snapshot.isoformat() if stat.last_snapshot else None
                })
            
            # Sort by total hotspots (most crowded first)
            channels_data.sort(key=lambda x: x['total_hotspots'], reverse=True)
            
            # Get overall statistics
            total_snapshots = sum(c['snapshot_count'] for c in channels_data)
            total_hotspots = sum(c['total_hotspots'] for c in channels_data)
            
            # Identify most crowded channel
            most_crowded = channels_data[0] if channels_data else None
            
            return {
                'total_snapshots': total_snapshots,
                'total_hotspots': total_hotspots,
                'channels': channels_data,
                'most_crowded_channel': most_crowded,
                'total_channels': len(channels_data),
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting heatmap analytics: {e}")
            return {
                'total_snapshots': 0,
                'total_hotspots': 0,
                'channels': [],
                'most_crowded_channel': None,
                'total_channels': 0,
                'period_days': days
            }
    
    def clear_old_heatmap_snapshots(self, days=7):
        """Clear old heatmap snapshots older than specified days"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            old_snapshots = self.HeatmapSnapshot.query.filter(
                self.HeatmapSnapshot.created_at < cutoff_date
            ).all()
            
            deleted_count = 0
            for snapshot in old_snapshots:
                # Delete file if exists
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                # Delete database record
                self.db.session.delete(snapshot)
                deleted_count += 1
            
            self.db.session.commit()
            return deleted_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old heatmap snapshots: {e}")
            raise e
    
    def save_cash_snapshot(self, channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, file_size, detection_count):
        """Save cash detection snapshot to database"""
        try:
            snapshot = self.CashSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_message=alert_message,
                alert_data=json.dumps(alert_data),
                detection_count=detection_count,
                file_size=file_size
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            logger.info(f"Cash snapshot saved to database: ID {snapshot.id}")
            
            # Send Telegram notification
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='cash_alert',
                alert_message=alert_message or f"Cash detected: {detection_count} instance(s)",
                snapshot_path=snapshot_path,
                alert_data=alert_data
            )
            
            return snapshot.id
            
        except Exception as e:
            self.db.session.rollback()
            # Check if it's a unique constraint violation (sequence out of sync)
            error_str = str(e)
            if 'UniqueViolation' in error_str and 'cash_snapshots_pkey' in error_str:
                logger.warning(f"Sequence out of sync for cash_snapshots, attempting to fix...")
                try:
                    # Get max ID and reset sequence
                    max_id_result = self.db.session.execute(
                        self.db.text("SELECT COALESCE(MAX(id), 0) FROM cash_snapshots")
                    ).scalar()
                    new_seq = max_id_result + 1
                    self.db.session.execute(
                        self.db.text(f"SELECT setval('cash_snapshots_id_seq', {new_seq}, false)")
                    )
                    self.db.session.commit()
                    logger.info(f"Fixed cash_snapshots sequence to {new_seq}, retrying save...")
                    # Retry the save
                    return self.save_cash_snapshot(channel_id, snapshot_filename, snapshot_path, alert_message, alert_data, file_size, detection_count)
                except Exception as fix_error:
                    logger.error(f"Failed to fix sequence: {fix_error}")
            logger.error(f"Error saving cash snapshot: {e}")
            raise e
    
    def get_cash_detection_analytics(self, days=7):
        """Get comprehensive cash detection analytics"""
        try:
            from datetime import datetime, timedelta
            from sqlalchemy import func, cast, Date
            
            date_threshold = datetime.now() - timedelta(days=days)
            
            # Get total snapshots by channel
            channel_stats = self.db.session.query(
                self.CashSnapshot.channel_id,
                func.count(self.CashSnapshot.id).label('snapshot_count'),
                func.sum(self.CashSnapshot.detection_count).label('total_detections'),
                func.avg(self.CashSnapshot.detection_count).label('avg_detections'),
                func.max(self.CashSnapshot.created_at).label('last_detection')
            ).filter(
                self.CashSnapshot.created_at >= date_threshold
            ).group_by(
                self.CashSnapshot.channel_id
            ).all()
            
            # Get hourly distribution (peak hours)
            hourly_stats = self.db.session.query(
                func.extract('hour', self.CashSnapshot.created_at).label('hour'),
                func.count(self.CashSnapshot.id).label('count')
            ).filter(
                self.CashSnapshot.created_at >= date_threshold
            ).group_by(
                func.extract('hour', self.CashSnapshot.created_at)
            ).order_by(
                func.count(self.CashSnapshot.id).desc()
            ).all()
            
            # Get daily trend
            daily_stats = self.db.session.query(
                cast(self.CashSnapshot.created_at, Date).label('snapshot_date'),
                func.count(self.CashSnapshot.id).label('count'),
                func.sum(self.CashSnapshot.detection_count).label('detections')
            ).filter(
                self.CashSnapshot.created_at >= date_threshold
            ).group_by(
                cast(self.CashSnapshot.created_at, Date)
            ).order_by(
                cast(self.CashSnapshot.created_at, Date)
            ).all()
            
            # Format results
            channels_data = []
            for ch in channel_stats:
                # Safely handle last_detection - could be datetime, string, or None
                last_detection_str = None
                if ch.last_detection:
                    try:
                        if isinstance(ch.last_detection, str):
                            last_detection_str = ch.last_detection
                        else:
                            # It's a datetime object
                            last_detection_str = ch.last_detection.isoformat()
                    except Exception as e:
                        logger.warning(f"Error formatting last_detection: {e}, type: {type(ch.last_detection)}")
                        last_detection_str = str(ch.last_detection) if ch.last_detection else None
                
                channels_data.append({
                    'channel_id': ch.channel_id,
                    'snapshot_count': ch.snapshot_count,
                    'total_detections': int(ch.total_detections or 0),
                    'avg_detections': round(float(ch.avg_detections or 0), 2),
                    'last_detection': last_detection_str
                })
            
            hourly_data = [
                {'hour': int(h.hour), 'count': h.count}
                for h in hourly_stats
            ]
            
            daily_data = []
            for d in daily_stats:
                try:
                    # Handle date conversion safely
                    # cast(..., Date) might return date object, datetime object, or string depending on DB
                    date_str = None
                    if d.snapshot_date:
                        if isinstance(d.snapshot_date, str):
                            date_str = d.snapshot_date
                        elif hasattr(d.snapshot_date, 'isoformat'):
                            # It's a date or datetime object
                            date_str = d.snapshot_date.isoformat()
                        elif hasattr(d.snapshot_date, 'strftime'):
                            # It's a date object with strftime
                            date_str = d.snapshot_date.strftime('%Y-%m-%d')
                        else:
                            # Fallback: convert to string
                            date_str = str(d.snapshot_date)
                    
                    daily_data.append({
                        'date': date_str,
                        'snapshot_count': d.count,
                        'detections': int(d.detections or 0)
                    })
                except Exception as e:
                    logger.error(f"Error formatting daily stat date: {e}, type: {type(d.snapshot_date)}, value: {d.snapshot_date}", exc_info=True)
                    daily_data.append({
                        'date': str(d.snapshot_date) if d.snapshot_date else None,
                        'snapshot_count': d.count,
                        'detections': int(d.detections or 0)
                    })
            
            # Calculate totals
            total_snapshots = sum(c['snapshot_count'] for c in channels_data)
            total_detections = sum(c['total_detections'] for c in channels_data)
            
            # Identify most active channel
            most_active = max(channels_data, key=lambda x: x['snapshot_count']) if channels_data else None
            
            # Identify peak hour
            peak_hour = hourly_data[0] if hourly_data else None
            
            return {
                'total_snapshots': total_snapshots,
                'total_detections': total_detections,
                'channels': channels_data,
                'daily_trend': daily_data,
                'hourly_distribution': hourly_data,
                'most_active_channel': most_active,
                'peak_hour': peak_hour,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting cash detection analytics: {e}")
            return {
                'total_snapshots': 0,
                'total_detections': 0,
                'channels': [],
                'daily_trend': [],
                'hourly_distribution': [],
                'most_active_channel': None,
                'peak_hour': None,
                'period_days': days
            }
    
    def get_cash_snapshots(self, channel_id=None, limit=50):
        """Get cash detection snapshots from database"""
        try:
            query = self.CashSnapshot.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.order_by(
                self.CashSnapshot.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': snap.id,
                    'channel_id': snap.channel_id,
                    'snapshot_filename': snap.snapshot_filename,
                    'snapshot_path': snap.snapshot_path,
                    'alert_message': snap.alert_message,
                    'alert_data': json.loads(snap.alert_data) if snap.alert_data else {},
                    'detection_count': snap.detection_count,
                    'file_size': snap.file_size,
                    'created_at': snap.created_at.isoformat(),
                    'snapshot_url': f'/static/cash_snapshots/{snap.snapshot_filename}'
                } for snap in snapshots
            ]
            
        except Exception as e:
            logger.error(f"Error getting cash snapshots: {e}")
            return []
    
    def delete_cash_snapshot(self, snapshot_id):
        """Delete cash detection snapshot from database"""
        try:
            snapshot = self.CashSnapshot.query.get(snapshot_id)
            if snapshot:
                # Delete the actual file
                import os
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting cash snapshot: {e}")
            raise e
    
    def clear_old_cash_snapshots(self, days=7):
        """Clear old cash detection snapshots older than specified days"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            old_snapshots = self.CashSnapshot.query.filter(
                self.CashSnapshot.created_at < cutoff_date
            ).all()
            
            deleted_count = 0
            for snapshot in old_snapshots:
                # Delete file if exists
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                # Delete database record
                self.db.session.delete(snapshot)
                deleted_count += 1
            
            self.db.session.commit()
            return deleted_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old cash snapshots: {e}")
            raise e
    
    # Fall Detection snapshot methods
    def save_fall_snapshot(self, channel_id, snapshot_filename, snapshot_path, 
                          alert_message, alert_data, file_size, fall_duration):
        """Save fall detection snapshot to database"""
        try:
            snapshot = self.FallSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_message=alert_message,
                alert_data=json.dumps(alert_data),
                fall_duration=fall_duration,
                file_size=file_size
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            # Send Telegram notification
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='fall_alert',
                alert_message=alert_message or f"Fall detected (duration: {fall_duration:.1f}s)",
                snapshot_path=snapshot_path,
                alert_data=alert_data
            )
            
            return snapshot.id
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error saving fall snapshot: {e}")
            raise e
    
    def get_fall_snapshots(self, channel_id=None, limit=50):
        """Get fall detection snapshots from database"""
        try:
            query = self.FallSnapshot.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.order_by(self.FallSnapshot.created_at.desc()).limit(limit).all()
            
            return [
                {
                    'id': snap.id,
                    'channel_id': snap.channel_id,
                    'snapshot_filename': snap.snapshot_filename,
                    'snapshot_path': snap.snapshot_path,  # Include full path
                    'alert_message': snap.alert_message,
                    'alert_data': json.loads(snap.alert_data) if snap.alert_data else {},
                    'fall_duration': snap.fall_duration,
                    'file_size': snap.file_size,
                    'created_at': snap.created_at.isoformat(),
                    # Use snapshot_path (which includes folder structure) instead of just filename
                    'snapshot_url': f'/static/{snap.snapshot_path}' if snap.snapshot_path else f'/static/fall_snapshots/{snap.snapshot_filename}'
                } for snap in snapshots
            ]
            
        except Exception as e:
            logger.error(f"Error getting fall snapshots: {e}")
            return []
    
    def delete_fall_snapshot(self, snapshot_id):
        """Delete fall detection snapshot from database"""
        try:
            snapshot = self.FallSnapshot.query.get(snapshot_id)
            if snapshot:
                # Delete the actual file
                import os
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting fall snapshot: {e}")
            raise e
    
    def clear_old_fall_snapshots(self, days=7):
        """Clear old fall detection snapshots older than specified days"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            old_snapshots = self.FallSnapshot.query.filter(
                self.FallSnapshot.created_at < cutoff_date
            ).all()
            
            deleted_count = 0
            for snapshot in old_snapshots:
                # Delete file if exists
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                # Delete database record
                self.db.session.delete(snapshot)
                deleted_count += 1
            
            self.db.session.commit()
            return deleted_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old fall snapshots: {e}")
            raise e
    
    # Grooming Detection Methods
    def save_grooming_snapshot(self, channel_id, snapshot_filename, snapshot_path, 
                               alert_message, alert_data, violation_type, violation_item, file_size):
        """Save grooming violation snapshot to database"""
        try:
            snapshot = self.GroomingSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_message=alert_message,
                alert_data=json.dumps(alert_data),
                violation_type=violation_type,
                violation_item=violation_item,
                file_size=file_size
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            # Send Telegram notification
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='grooming_alert',
                alert_message=alert_message or f"Grooming violation: {violation_type} - {violation_item}",
                snapshot_path=snapshot_path,
                alert_data={'violation_type': violation_type, 'violation_item': violation_item}
            )
            
            return snapshot.id
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error saving grooming snapshot: {e}")
            raise e
    
    def get_grooming_snapshots(self, channel_id=None, limit=50):
        """Get grooming violation snapshots from database"""
        try:
            query = self.GroomingSnapshot.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.order_by(self.GroomingSnapshot.created_at.desc()).limit(limit).all()
            
            return [
                {
                    'id': snap.id,
                    'channel_id': snap.channel_id,
                    'snapshot_filename': snap.snapshot_filename,
                    'alert_message': snap.alert_message,
                    'alert_data': json.loads(snap.alert_data) if snap.alert_data else {},
                    'violation_type': snap.violation_type,
                    'violation_item': snap.violation_item,
                    'file_size': snap.file_size,
                    'created_at': snap.created_at.isoformat(),
                    'snapshot_url': f'/static/grooming_snapshots/{snap.snapshot_filename}'
                } for snap in snapshots
            ]
            
        except Exception as e:
            logger.error(f"Error getting grooming snapshots: {e}")
            return []
    
    def delete_grooming_snapshot(self, snapshot_id):
        """Delete grooming violation snapshot from database"""
        try:
            snapshot = self.GroomingSnapshot.query.get(snapshot_id)
            if snapshot:
                # Delete the actual file
                import os
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting grooming snapshot: {e}")
            raise e
    
    def clear_old_grooming_snapshots(self, days=7):
        """Clear old grooming violation snapshots older than specified days"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            old_snapshots = self.GroomingSnapshot.query.filter(
                self.GroomingSnapshot.created_at < cutoff_date
            ).all()
            
            deleted_count = 0
            for snapshot in old_snapshots:
                # Delete file if exists
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                # Delete database record
                self.db.session.delete(snapshot)
                deleted_count += 1
            
            self.db.session.commit()
            return deleted_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old grooming snapshots: {e}")
            raise e
    
    # ============= User Authentication Methods =============
    
    def create_user(self, username, password, role='user'):
        """Create a new user"""
        from werkzeug.security import generate_password_hash
        
        try:
            # Check if user already exists
            existing_user = self.User.query.filter_by(username=username).first()
            if existing_user:
                return None, "Username already exists"
            
            # Create new user
            user = self.User(
                username=username,
                password_hash=generate_password_hash(password),
                role=role
            )
            self.db.session.add(user)
            self.db.session.commit()
            return user, None
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error creating user: {e}")
            return None, str(e)
    
    def verify_user(self, username, password):
        """Verify user credentials"""
        from werkzeug.security import check_password_hash
        
        try:
            user = self.User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, password):
                # Update last login
                user.last_login = get_ist_now()
                self.db.session.commit()
                return user
            return None
            
        except Exception as e:
            logger.error(f"Error verifying user: {e}")
            return None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        try:
            return self.User.query.get(user_id)
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def update_user_password(self, username, new_password):
        """Update user password"""
        from werkzeug.security import generate_password_hash
        
        try:
            user = self.User.query.filter_by(username=username).first()
            if user:
                user.password_hash = generate_password_hash(new_password)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error updating password: {e}")
            return False
    
    # ========== Dress Code Monitoring Methods ==========
    
    def add_dresscode_alert(self, channel_id, violations, uniform_color, snapshot_path, employee_id=None):
        """Add a dress code violation alert to database"""
        import os
        
        try:
            # Get file size
            file_size = 0
            if os.path.exists(snapshot_path):
                file_size = os.path.getsize(snapshot_path)
            
            # Extract filename from path
            snapshot_filename = os.path.basename(snapshot_path)
            
            alert = self.DressCodeAlert(
                channel_id=channel_id,
                employee_id=employee_id or 'unknown',
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                violations=violations,
                uniform_color=uniform_color,
                is_compliant=False,
                file_size=file_size
            )
            
            self.db.session.add(alert)
            self.db.session.commit()
            logger.info(f"Dress code alert saved: {snapshot_filename}")
            
            # Send Telegram notification
            violations_str = violations if isinstance(violations, str) else ', '.join(violations) if isinstance(violations, list) else str(violations)
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='dresscode_alert',
                alert_message=f"Dress code violation: {violations_str}",
                snapshot_path=snapshot_path,
                alert_data={'violations': violations_str, 'uniform_color': uniform_color}
            )
            
            return alert
            
        except Exception as e:
            try:
                self.db.session.rollback()
            except:
                pass  # Ignore rollback errors if session is not available
            logger.error(f"Error saving dress code alert: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_dresscode_alerts(self, channel_id=None, limit=50):
        """Get recent dress code alerts"""
        try:
            query = self.DressCodeAlert.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            alerts = query.order_by(self.DressCodeAlert.created_at.desc()).limit(limit).all()
            
            return [{
                'id': alert.id,
                'channel_id': alert.channel_id,
                'employee_id': alert.employee_id,
                'snapshot_path': alert.snapshot_path,
                'snapshot_filename': alert.snapshot_filename,
                'violations': alert.violations,
                'uniform_color': alert.uniform_color,
                'created_at': alert.created_at.isoformat() if alert.created_at else None,
                'timestamp': alert.created_at.isoformat() if alert.created_at else None
            } for alert in alerts]
            
        except Exception as e:
            logger.error(f"Error getting dress code alerts: {e}")
            return []
    
    def delete_dresscode_alert(self, alert_id):
        """Delete a dress code alert"""
        import os
        
        try:
            alert = self.DressCodeAlert.query.get(alert_id)
            if alert:
                # Delete snapshot file
                if os.path.exists(alert.snapshot_path):
                    os.remove(alert.snapshot_path)
                
                self.db.session.delete(alert)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting dress code alert: {e}")
            return False
    
    def get_dresscode_stats(self, channel_id=None, days=7):
        """Get dress code violation statistics"""
        try:
            from datetime import datetime, timedelta
            
            start_date = datetime.now() - timedelta(days=days)
            
            query = self.DressCodeAlert.query.filter(
                self.DressCodeAlert.created_at >= start_date
            )
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            alerts = query.all()
            
            # Count violations by type
            violation_counts = {}
            uniform_counts = {}
            
            for alert in alerts:
                # Count by uniform color
                if alert.uniform_color:
                    uniform_counts[alert.uniform_color] = uniform_counts.get(alert.uniform_color, 0) + 1
                
                # Count individual violations
                if alert.violations:
                    for violation in alert.violations.split(','):
                        violation = violation.strip()
                        violation_counts[violation] = violation_counts.get(violation, 0) + 1
            
            return {
                'total_violations': len(alerts),
                'violation_types': violation_counts,
                'uniform_colors': uniform_counts,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting dress code stats: {e}")
            return {
                'total_violations': 0,
                'violation_types': {},
                'uniform_colors': {},
                'period_days': days
            }
    
    def add_ppe_alert(self, channel_id, violations, violation_types=None, snapshot_path=None, employee_id=None, alert_data=None):
        """Add a PPE violation alert to database"""
        import os
        import json
        
        try:
            # Get file size if snapshot exists
            file_size = 0
            snapshot_filename = None
            if snapshot_path and os.path.exists(snapshot_path):
                file_size = os.path.getsize(snapshot_path)
                snapshot_filename = os.path.basename(snapshot_path)
            
            # Convert violations list to string if needed
            if isinstance(violations, list):
                violations_str = ', '.join(violations)
            else:
                violations_str = str(violations)
            
            # Convert violation_types to JSON string if provided
            violation_types_str = None
            if violation_types:
                if isinstance(violation_types, (list, dict)):
                    violation_types_str = json.dumps(violation_types)
                else:
                    violation_types_str = str(violation_types)
            
            # Convert alert_data to JSON string if provided
            alert_data_str = None
            if alert_data:
                if isinstance(alert_data, dict):
                    alert_data_str = json.dumps(alert_data)
                else:
                    alert_data_str = str(alert_data)
            
            alert = self.PPEAlert(
                channel_id=channel_id,
                employee_id=employee_id or 'unknown',
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                violations=violations_str,
                violation_types=violation_types_str,
                alert_data=alert_data_str,
                is_compliant=False,
                file_size=file_size
            )
            
            self.db.session.add(alert)
            self.db.session.commit()
            logger.info(f"PPE alert saved: {violations_str} for channel {channel_id}")
            
            # Send Telegram notification
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='ppe_alert',
                alert_message=f"PPE violation: {violations_str}",
                snapshot_path=snapshot_path,
                alert_data={'violations': violations_str, 'violation_types': violation_types}
            )
            return alert
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error saving PPE alert: {e}")
            return None
    
    def get_ppe_alerts(self, channel_id=None, limit=50):
        """Get recent PPE violation alerts"""
        import json
        
        try:
            query = self.PPEAlert.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            alerts = query.order_by(self.PPEAlert.created_at.desc()).limit(limit).all()
            
            result = []
            for alert in alerts:
                # Parse JSON fields
                violation_types = None
                alert_data = None
                
                if alert.violation_types:
                    try:
                        violation_types = json.loads(alert.violation_types)
                    except:
                        violation_types = alert.violation_types
                
                if alert.alert_data:
                    try:
                        alert_data = json.loads(alert.alert_data)
                    except:
                        alert_data = alert.alert_data
                
                result.append({
                    'id': alert.id,
                    'channel_id': alert.channel_id,
                    'employee_id': alert.employee_id,
                    'snapshot_path': alert.snapshot_path,
                    'snapshot_filename': alert.snapshot_filename,
                    'violations': alert.violations,
                    'violation_types': violation_types,
                    'alert_data': alert_data,
                    'is_compliant': alert.is_compliant,
                    'timestamp': alert.created_at.isoformat() if alert.created_at else None,
                    'created_at': alert.created_at.strftime('%Y-%m-%d %H:%M:%S') if alert.created_at else None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting PPE alerts: {e}")
            return []
    
    def delete_ppe_alert(self, alert_id):
        """Delete a PPE violation alert"""
        import os
        
        try:
            alert = self.PPEAlert.query.get(alert_id)
            if alert:
                # Delete snapshot file if exists
                if alert.snapshot_path and os.path.exists(alert.snapshot_path):
                    os.remove(alert.snapshot_path)
                
                self.db.session.delete(alert)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting PPE alert: {e}")
            return False
    
    def add_queue_violation(self, channel_id, violation_type, violation_message, queue_count=0, counter_count=0, wait_time_seconds=0.0, snapshot_path=None, alert_data=None):
        """Add a queue violation to database"""
        import os
        import json
        
        try:
            # Get file size if snapshot exists
            file_size = 0
            snapshot_filename = None
            if snapshot_path and os.path.exists(snapshot_path):
                file_size = os.path.getsize(snapshot_path)
                snapshot_filename = os.path.basename(snapshot_path)
            
            # Convert alert_data to JSON string if provided
            alert_data_str = None
            if alert_data:
                if isinstance(alert_data, dict):
                    alert_data_str = json.dumps(alert_data)
                else:
                    alert_data_str = str(alert_data)
            
            violation = self.QueueViolation(
                channel_id=channel_id,
                violation_type=violation_type,
                violation_message=violation_message,
                queue_count=queue_count,
                counter_count=counter_count,
                wait_time_seconds=wait_time_seconds,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_data=alert_data_str,
                file_size=file_size
            )
            
            self.db.session.add(violation)
            self.db.session.commit()
            
            logger.info(f"Queue violation saved: {violation_type} for channel {channel_id}")
            
            # Send Telegram notification with snapshot
            alert_message = f"Queue violation: {violation_message}"
            if queue_count > 0 or counter_count > 0:
                alert_message += f" (Queue: {queue_count}, Counter: {counter_count})"
            if wait_time_seconds > 0:
                wait_min = wait_time_seconds / 60
                alert_message += f" (Wait: {wait_min:.1f} min)"
            
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='queue_violation',
                alert_message=alert_message,
                snapshot_path=snapshot_path,
                alert_data=alert_data
            )
            
            return violation.id
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error adding queue violation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_queue_violations(self, channel_id=None, limit=50):
        """Get recent queue violations"""
        import json
        
        try:
            query = self.QueueViolation.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            violations = query.order_by(self.QueueViolation.created_at.desc()).limit(limit).all()
            
            result = []
            for violation in violations:
                # Parse JSON fields
                alert_data = None
                if violation.alert_data:
                    try:
                        alert_data = json.loads(violation.alert_data)
                    except:
                        alert_data = violation.alert_data
                
                result.append({
                    'id': violation.id,
                    'channel_id': violation.channel_id,
                    'violation_type': violation.violation_type,
                    'violation_message': violation.violation_message,
                    'queue_count': violation.queue_count,
                    'counter_count': violation.counter_count,
                    'wait_time_seconds': violation.wait_time_seconds,
                    'snapshot_filename': violation.snapshot_filename,
                    'snapshot_path': violation.snapshot_path,
                    'alert_data': alert_data,
                    'file_size': violation.file_size,
                    'created_at': violation.created_at.isoformat() if violation.created_at else None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting queue violations: {e}")
            return []
    
    def delete_queue_violation(self, violation_id):
        """Delete a queue violation"""
        import os
        
        try:
            violation = self.QueueViolation.query.get(violation_id)
            if violation:
                # Delete snapshot file if exists
                if violation.snapshot_path and os.path.exists(violation.snapshot_path):
                    os.remove(violation.snapshot_path)
                
                self.db.session.delete(violation)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting queue violation: {e}")
            return False
    
    def add_table_service_violation(self, channel_id, table_id, waiting_time, snapshot_path=None, timestamp=None, alert_data=None):
        """Add a table service violation to database"""
        import os
        import json
        from datetime import datetime
        
        try:
            # Get file size if snapshot exists
            file_size = 0
            snapshot_filename = None
            if snapshot_path:
                full_path = os.path.join("static", snapshot_path)
                if os.path.exists(full_path):
                    file_size = os.path.getsize(full_path)
                    snapshot_filename = os.path.basename(snapshot_path)
            
            # Extract order_wait_time and service_wait_time from alert_data if available
            order_wait_time = None
            service_wait_time = None
            if alert_data and isinstance(alert_data, dict):
                order_wait_time = alert_data.get("order_wait_time")
                service_wait_time = alert_data.get("service_wait_time")
            
            # Convert alert_data to JSON string if provided
            alert_data_str = None
            if alert_data:
                if isinstance(alert_data, dict):
                    alert_data_str = json.dumps(alert_data)
                else:
                    alert_data_str = str(alert_data)
            
            violation = self.TableServiceViolation(
                channel_id=channel_id,
                table_id=table_id,
                waiting_time=waiting_time,
                order_wait_time=order_wait_time,
                service_wait_time=service_wait_time,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_data=alert_data_str,
                file_size=file_size,
                created_at=timestamp if timestamp else get_ist_now()
            )
            
            self.db.session.add(violation)
            self.db.session.commit()
            
            waiting_time_str = f"{waiting_time:.1f}s" if waiting_time is not None else "N/A"
            order_wait_str = f"{order_wait_time:.1f}s" if order_wait_time is not None else "N/A"
            service_wait_str = f"{service_wait_time:.1f}s" if service_wait_time is not None else "N/A"
            logger.info(f"Table service violation saved: Table {table_id} in channel {channel_id}, waiting time: {waiting_time_str}, order wait: {order_wait_str}, service wait: {service_wait_str}")
            
            # Send Telegram notification
            wait_min = waiting_time / 60 if waiting_time else 0
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='table_service_violation',
                alert_message=f"Table {table_id}: Service delay - {wait_min:.1f} min wait time",
                snapshot_path=snapshot_path,
                alert_data={'table_id': table_id, 'waiting_time': waiting_time, 'order_wait_time': order_wait_time, 'service_wait_time': service_wait_time}
            )
            
            return violation.id
            
        except Exception as e:
            logger.error(f"Error adding table service violation: {e}")
            self.db.session.rollback()
            return None
    
    def add_table_service_order(self, channel_id, table_id, order_wait_time, service_wait_time, timestamp=None, alert_data=None):
        """Add a completed table service order to database (all orders, not just violations)"""
        import json
        
        try:
            # Use order_wait_time as the primary waiting_time
            waiting_time = order_wait_time if order_wait_time is not None else (service_wait_time if service_wait_time is not None else 0.0)
            
            # Convert alert_data to JSON string if provided
            alert_data_str = None
            if alert_data:
                if isinstance(alert_data, dict):
                    alert_data_str = json.dumps(alert_data)
                else:
                    alert_data_str = str(alert_data)
            
            order = self.TableServiceViolation(
                channel_id=channel_id,
                table_id=table_id,
                waiting_time=waiting_time,
                order_wait_time=order_wait_time,
                service_wait_time=service_wait_time,
                snapshot_filename=None,  # No snapshot for regular orders
                snapshot_path=None,
                alert_data=alert_data_str,
                file_size=0,
                created_at=timestamp if timestamp else get_ist_now()
            )
            
            self.db.session.add(order)
            self.db.session.commit()
            
            logger.debug(f"Table service order saved: Table {table_id} in channel {channel_id}, order wait: {order_wait_time:.1f}s, service wait: {service_wait_time:.1f}s")
            
            return order.id
            
        except Exception as e:
            logger.error(f"Error adding table service order: {e}")
            self.db.session.rollback()
            return None
    
    def get_table_service_violations(self, channel_id=None, table_id=None, violation_type=None, limit=50, days=None):
        """Get recent table service violations"""
        import json
        from datetime import datetime, timedelta
        
        try:
            query = self.TableServiceViolation.query
            
            # Apply date filter if days is provided
            if days is not None and isinstance(days, (int, float)) and days > 0:
                date_threshold = datetime.now() - timedelta(days=days)
                query = query.filter(self.TableServiceViolation.created_at >= date_threshold)
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            if table_id:
                query = query.filter_by(table_id=table_id)
            
            violations = query.order_by(self.TableServiceViolation.created_at.desc()).limit(limit).all()
            
            # Filter by violation_type if specified (stored in alert_data JSON)
            if violation_type:
                filtered_violations = []
                for violation in violations:
                    if violation.alert_data:
                        try:
                            alert_data = json.loads(violation.alert_data) if isinstance(violation.alert_data, str) else violation.alert_data
                            if alert_data.get('violation_type') == violation_type:
                                filtered_violations.append(violation)
                        except:
                            pass
                violations = filtered_violations
            
            result = []
            for violation in violations:
                # Parse JSON fields
                alert_data = None
                if violation.alert_data:
                    try:
                        alert_data = json.loads(violation.alert_data)
                    except:
                        alert_data = violation.alert_data
                
                result.append({
                    'id': violation.id,
                    'channel_id': violation.channel_id,
                    'table_id': violation.table_id,
                    'waiting_time': violation.waiting_time,
                    'order_wait_time': violation.order_wait_time,
                    'service_wait_time': violation.service_wait_time,
                    'snapshot_filename': violation.snapshot_filename,
                    'snapshot_path': violation.snapshot_path,
                    'alert_data': alert_data,
                    'file_size': violation.file_size,
                    'created_at': violation.created_at.isoformat() if violation.created_at else None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting table service violations: {e}")
            return []
    
    def delete_table_service_violation(self, violation_id):
        """Delete a table service violation"""
        import os
        from pathlib import Path
        
        try:
            violation = self.TableServiceViolation.query.get(violation_id)
            if violation:
                # Delete snapshot file if exists - handle both relative and absolute paths
                if violation.snapshot_path:
                    # Try absolute path first
                    if os.path.isabs(violation.snapshot_path):
                        full_path = violation.snapshot_path
                    else:
                        # Try relative to static directory
                        full_path = os.path.join("static", violation.snapshot_path)
                        if not os.path.exists(full_path):
                            # Try with just the filename in static/service_discipline
                            filename = os.path.basename(violation.snapshot_path)
                            full_path = os.path.join("static", "service_discipline", filename)
                    
                    if os.path.exists(full_path):
                        try:
                            os.remove(full_path)
                            logger.info(f"Deleted snapshot file: {full_path}")
                        except PermissionError as pe:
                            logger.error(f"Permission denied deleting file {full_path}: {pe}")
                            # Continue with DB deletion even if file deletion fails
                        except Exception as file_err:
                            logger.warning(f"Could not delete file {full_path}: {file_err}")
                            # Continue with DB deletion even if file deletion fails
                    else:
                        logger.warning(f"Snapshot file not found: {full_path} (violation_id: {violation_id})")
                
                self.db.session.delete(violation)
                self.db.session.commit()
                logger.info(f"Deleted table service violation: {violation_id}")
                return True
            logger.warning(f"Table service violation not found: {violation_id}")
            return False
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting table service violation {violation_id}: {e}", exc_info=True)
            return False
    
    def clear_old_table_service_violations(self, days=7):
        """Clear old table service violations older than specified days"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            old_violations = self.TableServiceViolation.query.filter(
                self.TableServiceViolation.timestamp < cutoff_date
            ).all()
            
            deleted_count = 0
            for violation in old_violations:
                # Delete snapshot file if exists
                if violation.snapshot_path:
                    full_path = os.path.join("static", violation.snapshot_path)
                    if os.path.exists(full_path):
                        try:
                            os.remove(full_path)
                        except:
                            pass
                
                # Delete database record
                self.db.session.delete(violation)
                deleted_count += 1
            
            self.db.session.commit()
            logger.info(f"Cleared {deleted_count} old table service violations (older than {days} days)")
            return deleted_count
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old table service violations: {e}")
            return 0
    
    def get_ppe_stats(self, channel_id=None, days=7):
        """Get PPE violation statistics"""
        try:
            from datetime import datetime, timedelta
            
            start_date = datetime.now() - timedelta(days=days)
            
            query = self.PPEAlert.query.filter(
                self.PPEAlert.created_at >= start_date
            )
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            alerts = query.all()
            
            # Count violations by type
            violation_counts = {}
            
            for alert in alerts:
                # Count individual violations
                if alert.violations:
                    for violation in alert.violations.split(','):
                        violation = violation.strip()
                        violation_counts[violation] = violation_counts.get(violation, 0) + 1
            
            return {
                'total_violations': len(alerts),
                'violation_types': violation_counts,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting PPE stats: {e}")
            return {
                'total_violations': 0,
                'violation_types': {},
                'period_days': days
            }
    
    def save_mopping_snapshot(self, channel_id, snapshot_filename, snapshot_path, alert_message, 
                              alert_data, file_size, detection_count, detection_time=None):
        """
        Save mopping detection snapshot to database
        
        Args:
            channel_id: Channel identifier
            snapshot_filename: Name of the snapshot file
            snapshot_path: Full path to snapshot
            alert_message: Alert message text
            alert_data: JSON dict with detection details
            file_size: File size in bytes
            detection_count: Number of mopping instances detected
            detection_time: DateTime when mopping was detected
            
        Returns:
            snapshot_id: ID of saved snapshot
        """
        import json
        
        try:
            snapshot = self.MoppingSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_message=alert_message,
                alert_data=json.dumps(alert_data) if isinstance(alert_data, dict) else alert_data,
                detection_count=detection_count,
                detection_time=detection_time or get_ist_now(),
                file_size=file_size
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            logger.info(f"Mopping snapshot saved: {snapshot_filename} (ID: {snapshot.id})")
            return snapshot.id
            
        except Exception as e:
            logger.error(f"Error saving mopping snapshot to database: {e}")
            self.db.session.rollback()
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_mopping_snapshots(self, channel_id=None, limit=50, offset=0):
        """
        Get mopping detection snapshots from database
        
        Args:
            channel_id: Filter by channel (optional)
            limit: Maximum number of records to return
            offset: Pagination offset
            
        Returns:
            List of mopping snapshot records
        """
        import json
        
        try:
            query = self.MoppingSnapshot.query.order_by(self.MoppingSnapshot.created_at.desc())
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.limit(limit).offset(offset).all()
            
            result = []
            for snap in snapshots:
                try:
                    alert_data = json.loads(snap.alert_data) if snap.alert_data else {}
                except:
                    alert_data = {}
                
                result.append({
                    'id': snap.id,
                    'channel_id': snap.channel_id,
                    'snapshot_filename': snap.snapshot_filename,
                    'snapshot_path': snap.snapshot_path,
                    'snapshot_url': f"/static/mopping_snapshots/{snap.snapshot_filename}",
                    'alert_message': snap.alert_message,
                    'alert_data': alert_data,
                    'detection_count': snap.detection_count,
                    'detection_time': snap.detection_time.isoformat() if snap.detection_time else None,
                    'file_size': snap.file_size,
                    'created_at': snap.created_at.isoformat()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting mopping snapshots: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_mopping_statistics(self, channel_id=None, days=7):
        """Get mopping detection statistics"""
        try:
            from datetime import datetime, timedelta
            
            start_date = datetime.now() - timedelta(days=days)
            
            query = self.MoppingSnapshot.query.filter(
                self.MoppingSnapshot.created_at >= start_date
            )
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.all()
            
            total_detections = sum(snap.detection_count for snap in snapshots)
            
            # Group by day
            daily_counts = {}
            for snap in snapshots:
                day = snap.created_at.date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1
            
            return {
                'total_alerts': len(snapshots),
                'total_detections': total_detections,
                'daily_counts': daily_counts,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting mopping stats: {e}")
            return {
                'total_alerts': 0,
                'total_detections': 0,
                'daily_counts': {},
                'period_days': days
            }
    
    def delete_mopping_snapshot(self, snapshot_id):
        """Delete mopping snapshot from database and filesystem"""
        try:
            snapshot = self.MoppingSnapshot.query.get(snapshot_id)
            if snapshot:
                # Delete the actual file
                import os
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                        logger.info(f"Deleted mopping snapshot file: {snapshot.snapshot_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete file {snapshot.snapshot_path}: {e}")
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                logger.info(f"Deleted mopping snapshot from database: ID {snapshot_id}")
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting mopping snapshot: {e}")
            raise e
    
    def clear_old_mopping_snapshots(self, days=7):
        """Clear old mopping snapshots older than specified days"""
        try:
            from datetime import datetime, timedelta
            import os
            
            date_threshold = datetime.now() - timedelta(days=days)
            old_snapshots = self.MoppingSnapshot.query.filter(
                self.MoppingSnapshot.created_at < date_threshold
            ).all()
            
            deleted_count = 0
            for snapshot in old_snapshots:
                # Delete file
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except Exception as e:
                        logger.warning(f"Could not delete file {snapshot.snapshot_path}: {e}")
                
                self.db.session.delete(snapshot)
                deleted_count += 1
            
            self.db.session.commit()
            logger.info(f"Cleared {deleted_count} old mopping snapshots (older than {days} days)")
            return deleted_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old mopping snapshots: {e}")
            raise e
    
    def save_smoking_snapshot(self, channel_id, snapshot_filename, snapshot_path, 
                             alert_message=None, alert_data=None, detection_count=0, 
                             detection_time=None):
        """Save smoking detection snapshot to database"""
        try:
            import json
            import os
            
            file_size = os.path.getsize(snapshot_path) if os.path.exists(snapshot_path) else 0
            
            snapshot = self.SmokingSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_message=alert_message,
                alert_data=json.dumps(alert_data) if alert_data else None,
                detection_count=detection_count,
                detection_time=detection_time or datetime.now(),
                file_size=file_size
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            logger.info(f"Smoking snapshot saved: {snapshot_filename}")
            
            # Send Telegram notification
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='smoking_alert',
                alert_message=alert_message or f"Smoke/Fire detected: {detection_count} instance(s)",
                snapshot_path=snapshot_path,
                alert_data=alert_data
            )
            
            return snapshot.id
            
        except Exception as e:
            logger.error(f"Error saving smoking snapshot: {e}")
            self.db.session.rollback()
            return None
    
    def get_smoking_snapshots(self, channel_id=None, limit=50):
        """Get smoking detection snapshots from database"""
        try:
            query = self.SmokingSnapshot.query.order_by(
                self.SmokingSnapshot.created_at.desc()
            )
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.limit(limit).all()
            
            return [{
                'id': snap.id,
                'channel_id': snap.channel_id,
                'snapshot_filename': snap.snapshot_filename,
                'snapshot_path': snap.snapshot_path,
                'alert_message': snap.alert_message,
                'detection_count': snap.detection_count,
                'detection_time': snap.detection_time.isoformat() if snap.detection_time else None,
                'file_size': snap.file_size,
                'created_at': snap.created_at.isoformat()
            } for snap in snapshots]
            
        except Exception as e:
            logger.error(f"Error getting smoking snapshots: {e}")
            return []
    
    def get_smoking_statistics(self, channel_id=None, days=7):
        """Get smoking detection statistics"""
        try:
            from datetime import datetime, timedelta
            
            start_date = datetime.now() - timedelta(days=days)
            
            query = self.SmokingSnapshot.query.filter(
                self.SmokingSnapshot.created_at >= start_date
            )
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.all()
            
            total_detections = sum(snap.detection_count for snap in snapshots)
            
            # Group by day
            daily_counts = {}
            for snap in snapshots:
                day = snap.created_at.date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1
            
            return {
                'total_alerts': len(snapshots),
                'total_detections': total_detections,
                'daily_counts': daily_counts,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting smoking stats: {e}")
            return {
                'total_alerts': 0,
                'total_detections': 0,
                'daily_counts': {},
                'period_days': days
            }
    
    def delete_smoking_snapshot(self, snapshot_id):
        """Delete smoking detection snapshot from database"""
        try:
            snapshot = self.SmokingSnapshot.query.get(snapshot_id)
            if snapshot:
                # Delete the actual file
                import os
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting smoking snapshot: {e}")
            raise e
    
    def clear_old_smoking_snapshots(self, days=7):
        """Clear old smoking detection snapshots older than specified days"""
        try:
            import os
            from datetime import timedelta
            
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            old_snapshots = self.SmokingSnapshot.query.filter(
                self.SmokingSnapshot.created_at < cutoff_date
            ).all()
            
            deleted_count = 0
            for snapshot in old_snapshots:
                # Delete file if exists
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except:
                        pass
                
                # Delete database record
                self.db.session.delete(snapshot)
                deleted_count += 1
            
            self.db.session.commit()
            return deleted_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old smoking snapshots: {e}")
            raise e






    # ==================== Phone Usage Detection Methods ====================
    
    def save_phone_snapshot(self, channel_id, snapshot_filename, snapshot_path, alert_message, 
                           alert_data, file_size, detection_count, detection_time=None):
        """Save phone usage detection snapshot to database"""
        import json
        
        try:
            snapshot = self.PhoneSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_message=alert_message,
                alert_data=json.dumps(alert_data) if isinstance(alert_data, dict) else alert_data,
                detection_count=detection_count,
                detection_time=detection_time or get_ist_now(),
                file_size=file_size
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            logger.info(f"Phone snapshot saved: {snapshot_filename} (ID: {snapshot.id})")
            
            # Send Telegram notification
            _send_telegram_alert(
                channel_id=channel_id,
                alert_type='phone_alert',
                alert_message=alert_message or f"Phone usage detected: {detection_count} instance(s)",
                snapshot_path=snapshot_path,
                alert_data=alert_data
            )
            
            return snapshot.id
            
        except Exception as e:
            logger.error(f"Error saving phone snapshot to database: {e}")
            self.db.session.rollback()
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_phone_snapshots(self, channel_id=None, limit=50, offset=0):
        """Get phone usage detection snapshots from database"""
        import json
        
        try:
            query = self.PhoneSnapshot.query
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.order_by(self.PhoneSnapshot.created_at.desc())\
                            .limit(limit)\
                            .offset(offset)\
                            .all()
            
            result = []
            for snap in snapshots:
                try:
                    alert_data = json.loads(snap.alert_data) if snap.alert_data else {}
                except:
                    alert_data = {}
                
                result.append({
                    'id': snap.id,
                    'channel_id': snap.channel_id,
                    'snapshot_filename': snap.snapshot_filename,
                    'snapshot_path': snap.snapshot_path,
                    'alert_message': snap.alert_message,
                    'alert_data': alert_data,
                    'detection_count': snap.detection_count,
                    'detection_time': snap.detection_time.isoformat() if snap.detection_time else None,
                    'file_size': snap.file_size,
                    'created_at': snap.created_at.isoformat() if snap.created_at else None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting phone snapshots: {e}")
            return []
    
    def delete_phone_snapshot(self, snapshot_id):
        """Delete a phone usage snapshot"""
        import os
        
        try:
            snapshot = self.PhoneSnapshot.query.get(snapshot_id)
            
            if snapshot:
                if os.path.exists(snapshot.snapshot_path):
                    os.remove(snapshot.snapshot_path)
                    logger.info(f"Deleted phone snapshot file: {snapshot.snapshot_path}")
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                
                logger.info(f"Deleted phone snapshot ID: {snapshot_id}")
                return True
            else:
                logger.warning(f"Phone snapshot ID {snapshot_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting phone snapshot: {e}")
            self.db.session.rollback()
            return False
    
    def clear_old_phone_snapshots(self, days=30):
        """Clear phone snapshots older than specified days"""
        import os
        from datetime import datetime, timedelta
        
        try:
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            old_snapshots = self.PhoneSnapshot.query.filter(
                self.PhoneSnapshot.created_at < cutoff_date
            ).all()
            
            count = 0
            for snapshot in old_snapshots:
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except Exception as e:
                        logger.error(f"Error deleting file {snapshot.snapshot_path}: {e}")
                
                self.db.session.delete(snapshot)
                count += 1
            
            self.db.session.commit()
            logger.info(f"Cleared {count} old phone snapshots (older than {days} days)")
            
            return count
            
        except Exception as e:
            logger.error(f"Error clearing old phone snapshots: {e}")
            self.db.session.rollback()
            return 0
    
    def get_phone_statistics(self, channel_id=None, days=7):
        """Get phone usage detection statistics"""
        from datetime import datetime, timedelta
        
        try:
            cutoff_date = get_ist_now() - timedelta(days=days)
            
            query = self.PhoneSnapshot.query.filter(
                self.PhoneSnapshot.created_at >= cutoff_date
            )
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.all()
            
            total_detections = sum(snap.detection_count for snap in snapshots)
            
            daily_counts = {}
            for snap in snapshots:
                day = snap.created_at.date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1
            
            return {
                'total_alerts': len(snapshots),
                'total_detections': total_detections,
                'daily_counts': daily_counts,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting phone statistics: {e}")
            return {
                'total_alerts': 0,
                'total_detections': 0,
                'daily_counts': {},
                'period_days': days
            }

    # Restricted Area Monitor Methods
    def save_restricted_area_snapshot(self, channel_id, snapshot_filename, snapshot_path, 
                                      alert_message, alert_data, file_size, violation_count, detection_time=None):
        """Save restricted area violation snapshot to database"""
        try:
            import json
            
            snapshot = self.RestrictedAreaSnapshot(
                channel_id=channel_id,
                snapshot_filename=snapshot_filename,
                snapshot_path=snapshot_path,
                alert_message=alert_message,
                alert_data=json.dumps(alert_data) if isinstance(alert_data, dict) else alert_data,
                violation_count=violation_count,
                detection_time=detection_time or get_ist_now(),
                file_size=file_size
            )
            
            self.db.session.add(snapshot)
            self.db.session.commit()
            
            logger.info(f"Restricted area snapshot saved: {snapshot_filename} (ID: {snapshot.id})")
            return snapshot.id
            
        except Exception as e:
            logger.error(f"Error saving restricted area snapshot: {e}")
            self.db.session.rollback()
            return None
    
    def get_restricted_area_snapshots(self, channel_id=None, limit=50, offset=0):
        """Get restricted area violation snapshots"""
        import json
        
        try:
            query = self.RestrictedAreaSnapshot.query.order_by(self.RestrictedAreaSnapshot.created_at.desc())
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.limit(limit).offset(offset).all()
            
            result = []
            for snap in snapshots:
                try:
                    alert_data = json.loads(snap.alert_data) if snap.alert_data else {}
                except:
                    alert_data = {}
                
                result.append({
                    'id': snap.id,
                    'channel_id': snap.channel_id,
                    'snapshot_filename': snap.snapshot_filename,
                    'snapshot_path': snap.snapshot_path,
                    'snapshot_url': f"/static/restricted_area_snapshots/{snap.snapshot_filename}",
                    'alert_message': snap.alert_message,
                    'alert_data': alert_data,
                    'violation_count': snap.violation_count,
                    'detection_time': snap.detection_time.isoformat() if snap.detection_time else None,
                    'file_size': snap.file_size,
                    'created_at': snap.created_at.isoformat()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting restricted area snapshots: {e}")
            return []
    
    def delete_restricted_area_snapshot(self, snapshot_id):
        """Delete restricted area snapshot"""
        try:
            snapshot = self.RestrictedAreaSnapshot.query.get(snapshot_id)
            if snapshot:
                import os
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                        logger.info(f"Deleted restricted area snapshot file: {snapshot.snapshot_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete file {snapshot.snapshot_path}: {e}")
                
                self.db.session.delete(snapshot)
                self.db.session.commit()
                logger.info(f"Deleted restricted area snapshot from database: ID {snapshot_id}")
                return True
            return False
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error deleting restricted area snapshot: {e}")
            raise e
    
    def clear_old_restricted_area_snapshots(self, days=7):
        """Clear old restricted area snapshots"""
        try:
            from datetime import datetime, timedelta
            import os
            
            date_threshold = get_ist_now() - timedelta(days=days)
            old_snapshots = self.RestrictedAreaSnapshot.query.filter(
                self.RestrictedAreaSnapshot.created_at < date_threshold
            ).all()
            
            deleted_count = 0
            for snapshot in old_snapshots:
                if os.path.exists(snapshot.snapshot_path):
                    try:
                        os.remove(snapshot.snapshot_path)
                    except Exception as e:
                        logger.warning(f"Could not delete file {snapshot.snapshot_path}: {e}")
                
                self.db.session.delete(snapshot)
                deleted_count += 1
            
            self.db.session.commit()
            logger.info(f"Cleared {deleted_count} old restricted area snapshots (older than {days} days)")
            return deleted_count
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error clearing old restricted area snapshots: {e}")
            raise e
    
    def get_restricted_area_statistics(self, channel_id=None, days=7):
        """Get restricted area monitoring statistics"""
        try:
            from datetime import datetime, timedelta
            
            start_date = get_ist_now() - timedelta(days=days)
            
            query = self.RestrictedAreaSnapshot.query.filter(
                self.RestrictedAreaSnapshot.created_at >= start_date
            )
            
            if channel_id:
                query = query.filter_by(channel_id=channel_id)
            
            snapshots = query.all()
            
            total_violations = sum(snap.violation_count for snap in snapshots)
            
            # Group by day
            daily_counts = {}
            for snap in snapshots:
                day = snap.created_at.date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1
            
            return {
                'total_alerts': len(snapshots),
                'total_violations': total_violations,
                'daily_counts': daily_counts,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting restricted area stats: {e}")
            return {
                'total_alerts': 0,
                'total_violations': 0,
                'daily_counts': {},
                'period_days': days
            }
