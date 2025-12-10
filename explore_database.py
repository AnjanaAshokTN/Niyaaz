"""
Simple script to explore your database
Run this to see what data is stored in your database
"""
import os
import sys
from app import app, db_manager
from datetime import datetime, timedelta

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def explore_database():
    """Explore database contents"""
    
    with app.app_context():
        # 1. RTSP Channels
        print_section("📹 RTSP Channels (Cameras)")
        channels = db_manager.get_rtsp_channels()
        if channels:
            for ch in channels:
                status = "✅ Active" if ch.is_active else "❌ Inactive"
                print(f"  {ch.channel_id}: {ch.name} - {status}")
                print(f"    URL: {ch.rtsp_url[:60]}...")
        else:
            print("  No channels found")
        
        # 2. Recent Alerts
        print_section("🚨 Recent Alerts (Last 10)")
        alerts = db_manager.get_alert_gifs(limit=10)
        if alerts:
            for alert in alerts:
                print(f"  [{alert.created_at}] {alert.channel_id}")
                print(f"    Type: {alert.alert_type}")
                print(f"    Message: {alert.alert_message}")
                print(f"    File: {alert.gif_filename}")
        else:
            print("  No alerts found")
        
        # 3. Alert Statistics
        print_section("📊 Alert Statistics (Last 7 Days)")
        alert_types = ['queue_alert', 'crowd_alert', 'dresscode_alert', 'smoking_alert']
        for alert_type in alert_types:
            count = db_manager.get_alert_count(alert_type, days=7)
            if count > 0:
                print(f"  {alert_type}: {count} alerts")
        
        # 4. Channel Configurations
        print_section("⚙️  Channel Configurations")
        configs = db_manager.ChannelConfig.query.limit(10).all()
        if configs:
            for config in configs:
                print(f"  {config.channel_id} - {config.app_name} ({config.config_type})")
                print(f"    Updated: {config.updated_at}")
        else:
            print("  No configurations found")
        
        # 5. Queue Analytics (if available)
        print_section("📈 Queue Analytics (Today)")
        today = datetime.now().date()
        queue_data = db_manager.QueueAnalytics.query.filter(
            db_manager.QueueAnalytics.timestamp >= datetime.combine(today, datetime.min.time())
        ).limit(5).all()
        
        if queue_data:
            for q in queue_data:
                print(f"  [{q.timestamp}] {q.channel_id}")
                print(f"    Queue: {q.queue_count}, Counter: {q.counter_count}")
                if q.alert_triggered:
                    print(f"    ⚠️  Alert: {q.alert_message}")
        else:
            print("  No queue data for today")
        
        # 6. Footfall Data (if available)
        print_section("👥 Footfall Data (Today)")
        footfall = db_manager.get_today_footfall_count()
        if footfall:
            print(f"  Total IN: {footfall.get('total_in', 0)}")
            print(f"  Total OUT: {footfall.get('total_out', 0)}")
            if 'channels' in footfall:
                for ch_id, data in footfall['channels'].items():
                    print(f"    {ch_id}: IN={data.get('in', 0)}, OUT={data.get('out', 0)}")
        else:
            print("  No footfall data for today")
        
        # 7. Database File Info
        print_section("💾 Database File Information")
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sakshi.db")
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"  Location: {db_path}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Last modified: {datetime.fromtimestamp(os.path.getmtime(db_path))}")
        else:
            print("  Database file not found (will be created on first run)")
        
        print("\n" + "="*60)
        print("  ✅ Database exploration complete!")
        print("="*60 + "\n")

if __name__ == '__main__':
    try:
        explore_database()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


