#!/usr/bin/env python3
"""
Sakshi AI Setup Script
Automated setup and configuration for Sakshi AI platform
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_directories():
    """Create necessary directories"""
    directories = [
        'static/detections',
        'static/detections/shutter_videos',
        'static/assets',
        'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required dependencies"""
    requirements_file = 'config/requirements.txt'
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_models():
    """Check if model files exist"""
    models_dir = Path('models')
    required_models = [
        'yolov8n.pt',
        'best_shoplift.pt',
        'best_qpos.pt',
        'best_generic.pt',
        'shutter_model.pt'
    ]
    
    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)
    
    if missing_models:
        print(f"⚠️  Missing model files: {', '.join(missing_models)}")
        print("   Please ensure all model files are in the models/ directory")
        return False
    else:
        print("✅ All required model files found")
        return True

def create_sample_config():
    """Create sample configuration files"""
    # Create sample rtsp_links.txt if it doesn't exist
    rtsp_config = 'config/rtsp_links.txt'
    if not os.path.exists(rtsp_config):
        sample_config = """# RTSP Links Configuration File
# Format: VIDEO_PATH, Channel_Name, App1, App2, App3, ...
#
# Available Apps:
# - PeopleCounter: Bidirectional footfall tracking
# - Shoplifting: Suspicious behavior detection
# - Heatmap: Customer engagement zones
# - Generic: Multi-class object detection
# - QPOS: Point-of-sale screen monitoring
# - QueueMonitor: Queue length and analytics
# - ShutterMonitor: Shop open/close time tracking
# - Security: Security personnel interaction monitoring
# - KitchenCompliance: Kitchen compliance monitoring
#
# Example configurations:
# rtsp://your-camera-ip/stream, Main Entrance, PeopleCounter
# /path/to/video.mp4, Checkout Area, QueueMonitor, Shoplifting
"""
        with open(rtsp_config, 'w') as f:
            f.write(sample_config)
        print(f"✅ Created sample configuration: {rtsp_config}")

def check_database():
    """Check database setup"""
    print("🗄️  Checking database configuration...")
    
    # Check if PostgreSQL is configured
    try:
        import psycopg2
        from sqlalchemy import create_engine, text
        
        DATABASE_URL = "postgresql://postgres:Tneural01@localhost:5432/sakshi"
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ PostgreSQL database connection successful")
        return True
        
    except ImportError:
        print("⚠️  PostgreSQL dependencies not installed")
        print("   Run: pip install psycopg2-binary")
        return False
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("   Please ensure PostgreSQL is running and database 'sakshi' exists")
        print("   Run: python3 setup_postgresql.py")
        return False

def main():
    """Main setup function"""
    print("🚀 Sakshi AI Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Check models
    print("\n🤖 Checking model files...")
    check_models()
    
    # Create sample config
    print("\n⚙️  Setting up configuration...")
    create_sample_config()
    
    # Check database
    print("\n🗄️  Checking database...")
    check_database()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Configure your video sources in config/rtsp_links.txt")
    print("2. Update Telegram settings in main_app.py (optional)")
    print("3. Run: python main_app.py")
    print("4. Open: http://localhost:5001")

if __name__ == "__main__":
    main()
