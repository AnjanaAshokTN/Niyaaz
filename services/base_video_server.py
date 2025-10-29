"""
Base Video Server for Processor Microservices
Provides HTTP endpoints for video streaming
"""

from flask import Flask, Response
import threading
import time

def create_video_server(processor, port):
    """Create a Flask app for video streaming from a processor"""
    app = Flask(__name__)
    
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'processor': processor.name if hasattr(processor, 'name') else 'unknown'}
    
    @app.route('/video_feed')
    def video_feed():
        def gen_frames():
            while True:
                time.sleep(0.04)  # ~25 FPS
                if hasattr(processor, 'get_frame'):
                    frame_bytes = processor.get_frame()
                    if frame_bytes:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               frame_bytes + b'\r\n')
        
        return Response(gen_frames(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return app

def start_video_server(processor, port):
    """Start video server in a separate thread"""
    app = create_video_server(processor, port)
    thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=port, debug=False, threaded=True),
        daemon=True,
        name=f"VideoServer-{port}"
    )
    thread.start()
    return thread

