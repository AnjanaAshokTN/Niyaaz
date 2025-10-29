"""
Video Server Mixin for Processor Microservices
Provides video streaming endpoint for processors
"""

from flask import Flask, Response
import logging
import threading
import time

def start_video_server_for_processors(processors, port):
    """
    Start a Flask video server for a list of processors
    
    Args:
        processors: List of processor objects with get_frame() method
        port: Port number to run the server on
    """
    video_app = Flask(__name__)
    
    @video_app.route('/video_feed/<channel_id>')
    def video_feed(channel_id):
        processor = next((p for p in processors if p.channel_id == channel_id), None)
        if not processor or not processor.is_alive():
            logging.warning(f"No processor found for channel {channel_id}")
            return ("Stream not found", 404)
        
        def gen_frames():
            while True:
                try:
                    time.sleep(0.04)  # ~25 FPS
                    if hasattr(processor, 'get_frame'):
                        frame_bytes = processor.get_frame()
                        if frame_bytes:
                            yield (b'--frame\r\n' 
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   frame_bytes + b'\r\n')
                except Exception as e:
                    logging.error(f"Error generating frame: {e}")
                    break
        
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @video_app.route('/health')
    def health():
        alive_count = sum(1 for p in processors if p.is_alive())
        return {
            'status': 'healthy',
            'processor_count': len(processors),
            'alive_count': alive_count
        }
    
    # Start Flask server in daemon thread
    def run_server():
        try:
            video_app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
        except Exception as e:
            logging.error(f"Video server error on port {port}: {e}")
    
    server_thread = threading.Thread(target=run_server, daemon=True, name=f"VideoServer-{port}")
    server_thread.start()
    logging.info(f"✅ Video server started on port {port}")
    
    return server_thread

