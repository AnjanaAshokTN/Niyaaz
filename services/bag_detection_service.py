from services.base_video_server import BaseVideoServer
from processors.bag_detection_processor import BagDetectionProcessor

class BagDetectionService(BaseVideoServer):
    def __init__(self):
        super().__init__()
        self.processor = BagDetectionProcessor()
        self.detections = []
        
    def process_frame(self, frame):
        processed_frame, alerts = self.processor.process_frame(frame)
        self.detections = alerts  # Store alerts for API access if needed
        return processed_frame
        
    def get_status(self):
        return {
            'detections': [
                {
                    'bag_id': tid,
                    'location': {'x': int(centroid[0]), 'y': int(centroid[1])},
                    'time_unattended': int(tseen)
                }
                for tid, centroid, tseen in self.detections
            ]
        }