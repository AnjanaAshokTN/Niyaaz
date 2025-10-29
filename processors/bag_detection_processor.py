#!/usr/bin/env python3

import time
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
import math

class BagDetectionProcessor:
    def __init__(self):
        # Configuration
        self.CONF_THRESHOLD = 0.35
        self.NMS_IOU = 0.45
        self.BAG_CLASSES = {"backpack", "handbag", "suitcase"}
        self.PERSON_CLASS = "person"
        self.TIME_THRESHOLD = 20.0  # seconds
        self.PROX_THRESHOLD_PX = 120
        self.STATIONARY_DISP_THRESHOLD = 10
        self.MAX_HISTORY = 120
        self.ALERT_REPEAT_COOLDOWN = 60.0

        # Initialize YOLO model
        self.yolo = YOLO("models/yolov8n.pt")
        
        # Initialize DeepSort tracker
        self.tracker = DeepSort(max_age=30, n_init=3)

        # State storage
        self.bag_history = {}
        self.bag_first_seen = {}
        self.bag_last_alerted = {}
        self.bag_is_flagged = {}

    def centroid_from_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def euclidean(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def is_stationary(self, history_deque, disp_threshold=None):
        if disp_threshold is None:
            disp_threshold = self.STATIONARY_DISP_THRESHOLD
        if len(history_deque) < 5:
            return False
        pts = [p for (_, p) in history_deque]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        return math.hypot(dx, dy) <= disp_threshold

    def process_frame(self, frame):
        t_now = time.time()

        # YOLO inference
        results = self.yolo(frame, conf=self.CONF_THRESHOLD, iou=self.NMS_IOU, verbose=False)

        # Convert YOLO results to tracker format
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo.model.names.get(cls_id, str(cls_id))
                x1,y1,x2,y2 = map(int, xyxy.tolist())
                w = x2 - x1
                h = y2 - y1
                detections.append({
                    "bbox": [x1, y1, w, h],
                    "confidence": conf,
                    "class_name": cls_name
                })

        # Prepare DeepSort input
        ds_inputs = []
        for d in detections:
            x,y,w,h = d["bbox"]
            ds_inputs.append(([x, y, w, h], d["confidence"], d["class_name"]))

        # Run tracker
        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)

        # Process tracks
        people_tracks = {}
        bag_tracks = {}

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            ltr = tr.to_tlbr()
            cls = tr.get_det_class()
            if cls is None:
                cls = getattr(tr, 'det_class', None)

            bbox = [int(x) for x in ltr]
            centroid = self.centroid_from_bbox(bbox)

            label = str(cls).lower() if cls is not None else ""
            if self.PERSON_CLASS in label:
                people_tracks[tid] = {"bbox": bbox, "centroid": centroid, "track": tr}
            elif any(b in label for b in self.BAG_CLASSES):
                bag_tracks[tid] = {"bbox": bbox, "centroid": centroid, "track": tr}

        # Update bag histories and process alerts
        alerts = []
        self._update_bag_histories(bag_tracks, people_tracks, t_now, alerts)

        # Visualization
        vis_frame = self._draw_visualizations(frame, people_tracks, bag_tracks, alerts)

        return vis_frame, alerts

    def _update_bag_histories(self, bag_tracks, people_tracks, t_now, alerts):
        # Update histories for current bags
        for tid, info in bag_tracks.items():
            c = info["centroid"]
            if tid not in self.bag_history:
                self.bag_history[tid] = deque(maxlen=self.MAX_HISTORY)
                self.bag_first_seen[tid] = t_now
                self.bag_is_flagged[tid] = False
                self.bag_last_alerted[tid] = 0.0
            self.bag_history[tid].append((t_now, c))

        # Clean up old histories
        for tid in list(self.bag_history.keys()):
            if tid not in bag_tracks:
                last_ts, _ = self.bag_history[tid][-1]
                if t_now - last_ts > 5.0:
                    del self.bag_history[tid]
                    del self.bag_first_seen[tid]
                    del self.bag_is_flagged[tid]
                    del self.bag_last_alerted[tid]

        # Check for alerts
        for tid, hist in self.bag_history.items():
            last_ts, last_centroid = hist[-1]
            
            # Find nearest person
            nearest_dist = float('inf')
            for pid, pinfo in people_tracks.items():
                dist = self.euclidean(last_centroid, pinfo["centroid"])
                if dist < nearest_dist:
                    nearest_dist = dist

            person_nearby = (nearest_dist <= self.PROX_THRESHOLD_PX)
            stationary = self.is_stationary(hist)
            time_seen = t_now - self.bag_first_seen.get(tid, t_now)

            if (not person_nearby) and stationary and (time_seen >= self.TIME_THRESHOLD):
                if t_now - self.bag_last_alerted.get(tid, 0.0) > self.ALERT_REPEAT_COOLDOWN:
                    self.bag_last_alerted[tid] = t_now
                    self.bag_is_flagged[tid] = True
                    alerts.append((tid, last_centroid, time_seen))
            elif person_nearby and self.bag_is_flagged.get(tid, False):
                self.bag_is_flagged[tid] = False

    def _draw_visualizations(self, frame, people_tracks, bag_tracks, alerts):
        vis = frame.copy()

        # Draw people
        for pid, pinfo in people_tracks.items():
            x1,y1,x2,y2 = pinfo["bbox"]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"P{pid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Draw bags
        for bid, binfo in bag_tracks.items():
            x1,y1,x2,y2 = binfo["bbox"]
            center = tuple(map(int, binfo["centroid"]))
            color = (255,0,0) if not self.bag_is_flagged.get(bid, False) else (0,0,255)
            
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            cv2.putText(vis, f"B{bid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if bid in self.bag_history:
                pts = [tuple(map(int, p)) for (_, p) in self.bag_history[bid]]
                for i in range(1, len(pts)):
                    cv2.line(vis, pts[i-1], pts[i], color, 2)

        # Draw alerts
        y0 = 30
        for (tid, centroid, tseen) in alerts:
            text = f"ALERT: Bag {tid} unattended for {int(tseen)}s"
            cv2.putText(vis, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            y0 += 30

        return vis