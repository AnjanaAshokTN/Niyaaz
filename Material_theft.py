import cv2
import time
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "best.pt"
TARGET_CLASS = "weighing_machine_item"
ALERT_COOLDOWN = 5   # seconds

# ---------------- MAIN ----------------
def process_weighing_camera(camera_id, rtsp):

    print(f"\n🎥 CAMERA {camera_id} – WEIGHING ALERT STARTED")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(rtsp)

    last_alert_time = 0
    prev_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        detected_classes = []
        item_detected = False

        # ---------------- READ DETECTIONS ----------------
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls[0])]
            cls = cls.lower().replace(" ", "_")
            detected_classes.append(cls)

            if cls == TARGET_CLASS:
                item_detected = True

                # OPTIONAL: draw box for debugging
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "ITEM ON WEIGHING MACHINE",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # ---------------- PRINT CLASSES ----------------
        if detected_classes:
            print(f"CAM{camera_id} → {detected_classes}")

        # ---------------- ALERT LOGIC ----------------
        current_time = time.time()

        if item_detected and not prev_detected:
            if current_time - last_alert_time > ALERT_COOLDOWN:
                print(f"🚨 ALERT: Weighing machine item detected (CAM {camera_id})")
                last_alert_time = current_time

        prev_detected = item_detected

        cv2.imshow(f"CAMERA {camera_id} - WEIGHING", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- RUN ----------------
if __name__ == "__main__":

    CAM_WEIGHING_RTSP = "rtsp://132.154.208.136:554/user=admin&password=NIVPL@5566&channel=15&stream=0.sdp?"
    process_weighing_camera(15, CAM_WEIGHING_RTSP)