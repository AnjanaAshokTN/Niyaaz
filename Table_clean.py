import cv2
import time
import math
from ultralytics import YOLO

# ---------------- CONFIG ----------------
TABLE_MATCH_DISTANCE = 100
PERSON_NEAR_DISTANCE = 180   # ⭐ KEY FIX

# ---------------- UTILS ----------------
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ---------------- MAIN ----------------
def process_camera(camera_id, rtsp):

    print(f"\n🎥 CAMERA {camera_id} STARTED")

    table_model = YOLO("best.pt")       # table_clean / table_unclean
    person_model = YOLO("yolo11n.pt")   # person detector

    cap = cv2.VideoCapture(rtsp)

    tables = {}
    next_table_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        table_results = table_model(frame, verbose=False)
        person_results = person_model(frame, verbose=False)

        detected_tables = []
        persons = []

        # ---------------- PERSON DETECTION ----------------
        for box in person_results[0].boxes:
            if person_results[0].names[int(box.cls[0])] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                persons.append((cx, cy))

        # ---------------- TABLE DETECTION ----------------
        for box in table_results[0].boxes:
            cls = table_results[0].names[int(box.cls[0])].lower()
            if "table" not in cls:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            detected_tables.append({
                "cls": cls,
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2)
            })

        # ---------------- TABLE LOGIC ----------------
        for t in detected_tables:

            cls = t["cls"]
            center = t["center"]
            bbox = t["bbox"]

            # ---------- MATCH TABLE ID ----------
            matched_id = None
            for tid, data in tables.items():
                if distance(center, data["center"]) < TABLE_MATCH_DISTANCE:
                    matched_id = tid
                    break

            if matched_id is None:
                matched_id = next_table_id
                tables[matched_id] = {
                    "center": center,
                    "unclean_start": None
                }
                next_table_id += 1

            table = tables[matched_id]
            table["center"] = center

            # ---------- PERSON NEAR CHECK (FIXED) ----------
            person_near = any(
                distance(center, p_center) < PERSON_NEAR_DISTANCE
                for p_center in persons
            )

            # ---------- UNCLEAN LOGIC ----------
            if cls == "table_unclean":

                if person_near:
                    # ✅ PERSON PRESENT → NO ALERT
                    table["unclean_start"] = None

                else:
                    if table["unclean_start"] is None:
                        table["unclean_start"] = time.time()

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        frame,
                        f"TABLE {matched_id} UNCLEAN",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2
                    )

            elif cls == "table_clean" and table["unclean_start"]:
                duration = time.time() - table["unclean_start"]
                print(f"✅ TABLE {matched_id} cleaned in {duration:.1f}s")
                table["unclean_start"] = None

        cv2.imshow(f"CAMERA {camera_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# # ---------------- RUN ----------------
# if __name__ == "__main__":
#     CAM6_RTSP = "rtsp://115.247.213.246:554/user=admin&password=NIVPL@5566&channel=6&stream=0"
#     process_camera(6, CAM6_RTSP)


# ---------------- RUN ----------------
if __name__ == "__main__":

    CAM3_RTSP = "rtsp://132.154.208.136:554/user=admin&password=NIVPL@5566&channel=3&stream=0.sdp?"
    CAM4_RTSP = "rtsp://132.154.208.136:554/user=admin&password=NIVPL@5566&channel=4&stream=0.sdp?"
    CAM5_RTSP = "rtsp://132.154.208.136:554/user=admin&password=NIVPL@5566&channel=5&stream=0.sdp?"
    CAM6_RTSP = "rtsp://132.154.208.136:554/user=admin&password=NIVPL@5566&channel=6&stream=0.sdp?"
    CAM7_RTSP = "rtsp://132.154.208.136:554/user=admin&password=NIVPL@5566&channel=7&stream=0.sdp?"
    CAM8_RTSP = "rtsp://132.154.208.136:554/user=admin&password=NIVPL@5566&channel=8&stream=0.sdp?"

    process_camera(3, CAM3_RTSP)
    process_camera(4, CAM4_RTSP)
    process_camera(5, CAM5_RTSP)
    process_camera(6, CAM6_RTSP)
    process_camera(7, CAM7_RTSP)
    process_camera(8, CAM8_RTSP)

