import cv2
from ultralytics import YOLO

def draw_alert(frame, text, color=(0, 0, 255)):
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3, cv2.LINE_AA)
    return frame


def main():

    CAMERA_ID = 1  # change

    if CAMERA_ID == 1:
        MODE = "uniform"
        ALLOWED_UNIFORMS = ["uniform_black", "uniform_cream"]

    elif CAMERA_ID == 2:
        MODE = "uniform"
        ALLOWED_UNIFORMS = ["uniform_blue"]

    elif CAMERA_ID in [9, 10]:
        MODE = "ppe"
        PPE_MISSING = ["no_apron", "no_gloves", "no_shoes", "no_hairnet"]

    model = YOLO("best.pt")

    video_source = "rtsp://admin:Niyaaz%401166@115.247.213.246:554/cam/realmonitor?channel=1&subtype=0"
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("❌ Could not open source.")
        return

    print(f"🎥 CAMERA {CAMERA_ID} MODE={MODE}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        alert_text = ""

        detected_classes = []

        # -------------------------------------
        # COLLECT ALL DETECTED CLASS NAMES
        # -------------------------------------
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            class_name = results[0].names[cls_id]
            class_name_clean = class_name.lower().replace(" ", "_").replace("-", "_")
            detected_classes.append(class_name_clean)

        if detected_classes:
            print("DETECTED CLASSES:", detected_classes)  # DEBUG

        # -------------------------------------
        # UNIFORM CAMERAS
        # -------------------------------------
        if MODE == "uniform":
            
            # CASH DRAWER OPEN (flexible matching)
            if any("cash" in cls and "open" in cls for cls in detected_classes):
                alert_text = "⚠ CASH DRAW OPEN!"

            # WRONG UNIFORM
            for cls in detected_classes:
                if cls.startswith("uniform_") and cls not in ALLOWED_UNIFORMS:
                    alert_text = "⚠ WRONG UNIFORM DETECTED!"

        # -------------------------------------
        # PPE CAMERAS
        # -------------------------------------
        elif MODE == "ppe":
            for cls in detected_classes:
                if cls in PPE_MISSING:
                    alert_text = f"⚠ {cls.replace('_', ' ').upper()}!"

        # -------------------------------------
        # SHOW ALERT
        # -------------------------------------
        if alert_text:
            frame = draw_alert(frame, alert_text)

        cv2.imshow("Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
