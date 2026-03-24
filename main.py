import cv2
import time
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# Adjust slots based on your video
SLOTS = [
    (50, 200, 150, 300),
    (200, 200, 300, 300),
    (350, 200, 450, 300),
    (500, 200, 600, 300)
]

vehicle_entry = {}
vehicle_exit = {}
records = []
total_revenue = 0

PRICE_PER_SEC = 1

def process_video(video_path):
    global total_revenue

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r

            if int(cls) == 2:  # car
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'car'))

        tracks = tracker.update_tracks(detections, frame=frame)

        occupied_slots = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())

            cx = int((l + l + w) / 2)
            cy = int((t + t + h) / 2)

            # Entry
            if track_id not in vehicle_entry:
                vehicle_entry[track_id] = time.time()

            # Slot check
            for i, (x1, y1, x2, y2) in enumerate(SLOTS):
                if x1 < cx < x2 and y1 < cy < y2:
                    occupied_slots.add(i)

            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # EXIT DETECTION
        current_ids = [t.track_id for t in tracks if t.is_confirmed()]

        for vid in list(vehicle_entry.keys()):
            if vid not in current_ids and vid not in vehicle_exit:

                vehicle_exit[vid] = time.time()

                entry_time = vehicle_entry[vid]
                exit_time = vehicle_exit[vid]

                duration = exit_time - entry_time
                fee = int(duration * PRICE_PER_SEC)

                total_revenue += fee

                records.append({
                    "Vehicle ID": vid,
                    "Entry Time": time.strftime('%H:%M:%S', time.localtime(entry_time)),
                    "Exit Time": time.strftime('%H:%M:%S', time.localtime(exit_time)),
                    "Duration (sec)": int(duration),
                    "Fee": fee
                })

        # Draw slots
        for i, (x1, y1, x2, y2) in enumerate(SLOTS):
            color = (0, 0, 255) if i in occupied_slots else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # UI text
        cv2.putText(frame, f"Free Slots: {len(SLOTS)-len(occupied_slots)}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.putText(frame, f"Total Revenue: {total_revenue}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        yield frame, records, total_revenue

    cap.release()