import cv2
import time
import math
import json
from threading import Thread, Lock
from collections import deque
import sys
import os

import numpy as np
from ultralytics import YOLO

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 

# ----------------- USER TUNABLES -----------------
DEVICE = "cpu"
print(f"Device: {DEVICE}")

DETECTOR_MODEL_PATH = "yolov8n.pt"  # COCO person model only

# capture resolution
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# detection downscale factor (lower => faster, less detail)
DETECT_SCALE = 0.35
DETECT_EVERY = 3


# smoothing and timings
PRESENCE_FRAMES = 5      # frames of continuous presence before we commit Working/Not Working
ABSENCE_FRAMES = 3       # frames of continuous absence before we commit Person Left
STATUS_HISTORY_LEN = 15  # extra smoothing to remove flicker

# UI/draw tuning
DRAW_EVERY = 2

# ----------------- CAMERA CONFIG -----------------
CAMERAS = {
   "cam1": {
        "name": "purchase_cabin",
        "rtsp": "rtsp://admin:Admin%40123@172.23.0.12:554/Streaming/Channels/103",
    },
      "cam2": {
          "name": "tenders_cabin",
          "rtsp": "rtsp://admin:Admin%40123@172.23.0.11:554/Streaming/Channels/103",
          },
    "cam3": {
        "name": "office_entry",
        "rtsp": "rtsp://admin:Admin%40123@172.23.0.16:554/Streaming/Channels/103",
    },
    "cam4": {
        "name": "server_entry",
        "rtsp": "rtsp://admin:Admin%40123@172.23.0.7:554/Streaming/Channels/103",
    },
    "cam5": {
        "name": "front_office2",
        "rtsp": "rtsp://admin:Admin%40123@172.23.0.5:554/Streaming/Channels/103",
    },
    "cam6": {
        "name": "front_office1",
        "rtsp": "rtsp://admin:Admin%40123@172.23.0.15:554/Streaming/Channels/103",
     },

}

# ----------------- Helpers -----------------
def get_centroid(box):
    x_min, y_min, x_max, y_max = box
    return ((x_min + x_max) // 2, (y_min + y_max) // 2)

def euclidean_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    return (x1, y1, x2, y2)

def smooth_box(old_box, new_box, alpha=0.6):
    if old_box is None:
        return new_box
    return tuple(
        int(alpha * n + (1 - alpha) * o)
        for o, n in zip(old_box, new_box)
    )


def resize_for_detector(frame, scale=DETECT_SCALE):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return small, scale

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / float(denom)

def is_seated_by_bbox(person_box):
    x1, y1, x2, y2 = person_box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = h / w
    if w < 20 or h < 40:
        return False
    return aspect <= 2.0   # True => seated, False => standing

# ----------------- Classes -----------------
class ObjectDetector:
    def __init__(self, model_path, device="cpu"):
        try:
            # loading the model once
            self.model = YOLO(model_path).to(device)
        except Exception as e:
            print(f"Error loading detector model {model_path}: {e}")
            raise
        self.device = device

    def detect_persons(self, frame):
        try:
            results = self.model(
                frame,
                verbose=False,
                device=self.device,
                half=(self.device != "cpu"),
                conf=0.20,
                iou=0.45,
            )
        except Exception:
            return []

        persons = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            try:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    persons.append((int(x1), int(y1), int(x2), int(y2)))
            except Exception:
                try:
                    arr = boxes.xyxy.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(arr)):
                        if clss[i] != 0:
                            continue
                        x1, y1, x2, y2 = arr[i].astype(int)
                        persons.append((int(x1), int(y1), int(x2), int(y2)))
                except Exception:
                    pass
        return persons

class VideoStream:
    def __init__(self, src=0, backend=cv2.CAP_ANY, open_retries=5, wait_s=1.0):
        self.src = src
        self.backend = backend
        self.open_retries = open_retries
        self.wait_s = wait_s
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.read_thread = None
        self._open_stream_with_retries()
        if self.stream is None or not self.stream.isOpened():
            print(f"❌ Error: Cannot open video source at {src}")
            sys.exit(1)
        (self.grabbed, self.frame) = self.stream.read()
        self.read_thread = Thread(target=self.update, args=(), daemon=True)

    def _open_stream_with_retries(self):
        for i in range(self.open_retries):
            try:
                cap = cv2.VideoCapture(self.src, self.backend)
            except Exception:
                cap = cv2.VideoCapture(self.src)
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.stream = cap
                    print(f"[VideoStream] opened stream on attempt {i+1} for {self.src}")
                    return
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
            else:
                try:
                    cap.release()
                except Exception:
                    pass
            print(f"[VideoStream] attempt {i+1} failed for {self.src}, retrying in {self.wait_s}s")
            time.sleep(self.wait_s)
        self.stream = None

    def start(self):
        if self.read_thread:
            self.read_thread.start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream is None:
                time.sleep(0.05)
                continue
            try:
                grabbed = self.stream.grab()
                if not grabbed:
                    time.sleep(0.01)
                    continue
                ret, frame = self.stream.retrieve()
                if ret:
                    self.grabbed, self.frame = True, frame
                else:
                    self.grabbed = False
            except Exception:
                try:
                    self.grabbed, self.frame = self.stream.read()
                except Exception:
                    self.grabbed = False
                    time.sleep(0.01)

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        if self.stream:
            try:
                self.stream.release()
            except Exception:
                pass

# ----------------- ROI helpers -----------------
def configure_rois_manual(cap, preview_scale=0.7):
    # same as earlier ROI configuration - keep simple for now
    import cv2 as _cv2
    drawing_points = []
    final_rois = {}
    desk_counter = 1

    window_name = "ROI Configuration"
    _cv2.namedWindow(window_name)

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing_points
        if event == _cv2.EVENT_LBUTTONDOWN:
            if len(drawing_points) < 2:
                drawing_points.append((x, y))
            else:
                drawing_points = [(x, y)]
    _cv2.setMouseCallback(window_name, mouse_cb)

    # wait for first frame
    while True:
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.02)

    full_h, full_w = frame.shape[:2]
    display_w, display_h = int(full_w * preview_scale), int(full_h * preview_scale)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        frame_to_draw = frame.copy()
        for name, coords in final_rois.items():
            _cv2.rectangle(frame_to_draw, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 2)
            _cv2.putText(frame_to_draw, name, (coords[0], coords[1]-10), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        preview = _cv2.resize(frame_to_draw, (display_w, display_h))
        if len(drawing_points) == 2:
            p1, p2 = drawing_points
            _cv2.rectangle(preview, p1, p2, (0,255,255), 2)
        _cv2.imshow(window_name, preview)
        key = _cv2.waitKey(1) & 0xFF
        if key == ord("s") and len(drawing_points) == 2:
            p1, p2 = drawing_points
            p1_full = (int(p1[0] / preview_scale), int(p1[1] / preview_scale))
            p2_full = (int(p2[0] / preview_scale), int(p2[1] / preview_scale))
            x_min, y_min = min(p1_full[0], p2_full[0]), min(p1_full[1], p2_full[1])
            x_max, y_max = max(p1_full[0], p2_full[0]), max(p1_full[1], p2_full[1])
            desk_name = f"Desk {desk_counter}"
            final_rois[desk_name] = [x_min, y_min, x_max, y_max]
            print(f"✅ SAVED {desk_name}: {final_rois[desk_name]}")
            drawing_points = []
            desk_counter += 1
        elif key == ord("n"):
            drawing_points = []
            desk_counter += 1
        elif key == ord("q"):
            _cv2.destroyWindow(window_name)
            break
    return final_rois

def load_rois_from_file(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} desks from {path}")
        return data
    except Exception:
        return {}

def save_rois_to_file(rois, path):
    try:
        with open(path, "w") as f:
            json.dump(rois, f, indent=2)
        print(f"💾 Saved {len(rois)} desks to {path}")
    except Exception as e:
        print(f"Error saving ROIs to {path}: {e}")

# ----------------- Globals -----------------
frames = {}   # cam_id -> latest frame
locks = {}    # cam_id -> Lock()

logs_lock = Lock()
desk_logs = {}   # cam_id -> desk_name -> { employee, status, working, not_working }

# ----------------- Main monitor -----------------
def monitor(cam_id, cap_stream, rois):
    detector = ObjectDetector(DETECTOR_MODEL_PATH, device=DEVICE)

    desk_status = {desk: "Empty" for desk in rois}
    presence_counter = {desk: 0 for desk in rois}
    absence_counter = {desk: 0 for desk in rois}
    status_history = {desk: deque(maxlen=STATUS_HISTORY_LEN) for desk in rois}

    # initialize logs for this camera
    with logs_lock:
        if cam_id not in desk_logs:
            desk_logs[cam_id] = {}
        for desk_name in rois.keys():
            desk_logs[cam_id].setdefault(desk_name, {
                "employee": desk_name,
                "status": "Empty",
                "working": 0.0,
                "not_working": 0.0,
            })

    frame_count = 0
    last_time = time.time()

    print(f"Starting monitoring for {cam_id} ...")

    last_persons_full = []

    last_box_per_desk = {desk: None for desk in rois}


    while True:
        grabbed, frame = cap_stream.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        dt = now - last_time
        last_time = now

        frame_count += 1
        run_detection = (frame_count % DETECT_EVERY == 0)
        h, w = frame.shape[:2]

        # 1) YOLO detection (on downscaled frame for speed)
        if run_detection:
            small_frame, scale = resize_for_detector(frame)
            persons_small = detector.detect_persons(small_frame)

            persons_full = []
            for (x1, y1, x2, y2) in persons_small:
                fx1, fy1, fx2, fy2 = (
                    int(x1 / scale),
                    int(y1 / scale),
                    int(x2 / scale),
                    int(y2 / scale),
                )
                fx1, fy1, fx2, fy2 = clamp_box((fx1, fy1, fx2, fy2), w, h)
                if fx2 - fx1 > 8 and fy2 - fy1 > 8:
                    persons_full.append((fx1, fy1, fx2, fy2))

            last_persons_full = persons_full
        else:
            persons_full = last_persons_full


        # 2) Decide per desk
        for desk_name, coords in rois.items():
            dx1, dy1, dx2, dy2 = coords
            roi_box = (dx1, dy1, dx2, dy2)
            desk_center = get_centroid(roi_box)

            best_box = None
            best_dist = float("inf")

            for box in persons_full:
                cx, cy = get_centroid(box)
                if not (dx1 <= cx <= dx2 and dy1 <= cy <= dy2):
                    continue
                if iou(box, roi_box) < 0.08:
                    continue
                d = euclidean_distance((cx, cy), desk_center)
                if d < best_dist:
                    best_dist = d
                    best_box = smooth_box(last_box_per_desk[desk_name], box)
                    last_box_per_desk[desk_name] = best_box


            # No person at this desk
            if best_box is None:
                presence_counter[desk_name] = 0
                absence_counter[desk_name] += 1

                if absence_counter[desk_name] >= ABSENCE_FRAMES:
                    current = "Person Left"
                else:
                    current = desk_status[desk_name]

                desk_status[desk_name] = current
                status_history[desk_name].append(current)
                continue

            # Person present
            px1, py1, px2, py2 = best_box
            presence_counter[desk_name] += 1
            absence_counter[desk_name] = 0

            seated = is_seated_by_bbox(best_box)
            raw_status = "Working" if seated else "Not Working"

            if presence_counter[desk_name] >= PRESENCE_FRAMES:
                current = raw_status
            else:
                current = desk_status[desk_name]

            desk_status[desk_name] = current
            status_history[desk_name].append(current)

        # 3) Smooth, draw, update logs (time-based accumulation)
        for desk_name, coords in rois.items():
            hist = status_history[desk_name]
            final_status = desk_status[desk_name] if len(hist) == 0 else max(set(hist), key=hist.count)
            desk_status[desk_name] = final_status

            # update time-based logs
            with logs_lock:
                entry = desk_logs.get(cam_id, {}).get(desk_name)
                if entry is not None:
                    entry["status"] = final_status
                    if final_status == "Working":
                        entry["working"] += dt
                    elif final_status == "Not Working":
                        entry["not_working"] += dt
                    # Person Left and Empty do not accumulate time (they are absence states)

            dx1, dy1, dx2, dy2 = coords
            if final_status == "Working":
                color = (0, 255, 0)        # green
            elif final_status == "Not Working":
                color = (0, 0, 255)        # red
            elif final_status == "Person Left":
                color = (0, 200, 255)      # orange/cyan
            else:
                color = (200, 200, 200)    # grey

            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
            if frame_count % DRAW_EVERY == 0:
                cv2.putText(
                    frame,
                    f"{desk_name}: {final_status}",
                    (dx1 + 10, max(dy1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        # 4) store frame for this camera
        with locks[cam_id]:
            frames[cam_id] = frame.copy()
            
        time.sleep(0.005)



# ----------------- FastAPI app -----------------
app = FastAPI()

# Allow CORS from frontend (change to specific host(s) in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/cameras")
def get_cameras():
    result = []
    for cam_id, cfg in CAMERAS.items():
        result.append({
            "id": cam_id,
            "name": cfg["name"]
        })
    return JSONResponse(result)


@app.get("/video_feed/{cam_id}")
def video_feed(cam_id: str):
    if cam_id not in frames:
        return {"error": "Camera not found"}

    def gen():
        while True:
            with locks[cam_id]:
                frame = None if frames[cam_id] is None else frames[cam_id].copy()
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )

            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/logs")
def get_logs():
    with logs_lock:
        result = {}
        for cam_id, desks in desk_logs.items():
            result[cam_id] = {}
            for desk_name, entry in desks.items():
                result[cam_id][desk_name] = {
                    "employee": entry["employee"],
                    "status": entry.get("status", "Empty"),
                    "working_hours": round(entry["working"] / 3600.0, 2),
                    "not_working_hours": round(entry["not_working"] / 3600.0, 2),
                }
    return JSONResponse(result)

# ----------------- Entry point -----------------
if __name__ == "__main__":
    # Ensure desk_logs keys exist even before cameras start (helps frontend)
    for cam_id in CAMERAS.keys():
        with logs_lock:
            desk_logs.setdefault(cam_id, {})

    for cam_id, cfg in CAMERAS.items():
        cam_name = cfg["name"]
        rtsp_url = cfg["rtsp"]
        roi_file = f"desks_{cam_name}.json"

        print(f"\n=== Setting up {cam_id} ({cam_name}) ===")

        rois = load_rois_from_file(roi_file)
        # If no ROIs file, open the stream and let user draw ROIs once
        if not rois:
            print(f"No {roi_file} found. Draw desk boxes for camera: {cam_name}")
            cap_setup = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap_setup.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            cap_setup.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            rois = configure_rois_manual(cap_setup, preview_scale=0.7)
            save_rois_to_file(rois, roi_file)
            cap_setup.release()

        # If still no ROIs, skip starting monitor thread but leave an empty log object
        if not rois:
            print(f"❌ No ROIs defined for {cam_name}. Skipping this camera.")
            with logs_lock:
                desk_logs.setdefault(cam_id, {})
            continue

        # initialize desk_logs entries for this camera/rois
        with logs_lock:
            desk_logs.setdefault(cam_id, {})
            for desk_name in rois.keys():
                desk_logs[cam_id].setdefault(desk_name, {
                    "employee": desk_name,
                    "status": "Empty",
                    "working": 0.0,
                    "not_working": 0.0,
                })

        # Start video stream
        stream = VideoStream(rtsp_url, cv2.CAP_FFMPEG).start()
        streams_lock = Lock()
        locks[cam_id] = streams_lock
        frames[cam_id] = None

        # Monitoring thread
        t = Thread(target=monitor, args=(cam_id, stream, rois), daemon=True)
        t.start()

    print("\n🚀 Starting FastAPI backend server on http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
