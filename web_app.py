import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import threading
import uuid
from datetime import datetime
from threading import Lock
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ==================== 1. CẤU HÌNH HỆ THỐNG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Ngưỡng thời gian vi phạm (giây)
VIOLATION_THRESHOLD = 2.0   # Cảnh báo khi quay đầu hoặc dùng điện thoại liên tục > 2s
NO_FACE_THRESHOLD = 3.0     # Cảnh báo khi mất mặt / che camera liên tục > 3s

# Ngưỡng góc xoay đầu (độ)
YAW_THRESHOLD = 25          # Góc quay đầu hẳn ra sau
YAW_SIDE_THRESHOLD = 10     # Góc quay đầu sang trái hoặc phải
PITCH_THRESHOLD = 10        # Góc cúi hoặc ngửa mặt

YOLO_CONFIDENCE = 0.5       # Ngưỡng tin cậy tối thiểu để phát hiện vật thể
FRAME_SKIP = 5              # Quét YOLO mỗi 5 frame để giảm tải CPU

# Thông số camera (tỉ lệ 4:3 chuẩn)
CAM_W, CAM_H = 640, 480

# Ma trận nội tại camera giả định (Intrinsic Matrix) dùng cho thuật toán giải PnP
CAM_MATRIX = np.array([[CAM_W, 0, CAM_W / 2], [0, CAM_W, CAM_H / 2], [0, 0, 1]], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

# 6 điểm mốc quan trọng trên khuôn mặt từ MediaPipe (Mũi, Cằm, 2 Mắt, 2 Khóe Miệng)
# [1: Mũi, 199: Cằm, 33: Mắt trái, 263: Mắt phải, 61: Miệng trái, 291: Miệng phải]
FACE_KEYPOINTS = [1, 199, 33, 263, 61, 291]

# Cấu hình font chữ tiếng Việt TrueType
FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/arial.ttf"
FONT_TITLE = ImageFont.truetype(FONT_PATH, 20)
FONT_SUB = ImageFont.truetype(FONT_PATH, 15)
FONT_BOX = ImageFont.truetype(FONT_PATH, 13)

# ==================== 2. KHỞI TẠO MÔ HÌNH AI ====================
YOLO_PATH = os.path.join(BASE_DIR, "models", "best.pt")
if not os.path.exists(YOLO_PATH):
    YOLO_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")

print("⏳ Đang tải mô hình YOLO & MediaPipe FaceMesh...")
yolo_model = YOLO(YOLO_PATH)

# refine_landmarks=False giúp tối ưu CPU vì không cần tính toán chi tiết mống mắt
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ Tải mô hình hoàn tất!")

# File nhật ký CSV
log_filename = os.path.join(BASE_DIR, f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(log_filename, 'w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow(['Thời gian', 'Loại vi phạm', 'Thời gian kéo dài (giây)', 'File ảnh'])

# ==================== 3. QUẢN LÝ TRẠNG THÁI (THREAD-SAFE) ====================
state_lock = Lock()
snapshots_history = []
stats = {
    "total_violations": 0,
    "phone": 0,
    "calculator": 0,
    "head_pose": 0,
    "no_face": 0,
    "is_running": True
}

# ==================== 4. CÁC HÀM XỬ LÝ CỐT LÕI (CORE FUNCTIONS) ====================

def get_head_pose(landmarks):
    """
    [CỐT LÕI 1]: TÍNH TOÁN HƯỚNG VÀ GÓC QUAY ĐẦU (HEAD POSE ESTIMATION)
    - Nguyên lý: Sử dụng thuật toán Perspective-n-Point (solvePnP) của OpenCV.
    - So sánh tọa độ 2D của 6 điểm mốc trên ảnh chụp camera với tọa độ 3D mô hình khuôn mặt chuẩn.
    - Tính ra vector xoay (rot_vec) -> ma trận xoay (Rodrigues) -> góc Euler (Pitch, Yaw, Roll).
    - Pitch: Cúi (-) / Ngửa (+) mặt.
    - Yaw: Quay đầu sang trái (+) / phải (-).
    """
    if not landmarks:
        return 0, 0, "Mất mặt / Che camera"

    pts = landmarks[0].landmark
    pts_2d = np.array([[int(pts[i].x * CAM_W), int(pts[i].y * CAM_H)] for i in FACE_KEYPOINTS], dtype=np.float64)
    pts_3d = np.array([[int(pts[i].x * CAM_W), int(pts[i].y * CAM_H), pts[i].z] for i in FACE_KEYPOINTS], dtype=np.float64)

    # Giải bài toán PnP tìm góc hướng đầu
    success, rot_vec, _ = cv2.solvePnP(pts_3d, pts_2d, CAM_MATRIX, DIST_COEFFS)
    if not success:
        return 0, 0, None

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    pitch, yaw = int(angles[0] * 360), int(angles[1] * 360)

    # Phân loại hành vi vi phạm dựa trên ngưỡng góc
    violation = None
    if abs(yaw) > YAW_THRESHOLD:
        violation = "Quay hẳn ra sau"
    elif yaw > YAW_SIDE_THRESHOLD:
        violation = "Quay sang trái"
    elif yaw < -YAW_SIDE_THRESHOLD:
        violation = "Quay sang phải"
    elif pitch > PITCH_THRESHOLD:
        violation = "Ngửa mặt lên"
    elif pitch < -PITCH_THRESHOLD:
        violation = "Cúi xuống"

    return pitch, yaw, violation


def check_yolo(frame):
    """
    [CỐT LÕI 2]: PHÁT HIỆN VẬT THỂ CẤM BẰNG YOLO11 (OBJECT DETECTION)
    - Nhận diện các vật thể: Điện thoại (cấm) và Máy tính cầm tay (hợp lệ).
    - Lọc các bounding box có độ tin cậy (confidence) > 0.5.
    - Trả về loại vi phạm và danh sách hộp bao (bounding boxes) kèm màu sắc.
    """
    violation, bboxes = None, []
    results = yolo_model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) <= YOLO_CONFIDENCE:
                continue
            name = yolo_model.names[int(box.cls[0])].lower()
            coords = list(map(int, box.xyxy[0]))
            if "phone" in name:
                violation = "Sử dụng ĐIỆN THOẠI"
                bboxes.append((coords, "ĐIỆN THOẠI (CẤM)", (0, 0, 255)))
            elif "calculator" in name:
                bboxes.append((coords, "MÁY TÍNH (HỢP LỆ)", (0, 255, 0)))
    return violation, bboxes


def draw_vietnamese_hud(frame, status_text, status_color_bgr, angle_text, bboxes):
    """
    [CỐT LÕI 3]: VẼ GIAO DIỆN VÀ CHỮ TIẾNG VIỆT ĐẦY ĐỦ DẤU (HUD OVERLAY)
    - Do cv2.putText chỉ hỗ trợ ASCII không dấu, sử dụng Pillow (PIL) để vẽ chữ TrueType có dấu.
    - Vẽ hộp bao vật thể bằng OpenCV.
    - Chuyển BGR -> RGB để PIL vẽ text sắc nét, có bóng mờ tăng độ tương phản.
    """
    for (x1, y1, x2, y2), _, box_color in bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Vẽ nhãn trên hộp bao vật thể
    for (x1, y1, x2, y2), label, box_color in bboxes:
        c_rgb = (box_color[2], box_color[1], box_color[0])
        draw.rectangle([(x1, y1 - 20), (x1 + len(label) * 9 + 8, y1)], fill=(10, 15, 25, 200))
        draw.text((x1 + 3, y1 - 19), label, font=FONT_BOX, fill=c_rgb)

    # Banner trạng thái lớn góc trên trái (BÌNH THƯỜNG / CẢNH BÁO)
    status_rgb = (status_color_bgr[2], status_color_bgr[1], status_color_bgr[0])
    draw.text((21, 21), status_text, font=FONT_TITLE, fill=(0, 0, 0))
    draw.text((20, 20), status_text, font=FONT_TITLE, fill=status_rgb)

    # Dòng hiển thị góc quay đầu
    if angle_text:
        draw.text((21, 51), angle_text, font=FONT_SUB, fill=(0, 0, 0))
        draw.text((20, 50), angle_text, font=FONT_SUB, fill=(255, 190, 50))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def save_snapshot(violation_type, duration, frame):
    """
    [CỐT LÕI 4]: LƯU BẰNG CHỨNG VÀ CẬP NHẬT SNAPSHOT
    - Lưu file ảnh JPG vào thư mục screenshots/ với định danh thời gian và mã UUID chống trùng.
    - Cập nhật số liệu thống kê và danh sách vi phạm cho giao diện web.
    """
    now_dt = datetime.now()
    time_str = now_dt.strftime("%H:%M:%S")
    uid = uuid.uuid4().hex[:6]
    file_name = f"violation_{now_dt.strftime('%Y%m%d_%H%M%S')}_{uid}.jpg"
    abs_path = os.path.join(SCREENSHOTS_DIR, file_name)
    cv2.imwrite(abs_path, frame)

    with open(log_filename, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([now_dt.strftime("%Y-%m-%d %H:%M:%S"), violation_type, f"{duration:.2f}", file_name])

    with state_lock:
        stats["total_violations"] += 1
        if "ĐIỆN THOẠI" in violation_type or "DIEN THOAI" in violation_type:
            stats["phone"] += 1
        elif "Mất mặt" in violation_type or "Mat mat" in violation_type:
            stats["no_face"] += 1
        else:
            stats["head_pose"] += 1

        item = {
            "id": stats["total_violations"],
            "time": time_str,
            "type": violation_type,
            "duration": f"{duration:.1f}s",
            "image_url": f"/screenshots/{file_name}"
        }
        snapshots_history.insert(0, item)
        # Giữ tối đa 50 ảnh gần nhất trong bộ nhớ
        if len(snapshots_history) > 50:
            snapshots_history.pop()


# ==================== 5. TIẾN TRÌNH CAMERA NỀN (BACKGROUND CAMERA WORKER) ====================

class CameraWorker:
    """
    [CỐT LÕI 5]: TIẾN TRÌNH NỀN ĐỌC CAMERA VÀ CHẠY AI (PRODUCER PATTERN)
    - Khởi tạo và giữ camera độc quyền trong một thread nền duy nhất.
    - Xử lý đọc frame, chạy YOLO và MediaPipe 1 lần duy nhất, mã hóa JPEG và lưu vào buffer.
    - Mọi client truy cập web stream chỉ đọc buffer JPEG từ worker này:
      -> Triệt tiêu hoàn toàn lỗi xung đột camera khi mở nhiều tab hoặc F5.
      -> Tiết kiệm tối đa CPU/GPU, tránh chạy AI trùng lặp.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.lock = Lock()
        self.running = True
        self.latest_jpeg = None

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        frame_count = 0
        violation_start = None
        current_violation = None
        has_snapped_current = False
        last_warning_time = 0
        yolo_violation = None
        yolo_bboxes = []

        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1

            with state_lock:
                is_active = stats["is_running"]

            if is_active:
                # 1. Nhận diện vật thể YOLO (mỗi FRAME_SKIP frame)
                if frame_count % FRAME_SKIP == 0:
                    yolo_violation, yolo_bboxes = check_yolo(frame)

                # 2. Phân tích tư thế đầu MediaPipe FaceMesh
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                pitch, yaw, pose_violation = get_head_pose(results.multi_face_landmarks)

                # 3. Theo dõi thời gian vi phạm liên tục
                active_violation = yolo_violation or pose_violation
                now = time.time()
                if active_violation != current_violation:
                    violation_start = now if active_violation else None
                    current_violation = active_violation
                    has_snapped_current = False  # Reset cờ khi hành vi thay đổi

                duration = (now - violation_start) if violation_start else 0.0
                thresh = NO_FACE_THRESHOLD if active_violation == "Mất mặt / Che camera" else VIOLATION_THRESHOLD

                # 4. Chụp đúng 1 ảnh bằng chứng đại diện cho mỗi lần vi phạm (chống spam ảnh)
                is_warning = duration > thresh
                if is_warning and not has_snapped_current:
                    save_snapshot(active_violation, duration, frame)
                    has_snapped_current = True
                    last_warning_time = now
                elif is_warning and (now - last_warning_time > 10.0):
                    # Nếu tiếp tục vi phạm kéo dài trên 10 giây mới chụp thêm 1 ảnh
                    save_snapshot(active_violation, duration, frame)
                    last_warning_time = now

                # 5. Vẽ giao diện tiếng Việt lên khung hình
                if is_warning:
                    status_text = f"CẢNH BÁO: {active_violation} ({duration:.1f}s)"
                    status_color = (0, 0, 255)  # Đỏ
                elif active_violation:
                    status_text = f"Đang chú ý: {active_violation} ({duration:.1f}s)"
                    status_color = (0, 255, 255)  # Vàng
                else:
                    status_text = "BÌNH THƯỜNG"
                    status_color = (0, 255, 0)  # Xanh lá

                angle_text = f"Góc nhìn: Pitch: {pitch}° | Yaw: {yaw}°"
                frame = draw_vietnamese_hud(frame, status_text, status_color, angle_text, yolo_bboxes)
            else:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text((210, 220), "TẠM DỪNG GIÁM SÁT", font=FONT_TITLE, fill=(255, 165, 0))
                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Nén ảnh JPEG chất lượng cao và lưu vào buffer dùng chung
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.lock:
                self.latest_jpeg = buffer.tobytes()

            time.sleep(0.03)  # Giới hạn ~30 FPS

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


# Khởi động camera worker toàn cục
camera_worker = CameraWorker(0)

# ==================== 6. FASTAPI WEB SERVER ====================
app = FastAPI(title="AI Exam Proctoring Demo - Samsung SIC")
app.mount("/screenshots", StaticFiles(directory=SCREENSHOTS_DIR), name="screenshots")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def stream_generator():
    """Generator truyền luồng MJPEG từ buffer của CameraWorker."""
    while True:
        frame_bytes = camera_worker.get_jpeg()
        if frame_bytes is not None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshots")
def get_snapshots():
    with state_lock:
        return JSONResponse(content={
            "snapshots": list(snapshots_history),  # Shallow copy tránh race condition
            "stats": dict(stats)
        })


@app.post("/api/clear")
def clear_snapshots():
    with state_lock:
        snapshots_history.clear()
        stats["total_violations"] = 0
        stats["phone"] = 0
        stats["calculator"] = 0
        stats["head_pose"] = 0
        stats["no_face"] = 0
    return JSONResponse(content={"status": "success", "message": "Cleared all snapshots"})


@app.post("/api/toggle_stream")
def toggle_stream():
    with state_lock:
        stats["is_running"] = not stats["is_running"]
        current_state = stats["is_running"]
    return JSONResponse(content={"status": "success", "is_running": current_state})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)