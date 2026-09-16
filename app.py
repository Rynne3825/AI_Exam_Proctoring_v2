import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from datetime import datetime
from ultralytics import YOLO

# ==================== CẤU HÌNH ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "screenshots"), exist_ok=True)

VIOLATION_THRESHOLD = 2.0   # Giây vi phạm liên tục trước khi cảnh báo
NO_FACE_THRESHOLD = 3.0     # Giây mất mặt trước khi cảnh báo
YAW_THRESHOLD = 25          # Quay hẳn ra sau (độ)
YAW_SIDE_THRESHOLD = 10     # Quay sang trái/phải (độ)
PITCH_THRESHOLD = 10        # Cúi hoặc ngửa (độ)
YOLO_CONFIDENCE = 0.5       # Ngưỡng tin cậy của YOLO
FRAME_SKIP = 5              # Quét YOLO mỗi 5 frame để giảm tải CPU
CAM_W, CAM_H = 640, 480

# ponytail: Ma trận camera chuẩn cho 640x480. Nâng cấp: cập nhật ma trận nếu đổi độ phân giải.
CAM_MATRIX = np.array([[CAM_W, 0, CAM_W / 2], [0, CAM_W, CAM_H / 2], [0, 0, 1]], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)
FACE_KEYPOINTS = [1, 199, 33, 263, 61, 291]

# ==================== MÔ HÌNH & CAMERA ====================
YOLO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
if not os.path.exists(YOLO_PATH):
    YOLO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "train", "weights", "best.pt")

print("⏳ Đang tải mô hình...")
yolo_model = YOLO(YOLO_PATH)
face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
print("✅ Tải mô hình thành công!")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam. Kiểm tra kết nối.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# ==================== CÁC HÀM XỬ LÝ ====================
def log_violation(violation_type, duration, frame):
    """Ghi vi phạm vào file CSV và lưu ảnh chụp bằng chứng."""
    now_dt = datetime.now()
    timestamp = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    img_name = f"{timestamp.replace(':', '-')}.jpg"
    img_path = os.path.join(BASE_DIR, "screenshots", img_name)
    cv2.imwrite(img_path, frame)

    csv_file = os.path.join(LOGS_DIR, f"violations_{now_dt.strftime('%Y%m%d')}.csv")
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Thời gian', 'Loại vi phạm', 'Thời gian kéo dài (giây)', 'Ảnh chụp'])
        writer.writerow([timestamp, violation_type, f"{duration:.2f}", img_name])
    print(f"📝 Log: {timestamp} - {violation_type} ({duration:.2f}s)")



def get_head_pose(landmarks):
    """Tính pitch, yaw và loại lỗi tư thế từ MediaPipe landmarks."""
    if not landmarks:
        return 0, 0, "Mat mat / Che camera"

    pts = landmarks[0].landmark
    pts_2d = np.array([[int(pts[i].x * CAM_W), int(pts[i].y * CAM_H)] for i in FACE_KEYPOINTS], dtype=np.float64)
    pts_3d = np.array([[int(pts[i].x * CAM_W), int(pts[i].y * CAM_H), pts[i].z] for i in FACE_KEYPOINTS], dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(pts_3d, pts_2d, CAM_MATRIX, DIST_COEFFS)
    if not success:
        return 0, 0, None

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    pitch, yaw = int(angles[0] * 360), int(angles[1] * 360)

    violation = None
    if abs(yaw) > YAW_THRESHOLD:
        violation = "Quay han ra sau"
    elif yaw > YAW_SIDE_THRESHOLD:
        violation = "Quay sang trai"
    elif yaw < -YAW_SIDE_THRESHOLD:
        violation = "Quay sang phai"
    elif pitch > PITCH_THRESHOLD:
        violation = "Ngua mat len"
    elif pitch < -PITCH_THRESHOLD:
        violation = "Cui xuong"

    return pitch, yaw, violation


def check_yolo(frame):
    """Quét điện thoại / máy tính cầm tay bằng YOLO."""
    violation, bboxes = None, []
    results = yolo_model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) <= YOLO_CONFIDENCE:
                continue
            name = yolo_model.names[int(box.cls[0])].lower()
            coords = list(map(int, box.xyxy[0]))
            if "phone" in name:
                violation = "Su dung DIEN THOAI"
                bboxes.append((coords, "PHONE (CAM)", (0, 0, 255)))
            elif "calculator" in name:
                bboxes.append((coords, "CALCULATOR (HOP LE)", (0, 255, 0)))
    return violation, bboxes


def main():
    print("🎥 Đang mở webcam. Nhấn 'q' để thoát.")
    frame_count = 0
    violation_start = None
    current_violation = None
    last_warning_time = 0
    yolo_violation = None
    yolo_bboxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # 1. Nhận diện vật thể YOLO
        if frame_count % FRAME_SKIP == 0:
            yolo_violation, yolo_bboxes = check_yolo(frame)
        for (x1, y1, x2, y2), label, color in yolo_bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. Nhận diện tư thế đầu MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        pitch, yaw, pose_violation = get_head_pose(results.multi_face_landmarks)

        # 3. Đánh giá vi phạm và bấm giờ
        active_violation = yolo_violation or pose_violation
        now = time.time()
        if active_violation != current_violation:
            violation_start = now if active_violation else None
            current_violation = active_violation

        duration = (now - violation_start) if violation_start else 0.0
        thresh = NO_FACE_THRESHOLD if active_violation == "Mat mat / Che camera" else VIOLATION_THRESHOLD

        # 4. Ghi log nếu vượt ngưỡng
        is_warning = duration > thresh
        if is_warning and (now - last_warning_time > 1.0):
            log_violation(active_violation, duration, frame)
            last_warning_time = now

        # 5. Hiển thị thông tin lên khung hình
        if is_warning:
            status_text, status_color = f"CANH BAO: {active_violation} ({duration:.1f}s)", (0, 0, 255)
        elif active_violation:
            status_text, status_color = f"Dang chu y: {active_violation} ({duration:.1f}s)", (0, 255, 255)
        else:
            status_text, status_color = "BINH THUONG", (0, 255, 0)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Pitch: {pitch} | Yaw: {yaw}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

        cv2.imshow("AI Exam Proctoring", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📊 Log đã lưu tại: {log_filename}")


if __name__ == "__main__":
    main()