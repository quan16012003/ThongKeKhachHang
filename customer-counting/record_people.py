import cv2
from ultralytics import YOLO
from collections import deque
import pandas as pd
from datetime import datetime
import os

# Tạo thư mục nếu chưa tồn tại
os.makedirs('outputs', exist_ok=True)


# Khởi tạo mô hình YOLO
model = YOLO('yolov8n.pt')

# Kết nối camera
cap = cv2.VideoCapture(0)  # 0 là camera mặc định

# Biến đếm
entry_count = 0
exit_count = 0

# Vị trí đường line dọc
line_position = 320  # Điều chỉnh theo chiều ngang khung hình
line_thickness = 10  # Độ dày đường line

# Theo dõi đối tượng qua ID
object_tracker = {}
track_history = deque(maxlen=100)  # Lưu lịch sử các ID đã được đếm

def save_statistics(entry_count, exit_count):
    """Lưu thống kê vào file CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    statistics = [{'Time': timestamp, 'Entry Count': entry_count, 'Exit Count': exit_count}]
    df = pd.DataFrame(statistics)
    df.to_csv('outputs/statistics.csv', mode='a', header=False, index=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Phân tích frame
    results = model.track(source=frame, persist=True, conf=0.5)

    for box in results[0].boxes:
        cls = int(box.cls[0])  # Lớp của đối tượng
        if cls == 0:  # Chỉ đếm "person"
            # Lấy thông tin bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            obj_id = box.id.item() if hasattr(box, "id") and box.id is not None else None

            if obj_id:
                if obj_id not in object_tracker:
                    # Thêm đối tượng mới vào tracker
                    object_tracker[obj_id] = (center_x, center_y)
                else:
                    # Kiểm tra hướng di chuyển
                    prev_x, prev_y = object_tracker[obj_id]

                    if prev_x < line_position <= center_x:  # Qua từ trái sang phải
                        if obj_id not in track_history:
                            entry_count += 1
                            track_history.append(obj_id)
                    elif prev_x > line_position >= center_x:  # Qua từ phải sang trái
                        if obj_id not in track_history:
                            exit_count += 1
                            track_history.append(obj_id)

                # Cập nhật vị trí
                object_tracker[obj_id] = (center_x, center_y)

    # Vẽ đường line dọc và hiển thị số liệu
    cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (255, 0, 0), line_thickness)
    cv2.putText(frame, f"Entry Count: {entry_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit Count: {exit_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("Frame", frame)

    # Lưu dữ liệu định kỳ
    if len(track_history) % 10 == 0:
        save_statistics(entry_count, exit_count)

    # Dừng nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lưu dữ liệu cuối cùng
save_statistics(entry_count, exit_count)

cap.release()
cv2.destroyAllWindows()