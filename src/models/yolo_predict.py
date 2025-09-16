from ultralytics import YOLO

# Load model hasil training
model = YOLO("runs/yolo11s/best.pt")

# Jalankan evaluasi di validation set
metrics = model.val(data="config/data.yaml", imgsz=960, batch=16)

print(metrics)