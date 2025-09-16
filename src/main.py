import argparse
from ultralytics import YOLO
from models.yolo_trainer import train_yolo


def analyze_ecg_results(results, model):
    """
    Analisis hasil YOLO prediksi ECG (tanpa detail per box).
    """
    jumlah_abnormal = sum(
        1 for r in results for box in r.boxes if model.names[int(box.cls[0])] == "abnormal"
    )

    # Decision logic
    if len(results[0].boxes) == 0:
        print("✅ Detak jantung sehat (Normal)")
    elif jumlah_abnormal > 5:
        print("⚠️ Detak jantung abnormal, perlu istirahat")
    elif jumlah_abnormal > 0:
        print("⚠️ Detak jantung abnormal ringan, kurangi merokok dan lebihkan olahraga")
    else:
        print("⚠️ Deteksi muncul tapi bukan label 'abnormal'")

    print(f"Jumlah abnormal terdeteksi: {jumlah_abnormal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True)
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path model .pt")
    parser.add_argument("--data", type=str, default="config/data.yaml", help="Path data.yaml")
    parser.add_argument("--image", type=str, help="Path ke gambar ECG untuk prediksi")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if args.mode == "predict":
        if not args.image:
            raise ValueError("Harap berikan path gambar ECG dengan --image")

        model = YOLO(args.model)
        results = model(args.image, conf=0.3)
        analyze_ecg_results(results, model)

    elif args.mode == "train":
        train_yolo(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )
