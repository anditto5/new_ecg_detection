# YOLO training pipeline
from ultralytics import YOLO

def train_yolo(
    data="config/data.yaml",
    model="yolo11n.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=50,
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.01,
    project="runs/train",
    name="exp",
):
    """
    Train a YOLO11 model on the ECG dataset.

    Args:
        data (str): Path to data.yaml
        model (str): Base model (e.g., 'yolo11n.pt', 'yolo11s.pt', 'yolo11l.pt')
        epochs (int): Training epochs
        imgsz (int): Image size
        batch (int): Batch size
        patience (int): Early stopping patience
        optimizer (str): Optimizer ('SGD', 'Adam', 'AdamW')
        lr0 (float): Initial learning rate
        lrf (float): Final learning rate fraction
        project (str): Folder to save results
        name (str): Experiment name

    Returns:
        results (dict): Training results
    """
    model = YOLO(model)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        project=project,
        name=name,
    )

    print("✅ Training complete. Results saved in:", f"{project}/{name}")
    return results


if __name__ == "__main__":
    # Example quick run
    train_yolo(
        data="config/data.yaml",
        model="yolo11n.pt",
        epochs=5,
        imgsz=320,
        batch=4,
        name="debug_run"
    )
