import gradio as gr
from ultralytics import YOLO
import cv2
from PIL import Image

def analyze_ecg_results(results, model):
    jumlah_abnormal = sum(
        1 for r in results for box in r.boxes if model.names[int(box.cls[0])] == "abnormal"
    )
    if len(results[0].boxes) == 0:
        status = "✅ Detak jantung sehat (Normal)"
    elif jumlah_abnormal > 6:
        status = "⚠️ Detak jantung abnormal, perlu istirahat dan konsultasi dokter"
    elif jumlah_abnormal > 0:
        status = "⚠️ Detak jantung abnormal ringan, kurangi merokok, makan sehat, dan tambahkan waktu olahraga"
    else:
        status = "✅ Detak jantung sehat (Normal)"
    return status, jumlah_abnormal

def predict_ecg(image, model_path):
    model = YOLO(model_path)
    results = model(image, conf=0.4)

    status, jumlah_abnormal = analyze_ecg_results(results, model)

    plotted = results[0].plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    plotted_image = Image.fromarray(plotted)

    return plotted_image, f"{status}\nJumlah abnormal terdeteksi: {jumlah_abnormal}"

with gr.Blocks() as demo:
    gr.Markdown("## 🫀 ECG Detection with YOLO11")
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload ECG Image")
        model_input = gr.Textbox(value="runs/yolo11s/last.pt", label="Model Path")
    with gr.Row():
        image_output = gr.Image(type="pil", label="Hasil Deteksi")
        text_output = gr.Textbox(label="Analisis")
    btn = gr.Button("Deteksi Sekarang")
    btn.click(fn=predict_ecg, inputs=[image_input, model_input], outputs=[image_output, text_output])

if __name__ == "__main__":
    demo.launch()
 