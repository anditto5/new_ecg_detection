import streamlit as st
from ultralytics import YOLO
import os

def analyze_ecg_results(results, model):
    jumlah_abnormal = sum(
        1 for r in results for box in r.boxes if model.names[int(box.cls[0])] == "abnormal"
    )

    if len(results[0].boxes) == 0:
        status = "✅ Detak jantung sehat (Normal)"
    elif jumlah_abnormal > 10:
        status = "⚠️ Detak jantung abnormal, perlu istirahat dan konsultasi dokter"
    elif jumlah_abnormal > 0:
        status = "⚠️ Detak jantung abnormal ringan, kurangi merokok, makan sehat, dan tambahkan waktu olahraga"
    else:
        status = "⚠️ Deteksi muncul tapi bukan label 'abnormal'"

    return status, jumlah_abnormal


st.title("🫀 ECG Detection with YOLO11")

model_path = st.text_input("Model path (.pt)", "runs/yolo11s/last.pt")

uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_ecg.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan sebelum deteksi
    st.subheader("📷 Sebelum Deteksi")
    st.image(temp_path, caption="Uploaded ECG", use_container_width=True)

    # Load YOLO model
    model = YOLO(model_path)
    results = model(temp_path, conf=0.3, save=True)  # save hasil deteksi

    # Analisis
    status, jumlah_abnormal = analyze_ecg_results(results, model)

    # Ambil path gambar hasil YOLO
    result_image = results[0].plot()  # hasil sebagai array (BGR)
    import cv2
    import tempfile
    from PIL import Image

    # Simpan hasil ke file sementara
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_out.name, result_image)

    # Tampilkan sesudah deteksi
    st.subheader("✅ Sesudah Deteksi")
    st.image(temp_out.name, caption="Hasil Deteksi YOLO", use_container_width=True)

    # Hasil analisis
    st.subheader("📊 Hasil Analisis")
    st.write(status)
    st.write(f"Jumlah abnormal terdeteksi: **{jumlah_abnormal}**")