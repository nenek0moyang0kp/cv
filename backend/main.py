from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os
import shutil
import uuid
import subprocess
import math

class Detection(BaseModel):
    bbox: List[float]
    confidence: float
    class_id: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

# Fungsi untuk menyaring deteksi unik berdasarkan class_id dan posisi (dalam grid kasar)
def filter_unique_detections(detections: List[Detection], grid_size: int = 50) -> List[Detection]:
    seen = set()
    filtered = []

    for det in detections:
        x1, y1, _, _ = det.bbox
        grid_x = math.floor(x1 / grid_size)
        grid_y = math.floor(y1 / grid_size)
        key = (det.class_id, grid_x, grid_y)

        if key not in seen:
            seen.add(key)
            filtered.append(det)

    return filtered

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"

    BASE_DIR = Path(__file__).resolve().parent
    uploads_dir = BASE_DIR / "uploads"
    outputs_dir = BASE_DIR / "outputs"
    frontend_public_dir = BASE_DIR.parent / "public" / "outputs"

    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    frontend_public_dir.mkdir(parents=True, exist_ok=True)

    image_path = uploads_dir / unique_name
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Proses VIDEO
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        output_folder_name = f"detected_{uuid.uuid4().hex}"
        output_folder_path = outputs_dir / output_folder_name

        results = model.predict(
            source=str(image_path),
            save=True,
            save_txt=False,
            project=str(outputs_dir),
            name=output_folder_name
        )

        # Ambil semua deteksi dari semua frame
        result_data = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    result_data.append(Detection(
                        bbox=box.xyxy[0].tolist(),
                        confidence=box.conf[0].item(),
                        class_id=box.cls[0].item()
                    ))

        # Filter: hanya deteksi unik (tanpa tracking)
        unique_result_data = filter_unique_detections(result_data)

        # Ambil file video hasil deteksi
        detected_files = list(output_folder_path.glob("*"))
        detected_video = next((f for f in detected_files if f.suffix in [".mp4", ".avi", ".mov", ".mkv"]), None)

        if not detected_video or not detected_video.exists():
            return {"error": "Hasil deteksi tidak ditemukan"}

        # Transcode video
        transcoded_name = f"transcoded_{detected_video.stem}.mp4"
        transcoded_path = frontend_public_dir / transcoded_name
        ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"  # Ganti sesuai path FFmpeg kamu

        ffmpeg_cmd = [
            ffmpeg_path,
            "-i", str(detected_video),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            str(transcoded_path)
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True)
        except subprocess.CalledProcessError as e:
            return {"error": "Gagal melakukan transcoding video", "detail": str(e)}

        # Bersihkan folder YOLO sementara
        shutil.rmtree(output_folder_path, ignore_errors=True)

        return {
            "message": "Video berhasil diproses",
            "output_path": f"/outputs/{transcoded_name}",
            "result": [r.dict() for r in unique_result_data]
        }

    # Proses GAMBAR
    results = model(str(image_path))
    result_data = []
    for box in results[0].boxes:
        result_data.append({
            "bbox": box.xyxy[0].tolist(),
            "confidence": box.conf[0].item(),
            "class_id": box.cls[0].item()
        })

    return {"result": result_data}
