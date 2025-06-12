from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import cv2
import numpy as np

from main import measure_all  # fungsi dari main.py

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict-babyheight")
async def predict_height(image: UploadFile = File(...)):
    try:
        # Simpan gambar ke folder sementara
        image_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Baca gambar dari file lokal dan ubah jadi URL-like (pakai path lokal)
        image_url = f"file://{file_path}"

        # Buka gambar dengan cv2, lalu simpan ulang ke buffer jika diperlukan
        image_cv = cv2.imread(file_path)
        if image_cv is None:
            return JSONResponse(status_code=400, content={"error": "Gagal membaca gambar."})

        # Trik: Simpan gambar lokal ke web server kalau `measure_all` kamu hanya bisa handle URL.
        # Tapi kalau kamu mau ganti `download_image` agar juga handle path lokal, bisa lebih bagus.

        # Jalankan pipeline pengukuran panjang bayi
        result_cm = measure_all(file_path)

        # Hapus file setelah diproses (opsional)
        os.remove(file_path)

        if result_cm is None:
            return JSONResponse(status_code=422, content={"error": "Tidak bisa mengukur tinggi bayi dari gambar ini."})

        return {"status": "success", "predicted_height_cm": round(result_cm, 2)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
