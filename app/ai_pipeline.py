import os
import torch
from pathlib import Path
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default YOLO Model Path provided by the user
# Normally placed in env vars, but hardcoded as per the user's specific workflow path
YOLO_MODEL_PATH = Path(r".\models\bacteria_yolo_model.pt")

YOLO_MODEL = None
YOLO_LOADED = False
YOLO_LOAD_ERROR = ""

def load_yolo_model() -> None:
    global YOLO_MODEL, YOLO_LOADED, YOLO_LOAD_ERROR
    
    if YOLO is None:
        YOLO_LOAD_ERROR = "Library ultralytics tidak terinstall. Jalankan `pip install ultralytics`"
        print(f"Error: {YOLO_LOAD_ERROR}")
        return

    if not YOLO_MODEL_PATH.exists():
        YOLO_LOAD_ERROR = f"Model YOLO tidak ditemukan: {YOLO_MODEL_PATH}"
        print(f"Error: {YOLO_LOAD_ERROR}")
        return

    try:
        YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
        # Move model to device
        YOLO_MODEL.to(DEVICE)
        YOLO_LOADED = True
        print("Model YOLO berhasil dimuat!")
    except Exception as e:
        YOLO_LOAD_ERROR = f"Gagal load model YOLO: {str(e)}"
        print(f"Error: {YOLO_LOAD_ERROR}")
        YOLO_MODEL = None

def detect_and_crop(image: Image.Image, conf_threshold: float = 0.25):
    """
    Melakukan deteksi bakteri dengan YOLO dan memotong bounding box menjadi gambar terpisah.
    
    Returns: list of dict -> [{'crop': PIL.Image, 'box': [x1, y1, x2, y2], 'confidence': 0.9}, ...]
    """
    if not YOLO_LOADED or YOLO_MODEL is None:
        raise RuntimeError(f"YOLO Belum siap: {YOLO_LOAD_ERROR}")

    # Memastikan format image (RGB lebih aman buat ultralytics yang menggunakan opencv (BGR) di bawah kap)
    img_rgb = image.convert("RGB")
    
    # Menjalankan inference
    results = YOLO_MODEL(img_rgb, conf=conf_threshold)
    
    # Karena input list cuma 1 (image), result is at index 0
    result = results[0]
    boxes = result.boxes
    
    crops = []
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Dapatkan raw xyxy
            b = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy())
            
            # Koordinat integer
            x1, y1, x2, y2 = map(int, b)
            
            # Lakukan pemotongan menggunakan PIL
            cropped_img = img_rgb.crop((x1, y1, x2, y2))
            
            crops.append({
                "crop": cropped_img,
                "box": [x1, y1, x2, y2],
                "confidence": conf
            })
            
    return crops
