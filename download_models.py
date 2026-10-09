import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in MODELS.items():
        filepath = os.path.join(MODELS_DIR, name)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
            print(f"[OK] Model {name} already exists ({os.path.getsize(filepath):,} bytes)")
            continue
        print(f"[DOWNLOADING] {name} from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(filepath, "wb") as f:
                content = resp.read()
                f.write(content)
            print(f"[SUCCESS] Downloaded {name} ({len(content):,} bytes)")
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)

def verify_models():
    import cv2
    yunet_path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
    
    if os.path.exists(yunet_path) and os.path.getsize(yunet_path) > 10000:
        try:
            detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
            print("[VERIFIED] YuNet FaceDetector loaded successfully!")
        except Exception as e:
            print(f"[WARN] YuNet load error: {e}")
            
    if os.path.exists(sface_path) and os.path.getsize(sface_path) > 10000:
        try:
            recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
            print("[VERIFIED] SFace FaceRecognizer loaded successfully!")
        except Exception as e:
            print(f"[WARN] SFace load error: {e}")

if __name__ == "__main__":
    download_models()
    verify_models()
