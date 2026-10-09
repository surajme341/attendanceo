import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import sys
import json
import base64
import argparse
import numpy as np
import cv2

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

class FaceEngine:
    def __init__(self):
        self.detector = None
        self.recognizer = None
        self._init_models()

    def _init_models(self):
        if not os.path.exists(YUNET_PATH) or not os.path.exists(SFACE_PATH):
            # Attempt to auto-download if missing
            try:
                from download_models import download_models
                download_models()
            except Exception as e:
                print(f"[WARN] Failed to auto-download models: {e}", file=sys.stderr)

        if os.path.exists(YUNET_PATH) and os.path.getsize(YUNET_PATH) > 10000:
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000
                )
            except Exception as e:
                print(f"[WARN] YuNet init error: {e}", file=sys.stderr)

        if os.path.exists(SFACE_PATH) and os.path.getsize(SFACE_PATH) > 10000:
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
            except Exception as e:
                print(f"[WARN] SFace init error: {e}", file=sys.stderr)

    def decode_image(self, img_input):
        """Accepts base64 string, data URL, or file path."""
        if not img_input:
            return None
        
        # Check if file path
        if isinstance(img_input, str) and os.path.exists(img_input) and os.path.isfile(img_input):
            img = cv2.imread(img_input)
            return img

        # Check if base64 or data URL
        if isinstance(img_input, str):
            if "base64," in img_input:
                img_input = img_input.split("base64,")[1]
            try:
                img_bytes = base64.b64decode(img_input)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                return img
            except Exception as e:
                print(f"[ERROR] Base64 decode failed: {e}", file=sys.stderr)
                return None

        return None

    def assess_quality(self, img, face_box=None):
        """Calculates blur score (Laplacian variance), brightness, and resolution."""
        if img is None:
            return 0.0, {"blur": 0, "brightness": 0, "resolution": 0}

        crop = img
        if face_box is not None:
            x, y, w, h = [int(v) for v in face_box[:4]]
            ih, iw = img.shape[:2]
            x, y = max(0, x), max(0, y)
            w, h = min(iw - x, w), min(ih - y, h)
            if w > 10 and h > 10:
                crop = img[y:y+h, x:x+w]

        if crop is None or crop.size == 0:
            crop = img

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        res_score = min(1.0, (crop.shape[0] * crop.shape[1]) / (150.0 * 150.0))

        # Normalized quality score between 0.0 and 1.0
        # Normal sharp face has blur > 50
        blur_norm = min(1.0, blur_score / 120.0)
        # Brightness optimal between 60 and 200
        bright_norm = 1.0 - (abs(brightness - 128.0) / 128.0)
        bright_norm = max(0.0, min(1.0, bright_norm))

        overall_quality = 0.5 * blur_norm + 0.3 * res_score + 0.2 * bright_norm
        metrics = {
            "blur": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "overall": round(overall_quality, 3)
        }
        return overall_quality, metrics

    def detect_faces(self, img, score_threshold=0.5):
        if img is None:
            return []

        h, w = img.shape[:2]
        if self.detector is None:
            # Fallback to OpenCV Haar Cascade if YuNet is not initialized
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            res = []
            for (fx, fy, fw, fh) in faces:
                res.append(np.array([fx, fy, fw, fh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.85]))
            return res

        self.detector.setInputSize((w, h))
        self.detector.setScoreThreshold(score_threshold)
        _, faces = self.detector.detect(img)
        if faces is None:
            return []
        return faces

    def extract_embedding(self, img, face_data):
        if self.recognizer is None or img is None or face_data is None:
            # Fallback deterministic visual feature extraction
            x, y, w, h = [int(v) for v in face_data[:4]]
            ih, iw = img.shape[:2]
            x, y = max(0, x), max(0, y)
            w, h = min(iw - x, w), min(ih - y, h)
            if w <= 0 or h <= 0:
                return np.zeros(128, dtype=np.float32)
            crop = cv2.resize(img[y:y+h, x:x+w], (64, 64))
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            feat = cv2.dct(gray.astype(np.float32))[0:16, 0:8].flatten()
            norm = np.linalg.norm(feat)
            return (feat / (norm + 1e-6)).tolist()

        aligned = self.recognizer.alignCrop(img, face_data)
        feat = self.recognizer.feature(aligned)
        # SFace returns 128-d L2 normalized float32 feature
        feat_norm = feat / (np.linalg.norm(feat) + 1e-7)
        return feat_norm.flatten().tolist()

    def compare_embeddings(self, emb1, emb2):
        """Cosine similarity between two 128-d embeddings (returns 0.0 to 1.0)."""
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        dot = float(np.dot(v1, v2))
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_sim = dot / (norm1 * norm2)
        # Map cosine similarity to 0.0 - 1.0 range
        # SFace cosine similarity typically ranges from 0.36 (different) to 0.70+ (same identity)
        # Rescale so that 0.36 is 0% and 0.75+ is 100%
        scaled = (cos_sim - 0.25) / 0.55
        return float(max(0.0, min(1.0, scaled)))

    def enroll_face(self, img_input, min_quality=0.55):
        """Validates single face for student enrollment."""
        img = self.decode_image(img_input)
        if img is None:
            return {
                "success": False,
                "error_code": "INVALID_IMAGE",
                "message": "Invalid or unreadable image file. Supported formats: JPEG, PNG, WebP."
            }

        faces = self.detect_faces(img, score_threshold=0.6)
        if len(faces) == 0:
            return {
                "success": False,
                "error_code": "NO_FACE_DETECTED",
                "message": "No human face was detected in the uploaded photo. Please ensure a clear, well-lit frontal portrait."
            }

        if len(faces) > 1:
            return {
                "success": False,
                "error_code": "MULTIPLE_FACES_DETECTED",
                "message": f"Multiple faces ({len(faces)}) detected. Face enrollment requires exactly one clear student portrait.",
                "faces_count": len(faces)
            }

        face = faces[0]
        quality, metrics = self.assess_quality(img, face)

        if metrics["blur"] < 25.0:
            return {
                "success": False,
                "error_code": "BLURRY_PHOTO",
                "message": f"Photo is too blurry (sharpness score {metrics['blur']:.1f} < 25.0). Please upload a sharper image.",
                "metrics": metrics
            }

        if metrics["brightness"] < 30.0 or metrics["brightness"] > 235.0:
            return {
                "success": False,
                "error_code": "POOR_LIGHTING",
                "message": f"Poor lighting detected (brightness {metrics['brightness']:.1f}). Please avoid extreme shadows or over-exposure.",
                "metrics": metrics
            }

        if quality < min_quality:
            return {
                "success": False,
                "error_code": "LOW_QUALITY_FACE",
                "message": f"Overall face quality score ({quality:.2f}) is below minimum threshold ({min_quality:.2f}).",
                "metrics": metrics
            }

        # Valid single clear face
        embedding = self.extract_embedding(img, face)
        x, y, w, h = [int(v) for v in face[:4]]
        confidence = float(face[14]) if len(face) > 14 else 0.95

        return {
            "success": True,
            "message": "Face verified and embedding successfully generated.",
            "embedding": embedding,
            "quality_score": quality,
            "metrics": metrics,
            "box": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "confidence": round(confidence, 3)
            }
        }

    def scan_classroom(self, img_input, enrolled_students, threshold=0.70, min_quality=0.35, duplicate_protection=True):
        """Scans a group classroom photo and matches with enrolled students."""
        img = self.decode_image(img_input)
        if img is None:
            return {
                "success": False,
                "error_code": "INVALID_IMAGE",
                "message": "Invalid or unreadable classroom photo."
            }

        detected_faces = self.detect_faces(img, score_threshold=0.45)
        total_detected = len(detected_faces)

        # Parse enrolled student embeddings
        students_by_id = {}
        for s in enrolled_students:
            s_id = s.get("student_id") or s.get("id")
            emb = s.get("face_embedding") or s.get("embedding")
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except Exception:
                    emb = None
            if emb:
                students_by_id[s_id] = {
                    "id": s_id,
                    "roll_no": s.get("roll_no", ""),
                    "name": s.get("full_name") or s.get("name", "Student"),
                    "embedding": emb,
                    "matched": False,
                    "max_sim": 0.0
                }

        face_matches = []
        matched_student_occurrences = {} # student_id -> list of (match_index, sim)

        for idx, face in enumerate(detected_faces):
            fx, fy, fw, fh = [int(v) for v in face[:4]]
            conf = float(face[14]) if len(face) > 14 else 0.85
            
            # Check face size & quality
            quality, _ = self.assess_quality(img, face)
            if quality < min_quality or fw < 20 or fh < 20:
                # Ignore very blurry / sub-pixel artifacts
                continue

            face_emb = self.extract_embedding(img, face)

            # Compare against enrolled students
            best_match_id = None
            best_similarity = 0.0

            for s_id, s_data in students_by_id.items():
                sim = self.compare_embeddings(face_emb, s_data["embedding"])
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_id = s_id

            is_recognized = (best_match_id is not None) and (best_similarity >= threshold)

            box_info = {
                "x": fx,
                "y": fy,
                "width": fw,
                "height": fh,
                "confidence": round(conf, 3)
            }

            if is_recognized:
                cand = students_by_id[best_match_id]
                match_item = {
                    "box": box_info,
                    "student_id": cand["id"],
                    "student_name": cand["name"],
                    "roll_no": cand["roll_no"],
                    "match_confidence": round(best_similarity, 3),
                    "match_percentage": int(round(best_similarity * 100)),
                    "is_recognized": True,
                    "status": "PRESENT"
                }
                face_matches.append(match_item)
                
                # Record for duplicate protection
                if cand["id"] not in matched_student_occurrences:
                    matched_student_occurrences[cand["id"]] = []
                matched_student_occurrences[cand["id"]].append((len(face_matches) - 1, best_similarity))
            else:
                # Unknown / unassigned face
                match_item = {
                    "box": box_info,
                    "student_id": None,
                    "student_name": "Unknown",
                    "roll_no": "N/A",
                    "match_confidence": round(best_similarity, 3) if best_similarity > 0 else 0.0,
                    "match_percentage": int(round(best_similarity * 100)) if best_similarity > 0 else 0,
                    "is_recognized": False,
                    "status": "ABSENT"
                }
                face_matches.append(match_item)

        # Apply Duplicate Protection:
        # If the same student appears multiple times, mark only the one with the highest confidence as recognized
        if duplicate_protection:
            for s_id, occs in matched_student_occurrences.items():
                if len(occs) > 1:
                    # Sort descending by similarity
                    occs.sort(key=lambda x: x[1], reverse=True)
                    # Keep highest [0], turn rest [1:] into unknown
                    for match_idx, _ in occs[1:]:
                        dup_item = face_matches[match_idx]
                        dup_item["student_id"] = None
                        dup_item["student_name"] = "Unknown (Duplicate)"
                        dup_item["roll_no"] = "N/A"
                        dup_item["is_recognized"] = False
                        dup_item["status"] = "ABSENT"

        # Calculate final counts
        recognized_count = sum(1 for f in face_matches if f["is_recognized"])
        unknown_count = sum(1 for f in face_matches if not f["is_recognized"])

        # Create student result table
        # Track which enrolled students were recognized
        recognized_ids = set(f["student_id"] for f in face_matches if f["is_recognized"])

        students_table = []
        for s in enrolled_students:
            s_id = s.get("student_id") or s.get("id")
            s_name = s.get("full_name") or s.get("name", "Student")
            s_roll = s.get("roll_no", "")

            # Find if this student was matched in face_matches
            matched_entry = next((f for f in face_matches if f["student_id"] == s_id and f["is_recognized"]), None)

            if matched_entry:
                students_table.append({
                    "student_id": s_id,
                    "student_name": s_name,
                    "roll_no": s_roll,
                    "face_match": f"{matched_entry['match_percentage']}%",
                    "attendance": "Present",
                    "is_recognized": True
                })
            else:
                students_table.append({
                    "student_id": s_id,
                    "student_name": s_name,
                    "roll_no": s_roll,
                    "face_match": "No Match",
                    "attendance": "Absent",
                    "is_recognized": False
                })

        # Sort table by roll number
        students_table.sort(key=lambda x: x["roll_no"])

        return {
            "success": True,
            "total_faces": len(face_matches),
            "recognized_faces": recognized_count,
            "unknown_faces": unknown_count,
            "attendance_marked": recognized_count,
            "faces": face_matches,
            "students_table": students_table
        }

# Global engine singleton
engine = FaceEngine()

def run_cli():
    parser = argparse.ArgumentParser(description="ZCOER AI Face Recognition Engine")
    parser.add_argument("action", nargs="?", default="scan", choices=["enroll", "scan", "test"], help="Action to execute")
    parser.add_argument("--image", help="Path to image file or base64 data")
    parser.add_argument("--enrolled", help="JSON string or path to enrolled students file")
    parser.add_argument("--threshold", type=float, default=0.70, help="Matching similarity threshold")
    parser.add_argument("--min-quality", type=float, default=0.55, help="Minimum quality score")
    parser.add_argument("--json", action="store_true", help="Read payload from stdin")

    try:
        args = parser.parse_args()
    except Exception as e:
        sys.stderr.write(f"[FaceEngine] CLI arg parse error: {e}\n")
        print(json.dumps({"success": False, "error_code": "INVALID_ARGS", "message": str(e)}))
        sys.exit(0)

    if args.json:
        try:
            stdin_data = sys.stdin.read()
            if not stdin_data.strip():
                print(json.dumps({"success": False, "error_code": "EMPTY_STDIN", "message": "Received empty stdin payload."}))
                sys.exit(0)
            payload = json.loads(stdin_data)
            args.action = payload.get("action", args.action)
            args.image = payload.get("image", args.image)
            args.enrolled = payload.get("enrolled", args.enrolled)
            args.threshold = float(payload.get("threshold", args.threshold))
            args.min_quality = float(payload.get("min_quality", args.min_quality))
        except Exception as e:
            sys.stderr.write(f"[FaceEngine] Failed to parse JSON stdin: {e}\n")
            print(json.dumps({"success": False, "error_code": "INVALID_JSON_STDIN", "message": f"Invalid JSON stdin: {e}"}))
            sys.exit(0)

    try:
        if args.action == "test":
            print(json.dumps({
                "status": "healthy",
                "yunet_loaded": engine.detector is not None,
                "sface_loaded": engine.recognizer is not None
            }))
            sys.exit(0)

        if args.action == "enroll":
            res = engine.enroll_face(args.image, min_quality=args.min_quality)
            print(json.dumps(res))
            sys.exit(0)

        if args.action == "scan":
            enrolled_list = []
            if args.enrolled:
                if isinstance(args.enrolled, list):
                    enrolled_list = args.enrolled
                elif os.path.exists(args.enrolled):
                    with open(args.enrolled, "r", encoding="utf-8") as f:
                        enrolled_list = json.load(f)
                else:
                    try:
                        enrolled_list = json.loads(args.enrolled)
                    except Exception:
                        enrolled_list = []
            
            res = engine.scan_classroom(args.image, enrolled_list, threshold=args.threshold, min_quality=args.min_quality)
            print(json.dumps(res))
            sys.exit(0)

        print(json.dumps({"success": False, "error_code": "UNKNOWN_ACTION", "message": f"Unknown action: {args.action}"}))
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({
            "success": False,
            "error_code": "ENGINE_ERROR",
            "message": f"FaceEngine runtime error: {str(e)}"
        }))
        sys.exit(0)

if __name__ == "__main__":
    run_cli()
