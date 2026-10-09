"""Private HTTPS face engine service. Run on a Python-capable server, not InfinityFree."""
import hmac
import os
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from face_engine import engine

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
lock = threading.Lock()  # OpenCV detector input size is mutable.
MAX_BYTES = 16 * 1024 * 1024

@app.get('/health')
def health():
    return {'ready': engine.detector is not None and engine.recognizer is not None}

def execute(payload):
    action = payload.get('action')
    with lock:
        if action == 'test':
            return {'status': 'healthy' if engine.detector and engine.recognizer else 'unavailable',
                    'yunet_loaded': engine.detector is not None,
                    'sface_loaded': engine.recognizer is not None}
        if engine.detector is None or engine.recognizer is None:
            raise HTTPException(503, 'Face models are not loaded.')
        image = payload.get('image')
        # Never expose the CLI's file-path decoding to remote callers.
        if not isinstance(image, str) or not image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,', 'data:image/webp;base64,', 'data:image/jpg;base64,')):
            raise HTTPException(400, 'A base64 image data URL is required.')
        quality = float(payload.get('min_quality', 0.6))
        if not 0 <= quality <= 1:
            raise HTTPException(400, 'Invalid quality threshold.')
        if action == 'enroll':
            return engine.enroll_face(image, min_quality=quality)
        if action == 'scan':
            enrolled = payload.get('enrolled', [])
            threshold = float(payload.get('threshold', 0.7))
            if not isinstance(enrolled, list) or len(enrolled) > 1000 or not 0 <= threshold <= 1:
                raise HTTPException(400, 'Invalid scan parameters.')
            return engine.scan_classroom(image, enrolled, threshold=threshold, min_quality=quality,
                                         duplicate_protection=bool(payload.get('duplicate_protection', True)))
        raise HTTPException(400, 'Unknown action.')

@app.post('/engine')
async def process(request: Request):
    key = os.environ.get('FACE_SERVICE_KEY', '')
    if len(key) < 32:
        raise HTTPException(503, 'Configure a FACE_SERVICE_KEY of at least 32 characters.')
    if not hmac.compare_digest(request.headers.get('x-service-key', ''), key):
        raise HTTPException(403, 'Invalid service key.')
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, 'Image payload too large.')
    import json
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, 'Invalid JSON.')
    if not isinstance(payload, dict):
        raise HTTPException(400, 'Expected a JSON object.')
    try:
        return await run_in_threadpool(execute, payload)
    except (ValueError, TypeError):
        raise HTTPException(400, 'Invalid engine parameters.')
