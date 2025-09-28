# app.py - PakiPaki Flask API (홈페이지+API 한 서비스)
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io, os, tempfile
import numpy as np
import soundfile as sf
import librosa
import audioread  # noqa: F401 (librosa fallback용)

# ---------- 구성값 ----------
def env_list(key, default):
    v = os.getenv(key)
    if not v:
        return [x.strip() for x in default.split(",") if x.strip()]
    return [x.strip() for x in v.split(",") if x.strip()]

FRONTEND_ORIGINS = env_list("FRONTEND_ORIGINS", "*")  # 같은 오리진이면 신경 안 써도 됨
MAX_MB = float(os.getenv("MAX_MB", "30"))             # 업로드 한도(MB)
ALLOWED_EXT = set([x.lower() for x in env_list("ALLOWED_EXT", "wav")])  # 기본 WAV만

# ---------- 앱 생성 (정적 서빙) ----------
app = Flask(__name__, static_folder="static", static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = int(MAX_MB * 1024 * 1024)

# (같은 오리진이면 불필요하지만, 있어도 무해)
if FRONTEND_ORIGINS == ["*"]:
    CORS(app)
else:
    CORS(app, resources={
        r"/predict": {"origins": FRONTEND_ORIGINS},
        r"/health": {"origins": FRONTEND_ORIGINS},
    })

# ---------- 선택적 모델 로딩 ----------
model = None
feature_order = None
try:
    import joblib
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        feature_order = getattr(model, "feature_names_", None)
except Exception:
    model = None

# ---------- 유틸 ----------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.errorhandler(413)
def too_large(_):
    return jsonify(ok=False, error="File too large"), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify(ok=False, error="Not found"), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify(ok=False, error="Method not allowed"), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify(ok=False, error="Internal server error"), 500

# ---------- 라우트 ----------
@app.get("/")
def root():
    # https://<서비스>/ → static/index.html
    return app.send_static_file("index.html")

@app.get("/health")
def health():
    return jsonify(ok=True, model_loaded=bool(model), allowed_ext=sorted(list(ALLOWED_EXT))), 200

# ---------- 오디오 처리 ----------
def extract_features(y, sr):
    y = librosa.util.fix_length(y, size=max(len(y), sr // 2))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=5, fmin=100.0), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    feats = {"zcr": zcr, "rms": rms, "centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff}
    for i, v in enumerate(contrast, 1): feats[f"contrast_{i}"] = float(v)
    for i, v in enumerate(chroma, 1): feats[f"chroma_{i}"] = float(v)
    for i, v in enumerate(mfcc, 1): feats[f"mfcc_{i}"] = float(v)
    return feats

def vectorize(feats: dict, order=None):
    if order:
        return np.array([feats.get(k, 0.0) for k in order], dtype=float).reshape(1, -1)
    keys = (
        ["zcr","rms","centroid","bandwidth","rolloff"] +
        [f"contrast_{i}" for i in range(1,6)] +
        [f"chroma_{i}" for i in range(1,13)] +
        [f"mfcc_{i}" for i in range(1,14)]
    )
    return np.array([feats.get(k, 0.0) for k in keys], dtype=float).reshape(1, -1)

def decode_audio(filename: str, raw: bytes):
    # 1) soundfile
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        y = data if getattr(data, "ndim", 1) == 1 else data.mean(axis=1)
        return y, sr
    except Exception:
        pass
    # 2) librosa fallback (확장자 유지)
    suffix = "." + filename.rsplit(".", 1)[1].lower() if "." in filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw); tmp.flush()
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        return y, sr
    finally:
        try: os.remove(tmp_path)
        except: pass

# ---------- 예측 ----------
@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify(ok=False, error="Missing form field 'file'"), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify(ok=False, error="Empty filename"), 400
        if not allowed_file(f.filename):
            return jsonify(ok=False, error=f"Unsupported file type. Allowed: {sorted(list(ALLOWED_EXT))}"), 415
        raw = f.read()
        if len(raw) < 1024:
            return jsonify(ok=False, error="Audio too short"), 400

        y, sr = decode_audio(f.filename, raw)
        if y is None or sr is None:
            return jsonify(ok=False, error="Failed to decode audio"), 415

        feats = extract_features(y, sr)

        # 모델 사용
        if model is not None:
            try:
                X = vectorize(feats, feature_order)
                if hasattr(model, "predict_proba"):
                    proba = float(np.max(model.predict_proba(X)))
                    pred = model.predict(X)[0]
                    if isinstance(pred, (int, np.integer)):
                        diagnosis = "파킨슨병 의심" if int(pred) == 1 else "정상입니다"
                    else:
                        s = str(pred).lower()
                        diagnosis = "파킨슨병 의심" if ("parkinson" in s or "pos" in s or s == "1") else "정상입니다"
                    return jsonify(ok=True, diagnosis=diagnosis, confidence=proba, features=feats), 200
                else:
                    pred = model.predict(X)[0]
                    diagnosis = "파킨슨병 의심" if str(pred).lower() in ("1","true","parkinson","positive") else "정상입니다"
                    return jsonify(ok=True, diagnosis=diagnosis, confidence=None, features=feats), 200
            except Exception:
                # 모델 실패 시 휴리스틱으로 진행
                pass

        # 휴리스틱(모델 없음/실패)
        zcr = feats.get("zcr", 0.0)
        diagnosis = "정상입니다" if zcr < 0.2 else "파킨슨병 의심"
        confidence = float(min(0.99, max(0.5, abs(0.2 - zcr) + 0.5)))
        return jsonify(ok=True, diagnosis=diagnosis, confidence=confidence, features=feats), 200

    except Exception as e:
        return jsonify(ok=False, error="Server error", detail=str(e)), 500


    except Exception as e:
        return jsonify(ok=False, error="Server error", detail=str(e)), 500
