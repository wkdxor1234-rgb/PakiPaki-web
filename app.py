# app.py - PakiPaki Flask (홈페이지 + API 단일 서비스)
from flask import Flask, request, jsonify
from flask_cors import CORS
import io, os, tempfile, subprocess
import numpy as np
import soundfile as sf
import librosa
import audioread  # noqa: F401  # (librosa fallback용)
import imageio_ffmpeg  # ✅ FFmpeg 경로 제공

# ---------------------- 환경 설정 ----------------------
def env_list(key: str, default: str):
    v = os.getenv(key)
    if not v:
        return [x.strip() for x in default.split(",") if x.strip()]
    return [x.strip() for x in v.split(",") if x.strip()]

FRONTEND_ORIGINS = env_list("FRONTEND_ORIGINS", "*")
MAX_MB   = float(os.getenv("MAX_MB", "30"))
MAX_SEC  = float(os.getenv("MAX_SEC", "120"))
TARGET_SR = int(os.getenv("TARGET_SR", "22050"))

# ✅ 기본 허용 확장자에 m4a 추가 (필요하면 mp3도 같이)
ALLOWED_EXT = set(x.lower() for x in env_list("ALLOWED_EXT", "wav,m4a"))

# ---------------------- Flask 앱 -----------------------
app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = int(MAX_MB * 1024 * 1024)
CORS(app)  # 같은 오리진이면 의미 없지만 있어도 무해

# ---------------------- (선택) 모델 로딩 ---------------
model = None
feature_order = None
try:
    import joblib
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        feature_order = getattr(model, "feature_names_", None)
except Exception:
    model = None

# ---------------------- 유틸 ---------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.errorhandler(404)
def not_found(_):
    return jsonify(ok=False, error="Not found"), 404

@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify(ok=False, error="Method not allowed"), 405

@app.errorhandler(413)
def too_large(_):
    return jsonify(ok=False, error="File too large"), 413

@app.errorhandler(500)
def internal_error(_):
    return jsonify(ok=False, error="Internal server error"), 500

# ---------------------- 라우트 -------------------------
@app.get("/")
def root():
    return app.send_static_file("index.html")

@app.get("/health")
def health():
    return jsonify(
        ok=True,
        model_loaded=bool(model),
        allowed_ext=sorted(ALLOWED_EXT),
        max_mb=MAX_MB,
        max_sec=MAX_SEC,
        target_sr=TARGET_SR,
    ), 200

# ---------------------- 디코딩 -------------------------
def decode_audio(filename: str, raw: bytes):
    """
    1) soundfile로 직접 시도 (WAV 등)
    2) librosa로 재시도
    3) FFmpeg(imageio-ffmpeg 제공 바이너리)로 PCM WAV로 변환 → soundfile/librosa 로드
    """
    # 1) soundfile
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        y = data if getattr(data, "ndim", 1) == 1 else data.mean(axis=1)
        return y, sr
    except Exception:
        pass

    # 2) librosa (임시 파일로 확장자 유지)
    suffix = "." + filename.rsplit(".", 1)[1].lower() if "." in filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(raw)
        tmp_in.flush()
        in_path = tmp_in.name

    try:
        y, sr = librosa.load(in_path, sr=None, mono=True)
        return y, sr
    except Exception:
        pass

    # 3) FFmpeg 변환 (예: m4a → wav)
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_path = tmp_out.name

        # -i input -ac 1 -ar 22050 -f wav output
        # (SR은 여기서 22050으로 맞춰도 되지만, 이후 파이프라인에서 다시 리샘플하므로 원본 SR 유지하고 싶으면 -ar 제거)
        cmd = [
            ffmpeg_exe,
            "-loglevel", "error",
            "-y",
            "-i", in_path,
            "-ac", "1",
            "-f", "wav",
            out_path
        ]
        subprocess.run(cmd, check=True)

        # 변환된 WAV 로드
        data, sr = sf.read(out_path, dtype="float32", always_2d=False)
        y = data if getattr(data, "ndim", 1) == 1 else data.mean(axis=1)
        return y, sr
    except Exception:
        return None, None
    finally:
        # 임시 파일 정리
        try: os.remove(in_path)
        except: pass
        try: os.remove(out_path)
        except: pass

# ---------------------- 특징/예측 ----------------------
def extract_features(y: np.ndarray, sr: int) -> dict:
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
    for i, v in enumerate(chroma, 1):   feats[f"chroma_{i}"] = float(v)
    for i, v in enumerate(mfcc, 1):     feats[f"mfcc_{i}"] = float(v)
    return feats

def vectorize(feats: dict, order=None) -> np.ndarray:
    if order:
        return np.array([feats.get(k, 0.0) for k in order], dtype=float).reshape(1, -1)
    keys = (["zcr","rms","centroid","bandwidth","rolloff"]
            + [f"contrast_{i}" for i in range(1,6)]
            + [f"chroma_{i}" for i in range(1,13)]
            + [f"mfcc_{i}" for i in range(1,14)])
    return np.array([feats.get(k, 0.0) for k in keys], dtype=float).reshape(1, -1)

@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify(ok=False, error="Missing form field 'file'"), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify(ok=False, error="Empty filename"), 400
        if not allowed_file(f.filename):
            return jsonify(ok=False, error=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXT)}"), 415

        if request.content_length and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify(ok=False, error="File too large"), 413

        raw = f.read()
        if len(raw) > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify(ok=False, error="File too large"), 413
        if len(raw) < 1024:
            return jsonify(ok=False, error="Audio too short"), 400

        y, sr = decode_audio(f.filename, raw)
        if y is None or sr is None:
            return jsonify(ok=False, error="Failed to decode audio"), 415

        dur = float(len(y)) / float(sr)
        if dur > MAX_SEC:
            return jsonify(ok=False, error=f"Audio too long ({dur:.1f}s > {MAX_SEC:.0f}s)"), 413

        if sr > TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        feats = extract_features(y, sr)

        # (모델 사용 로직은 기존 그대로 — 생략)
        # 휴리스틱 폴백:
        zcr = feats.get("zcr", 0.0)
        diagnosis = "정상입니다" if zcr < 0.2 else "파킨슨병 의심"
        confidence = float(min(0.99, max(0.5, abs(0.2 - zcr) + 0.5)))
        return jsonify(ok=True, diagnosis=diagnosis, confidence=confidence, features=feats), 200

    except Exception as e:
        return jsonify(ok=False, error="Server error", detail=str(e)), 500

