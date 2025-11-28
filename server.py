from flask import Flask, request, jsonify
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from datetime import datetime
import os
import requests

app = Flask(__name__)

# ===========================
# モデルロード
# ===========================
battery_model = YOLO("/app/best.pt")   # ←あなたの独自モデル
other_model = YOLO("yolov8m.pt")       # ←市販 YOLOv8m（自動DL）

# ===========================
# Supabase 設定（Railway の Variables に入れる）
# ===========================
SUPABASE_URL = os.environ.get("https://njlztbylmzysvfmtwweh.supabase.co")
SUPABASE_API_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5qbHp0YnlsbXp5c3ZmbXR3d2VoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTA5NjkwMSwiZXhwIjoyMDc2NjcyOTAxfQ.uUdg3jv-GXSZ9GpC8eULMhW-NxWjCL7VH7kxClaLvkM")
SUPABASE_TABLE = "ai_results"

headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

# ===========================
# Utility: 画像読み込み
# ===========================
def load_image(file):
    return Image.open(BytesIO(file.read())).convert("RGB")

# ===========================
# Battery 判定
# ===========================
def predict_battery(img):
    res = battery_model.predict(img, conf=0.5, verbose=False)
    if len(res[0].boxes) == 0:
        return None
    return "battery"

# ===========================
# その他ゴミ判定
# ===========================
def predict_other(img):
    res = other_model.predict(img, conf=0.3, verbose=False)
    if len(res[0].boxes) == 0:
        return None
    cls = int(res[0].boxes.cls[0])
    return other_model.names[cls]

# ===========================
# Supabaseへ保存（class名のみ）
# ===========================
def push_to_supabase(class_name):
    if not SUPABASE_URL:
        return  # ローカル実行用

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    data = {
        "class": class_name,
        "timestamp": datetime.utcnow().isoformat()
    }
    requests.post(url, json=data, headers=headers)

# ===========================
# メインAPI
# ===========================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "画像がありません"}), 400

    img = load_image(request.files["image"])

    # ① Battery
    b = predict_battery(img)
    if b:
        push_to_supabase("battery")
        return jsonify({"result": "battery"})

    # ② Other YOLO
    o = predict_other(img)
    if o:
        push_to_supabase(o)
        return jsonify({"result": o})

    # ③ 無分類
    push_to_supabase("unknown")
    return jsonify({"result": "unknown"})


# ===========================
# Railway の PORT に対応
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
