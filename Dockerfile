FROM python:3.10-slim

WORKDIR /app

# 必要なシステムライブラリ
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# あなたの独自モデルをコピー
COPY best.pt /app/best.pt

# Flaskサーバー
COPY server.py .

CMD ["python", "server.py"]
