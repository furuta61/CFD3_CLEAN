FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -U pip setuptools wheel && pip install -r requirements.txt

RUN python - <<'PY'
import easyocr
easyocr.Reader(['en','ja'], download_enabled=True)
PY

COPY . .

CMD ["bash","-lc","streamlit run app_streamlit_v2.py --server.address 0.0.0.0 --server.port $PORT --server.headless true"]
# ---- 1) ベース（軽量＋Python3.11） ----
FROM python:3.11-slim

# ---- 2) OS依存のOCRライブラリ ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 build-essential git wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- 3) 作業ディレクトリ＆依存 ----
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip setuptools wheel && pip install -r requirements.txt

# EasyOCR のモデルを事前ダウンロード（初回遅延を回避）
RUN python - <<'PY'
import easyocr
easyocr.Reader(['en','ja'], download_enabled=True)
PY

# ---- 4) アプリ本体 ----
COPY . /app

# ---- 5) 実行（Render は $PORT を渡す）----
CMD ["bash","-lc","streamlit run app_streamlit_v2.py --server.address 0.0.0.0 --server.port $PORT --server.headless true"]
