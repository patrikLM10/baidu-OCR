# Baidu OCR API

Local OCR API server using **Unlimited-OCR** (DeepSeek-OCR based) GGUF model via llama.cpp.

Extracts text with bounding box detection from images. Designed for AI agents and RAG pipelines.

## Requirements

- Python 3.10+
- Pillow (`pip install Pillow`)
- FastAPI + uvicorn (`pip install fastapi uvicorn python-multipart`)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with PR #24969 + #24975
- GGUF model files (see [sabafallah/Unlimited-OCR-GGUF](https://huggingface.co/sabafallah/Unlimited-OCR-GGUF))

## Setup

Place model files in `llama.cpp/gguf_models/baidu/`:
- `unlimited-ocr-Q4_K_M.gguf` (LM weights)
- `mmproj-unlimited-ocr-bf16.gguf` (vision projector)

Update paths in `ocr_api.py` if your llama.cpp build is elsewhere.

## Usage

### Start API server

```powershell
cd D:\python\ocr\baidu-ocr-api
python -m uvicorn ocr_api:app --host 0.0.0.0 --port 8000
```

### API Endpoints

**POST /ocr** — Upload an image, get structured OCR output

```python
import requests
r = requests.post("http://localhost:8000/ocr", files={"file": open("document.jpg","rb")})
data = r.json()
print(data["detections"])
```

Response format:
```json
{
  "raw": "...full CLI output...",
  "detections": [
    {
      "label": "header",
      "bbox": {"x1": 79, "y1": 139, "x2": 180, "y2": 180},
      "text": "All the News That's Fit to Print"
    }
  ]
}
```

**GET /docs** — Swagger UI for interactive testing.

## Streamlit UI

```powershell
python -m streamlit run ocr_app.py
```

## Tkinter UI

```powershell
python ocr_ui.py
```

## Model

Uses [sabafallah/Unlimited-OCR-GGUF](https://huggingface.co/sabafallah/Unlimited-OCR-GGUF) with a custom llama.cpp build (PR #24969 + #24975).
