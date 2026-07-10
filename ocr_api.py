import subprocess
import re
import tempfile
import os
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent.parent
MTMD_CLI = LLAMA_CPP_DIR / "build" / "bin" / "llama-mtmd-cli.exe"
MODEL = LLAMA_CPP_DIR / "gguf_models" / "baidu" / "unlimited-ocr-Q4_K_M.gguf"
MMPROJ = LLAMA_CPP_DIR / "gguf_models" / "baidu" / "mmproj-unlimited-ocr-bf16.gguf"

DET_RE = re.compile(r'<\|det\|>(\w+)\s*\[([^\]]+)\]\s*<\|/det\|>(.*?)(?=(?:<\|det\|>|$))', re.DOTALL)

app = FastAPI(title="Unlimited OCR API")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(await file.read())
    tmp.close()

    result = subprocess.run(
        [str(MTMD_CLI), "-m", str(MODEL), "--mmproj", str(MMPROJ),
         "--image", tmp.name, "-p", "document parsing.",
         "--chat-template", "deepseek-ocr", "--no-jinja",
         "--temp", "0", "--flash-attn", "off", "--no-warmup",
         "-n", "4096", "-c", "16384"],
        capture_output=True, text=True, timeout=600
    )
    os.unlink(tmp.name)
    raw_text = result.stdout + result.stderr

    detections = []
    for m in DET_RE.finditer(raw_text):
        coords = [int(x) for x in m.group(2).split(",")]
        detections.append({
            "label": m.group(1),
            "bbox": {"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]},
            "text": m.group(3).strip()
        })

    return {"raw": raw_text, "detections": detections}
