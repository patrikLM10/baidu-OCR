import streamlit as st
import subprocess
import re
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd

LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent.parent
MTMD_CLI = LLAMA_CPP_DIR / "build" / "bin" / "llama-mtmd-cli.exe"
MODEL = LLAMA_CPP_DIR / "gguf_models" / "baidu" / "unlimited-ocr-Q4_K_M.gguf"
MMPROJ = LLAMA_CPP_DIR / "gguf_models" / "baidu" / "mmproj-unlimited-ocr-bf16.gguf"

DET_RE = re.compile(r'<\|det\|>(\w+)\s*\[([^\]]+)\]\s*<\|/det\|>(.*?)(?=(?:<\|det\|>|$))', re.DOTALL)

st.set_page_config(page_title="Unlimited OCR", layout="wide")
st.title("Unlimited OCR")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Input", use_container_width=True)

    if st.button("Run OCR", type="primary"):
        with st.spinner("Running OCR (this may take a minute)..."):
            tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_img.write(uploaded.getvalue())
            tmp_img.close()

            result = subprocess.run(
                [
                    str(MTMD_CLI),
                    "-m", str(MODEL),
                    "--mmproj", str(MMPROJ),
                    "--image", tmp_img.name,
                    "-p", "document parsing.",
                    "--chat-template", "deepseek-ocr",
                    "--no-jinja",
                    "--temp", "0",
                    "--flash-attn", "off",
                    "--no-warmup",
                    "-n", "4096",
                    "-c", "16384",
                ],
                capture_output=True, text=True, timeout=600
            )
            os.unlink(tmp_img.name)
            raw_text = result.stdout + result.stderr

        st.success("Done")

        col1, col2 = st.columns([1, 1])

        with col1:
            img = Image.open(uploaded)
            draw = ImageDraw.Draw(img)
            for m in DET_RE.finditer(raw_text):
                label = m.group(1)
                coords = [int(x) for x in m.group(2).split(",")]
                if len(coords) == 4:
                    draw.rectangle(coords, outline="#00FF00", width=3)
                    draw.text((coords[0] + 2, coords[1] + 2), label, fill="#00FF00")
            st.image(img, caption="Detections", use_container_width=True)

        with col2:
            tab1, tab2 = st.tabs(["Raw Output", "Structured Output"])
            with tab1:
                st.text_area("Raw OCR Output", raw_text, height=600)
            with tab2:
                rows = []
                for m in DET_RE.finditer(raw_text):
                    label = m.group(1)
                    coords = [int(x) for x in m.group(2).split(",")]
                    text = m.group(3).strip()
                    rows.append({"Label": label, "BBox": f"[{', '.join(map(str, coords))}]", "Text": text})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No structured detections found.")
