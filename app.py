import streamlit as st
import subprocess
import tempfile
import os
import re
import time

st.set_page_config(page_title="GLM-OCR", page_icon="📄", layout="centered")

st.title("📄 GLM-OCR")
st.caption("Local OCR using ggml-org/GLM-OCR-GGUF (Q8_0)")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "GLM-OCR-Q8_0.gguf")
MMPROJ_PATH = os.path.join(os.path.dirname(__file__), "models", "mmproj-GLM-OCR-Q8_0.gguf")
LLAMA_CLI = os.path.join(os.path.dirname(__file__), "llama", "llama-cli.exe")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Run the download script first.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

col1, col2 = st.columns([1, 3])
with col1:
    max_tokens = st.number_input("Max tokens", min_value=32, value=512, step=32)
with col2:
    prompt = st.text_input("Prompt", value="Extract all text from this image.")

if uploaded_file:
    st.image(uploaded_file, caption="Input image", width="stretch")

    if st.button("Run OCR", type="primary"):
        with st.spinner("Running OCR..."):
            ext = os.path.splitext(uploaded_file.name)[1] or ".png"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                start = time.time()
                result = subprocess.run(
                    [
                        LLAMA_CLI,
                        "-m", MODEL_PATH,
                        "-mm", MMPROJ_PATH,
                        "--image", tmp_path,
                        "-p", prompt,
                        "-n", str(max_tokens),
                        "-ngl", "0",
                    ],
                    capture_output=True, text=True, timeout=300,
                    cwd=os.path.dirname(LLAMA_CLI)
                )
                elapsed = time.time() - start

                output = result.stdout

                perf_match = re.search(
                    r'\[ Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]',
                    output
                )

                text_match = re.search(
                    r'>\s*' + re.escape(prompt) + r'\s*\n(.*?)\n\[',
                    output, re.DOTALL
                )

                st.subheader("Extracted Text")
                if text_match:
                    extracted = text_match.group(1).strip()
                    st.text_area("Result", extracted, height=200)
                else:
                    st.text_area("Raw output", output, height=200)

                if perf_match:
                    st.subheader("Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prompt Speed", f"{perf_match.group(1)} t/s")
                    col2.metric("Generation Speed", f"{perf_match.group(2)} t/s")
                    col3.metric("Total Time", f"{elapsed:.1f}s")
                else:
                    st.info(f"Completed in {elapsed:.1f}s")

            except subprocess.TimeoutExpired:
                st.error("Timed out after 300s")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)
else:
    st.info("Upload an image to get started.")
