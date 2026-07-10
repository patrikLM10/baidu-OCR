import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk, ImageDraw
import subprocess
import threading
import re
import os
from pathlib import Path

LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent.parent
MTMD_CLI = LLAMA_CPP_DIR / "build" / "bin" / "llama-mtmd-cli.exe"
MODEL = LLAMA_CPP_DIR / "gguf_models" / "baidu" / "unlimited-ocr-Q4_K_M.gguf"
MMPROJ = LLAMA_CPP_DIR / "gguf_models" / "baidu" / "mmproj-unlimited-ocr-bf16.gguf"

DET_RE = re.compile(r'<\|det\|>(\w+)\s*\[([^\]]+)\]\s*<\|/det\|>(.*?)(?=(?:<\|det\|>|$))', re.DOTALL)

class OCRUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unlimited OCR")
        self.root.geometry("1200x800")

        self.image_path = None
        self.photo = None
        self.img_pil = None

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(top, text="Select Image", command=self.select_image).pack(side=tk.LEFT, padx=2)
        self.status = tk.Label(top, text="Ready", fg="gray")
        self.status.pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="Run OCR", command=self.run_ocr, bg="#4CAF50", fg="white").pack(side=tk.RIGHT, padx=2)

        panes = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        panes.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

        left = tk.Frame(panes)
        self.canvas = tk.Canvas(left, bg="#1e1e1e", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=1)
        panes.add(left, width=600)

        right = tk.Frame(panes)
        self.output = scrolledtext.ScrolledText(right, wrap=tk.WORD, font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
        self.output.pack(fill=tk.BOTH, expand=1)
        panes.add(right, width=500)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if not path:
            return
        self.image_path = path
        self.img_pil = Image.open(path)
        self.show_image()
        self.status.config(text=f"Loaded: {os.path.basename(path)}", fg="white")

    def show_image(self, boxes=None):
        if self.img_pil is None:
            return
        img = self.img_pil.copy()
        if boxes:
            draw = ImageDraw.Draw(img)
            for label, x1, y1, x2, y2 in boxes:
                draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)
                draw.text((x1 + 2, y1 + 2), label, fill="#00FF00")

        cw = self.canvas.winfo_width() or 600
        ch = self.canvas.winfo_height() or 600
        img.thumbnail((cw, ch), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER)

    def run_ocr(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Select an image first")
            return
        self.output.delete(1.0, tk.END)
        self.status.config(text="Running OCR...", fg="yellow")
        self.root.update()

        def task():
            try:
                result = subprocess.run(
                    [
                        str(MTMD_CLI),
                        "-m", str(MODEL),
                        "--mmproj", str(MMPROJ),
                        "--image", self.image_path,
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
                self.root.after(0, self.show_result, result.stdout + result.stderr)
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self.status.config(text="Timed out (600s)", fg="red"))
            except Exception as e:
                self.root.after(0, lambda: self.status.config(text=f"Error: {e}", fg="red"))

        threading.Thread(target=task, daemon=True).start()

    def show_result(self, text):
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, text)

        boxes = []
        for m in DET_RE.finditer(text):
            label = m.group(1)
            coords = [int(x) for x in m.group(2).split(",")]
            if len(coords) == 4:
                boxes.append((label, *coords))

        if boxes and self.img_pil:
            self.show_image(boxes)

        self.status.config(text="Done", fg="#00FF00")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    ui = OCRUI()
    ui.run()
