import asyncio
import os
import re
import subprocess
import time
from pathlib import Path

from PIL import Image
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "GLM-OCR-Q8_0.gguf")
MMPROJ_PATH = os.path.join(os.path.dirname(__file__), "models", "mmproj-GLM-OCR-Q8_0.gguf")
LLAMA_CLI = os.path.join(os.path.dirname(__file__), "llama", "llama-cli.exe")
SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class OCRApp(App):
    TITLE = "GLM-OCR TUI"

    BINDINGS = [("ctrl+c", "cancel_ocr", "Stop OCR")]

    _proc = None

    CSS = """
    Screen {
        background: $surface;
    }

    #main {
        padding: 1;
        height: 100%;
    }

    #controls {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }

    #image-row {
        height: auto;
    }

    #image-path {
        width: 1fr;
    }

    #browse-btn {
        width: 16;
        margin-left: 1;
    }

    #preview {
        height: 3;
        border: dashed $accent;
        padding: 0 1;
        content-align: center middle;
        margin-top: 1;
    }

    #prompt-input {
        margin-top: 1;
    }

    #run-row {
        height: auto;
        margin-top: 1;
    }

    #max-tokens {
        width: 16;
    }

    #run-btn {
        width: 20;
        margin-left: 1;
    }

    #output-box {
        border: solid $secondary;
        height: 1fr;
        min-height: 5;
    }

    #output-log {
        height: 1fr;
    }

    #status-bar {
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
        content-align: center middle;
    }

    #perf-panel {
        height: auto;
        padding: 1;
        border: solid $success;
        margin-top: 1;
        display: none;
    }

    #perf-panel.visible {
        display: block;
    }

    .perf-grid {
        layout: grid;
        grid-size: 3;
        grid-columns: 1fr 1fr 1fr;
        height: auto;
    }

    .perf-item {
        content-align: center middle;
    }
    """

    def compose(self):
        yield Header()
        with Container(id="main"):
            with Vertical(id="controls"):
                with Horizontal(id="image-row"):
                    yield Input(placeholder="Path to image file...", id="image-path")
                    yield Button("Browse", id="browse-btn", variant="primary")
                yield Static("[dim]No image selected[/]", id="preview")
                yield Input(
                    value="Extract all text from this image.",
                    id="prompt-input",
                )
                with Horizontal(id="run-row"):
                    yield Input(value="512", id="max-tokens")
                    yield Button("Run OCR", id="run-btn", variant="success")
            with Vertical(id="output-box"):
                yield RichLog(id="output-log", markup=True, highlight=True)
            yield Static(id="status-bar")
            with Vertical(id="perf-panel"):
                with Horizontal(classes="perf-grid"):
                    yield Static("Prompt\n[bold]---[/]", id="perf-prompt", classes="perf-item")
                    yield Static("Generation\n[bold]---[/]", id="perf-gen", classes="perf-item")
                    yield Static("Total\n[bold]---[/]", id="perf-total", classes="perf-item")
        yield Footer()

    def on_mount(self):
        self.query_one("#image-path", Input).focus()

    @on(Input.Changed, "#image-path")
    def on_path_changed(self, event: Input.Changed):
        path = event.value.strip()
        preview = self.query_one("#preview")
        if not path or not os.path.isfile(path):
            preview.update("[dim]No image selected[/]")
            return
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED:
            preview.update(f"[red]Unsupported format: {ext}[/]")
            return
        try:
            with Image.open(path) as img:
                w, h = img.size
                preview.update(f"[green]Selected:[/] [bold]{Path(path).name}[/]  ({w}x{h})")
        except Exception:
            preview.update("[red]Failed to open image[/]")

    @on(Button.Pressed, "#browse-btn")
    def on_browse(self):
        self.push_screen(FileBrowser(), self._on_file_selected)

    def _on_file_selected(self, path: str | None):
        if path:
            self.query_one("#image-path", Input).value = path

    @on(Button.Pressed, "#run-btn")
    def on_run(self):
        if self._proc is not None:
            self._cancel_ocr()
            return
        self.run_ocr()

    def action_cancel_ocr(self):
        self._cancel_ocr()

    def _cancel_ocr(self):
        if self._proc is not None:
            self._proc.kill()
            self._proc = None
        run_btn = self.query_one("#run-btn", Button)
        run_btn.label = "Run OCR"
        run_btn.variant = "success"
        self.query_one("#status-bar", Static).update("[yellow]Cancelled[/]")

    @work(exclusive=True)
    async def run_ocr(self):
        img_path = self.query_one("#image-path", Input).value.strip()
        prompt = self.query_one("#prompt-input", Input).value.strip()
        max_tokens = self.query_one("#max-tokens", Input).value.strip()

        if not img_path:
            self.query_one("#status-bar", Static).update("[red]Please select an image[/]")
            return
        if not os.path.isfile(img_path):
            self.query_one("#status-bar", Static).update("[red]Image file not found[/]")
            return
        if not max_tokens.isdigit():
            self.query_one("#status-bar", Static).update("[red]Invalid max tokens[/]")
            return
        if not os.path.exists(MODEL_PATH):
            self.query_one("#status-bar", Static).update("[red]Model not found[/]")
            return

        log = self.query_one("#output-log", RichLog)
        status = self.query_one("#status-bar", Static)
        perf_panel = self.query_one("#perf-panel")
        run_btn = self.query_one("#run-btn", Button)

        log.clear()
        run_btn.label = "Stop"
        run_btn.variant = "error"
        status.update("[yellow]Running OCR...[/]")
        perf_panel.remove_class("visible")

        start = time.time()
        self._proc = None
        try:
            self._proc = await asyncio.create_subprocess_exec(
                LLAMA_CLI,
                "-m", MODEL_PATH,
                "-mm", MMPROJ_PATH,
                "--image", img_path,
                "-p", prompt,
                "-n", max_tokens,
                "-ngl", "0",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.path.dirname(LLAMA_CLI),
            )

            output_lines = []
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(text)
                log.write(text)

            await self._proc.wait()
            elapsed = time.time() - start
            output = "\n".join(output_lines)

            perf_match = re.search(
                r"\[ Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
                output,
            )

            text_match = re.search(
                r">\s*" + re.escape(prompt) + r"\s*\n(.*?)\n\[",
                output,
                re.DOTALL,
            )

            if text_match:
                extracted = text_match.group(1).strip()
                log.clear()
                log.write(f"[bold green]Extracted Text:[/]\n{extracted}")

            if perf_match:
                perf_panel.add_class("visible")
                self.query_one("#perf-prompt").update(
                    f"Prompt\n[bold]{perf_match.group(1)} t/s[/]"
                )
                self.query_one("#perf-gen").update(
                    f"Generation\n[bold]{perf_match.group(2)} t/s[/]"
                )
                self.query_one("#perf-total").update(
                    f"Total\n[bold]{elapsed:.1f}s[/]"
                )
                status.update(f"[green]Completed in {elapsed:.1f}s[/]")
            else:
                status.update(f"[green]Completed in {elapsed:.1f}s[/]")

        except asyncio.CancelledError:
            status.update("[yellow]Cancelled[/]")
        except subprocess.TimeoutExpired:
            status.update("[red]Timed out after 300s[/]")
        except Exception as e:
            status.update(f"[red]Error: {e}[/]")
        finally:
            self._proc = None
            run_btn.label = "Run OCR"
            run_btn.variant = "success"


class FileBrowser(ModalScreen):
    CSS = """
    FileBrowser {
        background: $surface 80%;
    }

    #dialog {
        width: 60;
        height: 80%;
        border: thick $primary;
        padding: 1;
        background: $surface;
    }

    #dir-label {
        height: 1;
        margin-bottom: 1;
    }

    #file-list {
        height: 1fr;
        border: solid $accent;
        margin-bottom: 1;
    }

    #actions {
        height: auto;
        align: center middle;
    }
    """

    def compose(self):
        with Vertical(id="dialog"):
            yield Label("Select an image file", id="dir-label")
            yield ListView(id="file-list")
            with Horizontal(id="actions"):
                yield Button("Select", id="select-btn", variant="success")
                yield Button("Cancel", id="cancel-btn", variant="default")

    def on_mount(self):
        self.current_path = Path.cwd()
        self._populate()

    def _populate(self):
        lv = self.query_one("#file-list", ListView)
        lv.clear()
        lv.append(ListItem(Static("[bold cyan][..] (parent)[/]")))

        try:
            items = sorted(
                self.current_path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            lv.append(ListItem(Static("[red]Permission denied[/]")))
            return

        for p in items:
            if p.is_dir():
                lv.append(ListItem(Static(f"[blue]{p.name}/[/]")))
            elif p.suffix.lower() in SUPPORTED:
                lv.append(ListItem(Static(f"[green]{p.name}[/]")))

    def on_list_view_selected(self, event: ListView.Selected):
        raw = str(event.item.children[0].content)
        text = re.sub(r"\[.*?\]", "", raw).strip()
        if "(parent)" in raw:
            self.current_path = self.current_path.parent
            self._populate()
            return
        full = self.current_path / text.rstrip("/")
        if full.is_dir():
            self.current_path = full
            self._populate()
            return
        if full.is_file() and full.suffix.lower() in SUPPORTED:
            self.dismiss(str(full))

    @on(Button.Pressed, "#select-btn")
    def on_select(self):
        lv = self.query_one("#file-list", ListView)
        if lv.index is not None:
            item = lv.children[lv.index]
            raw = str(item.children[0].content)
            text = re.sub(r"\[.*?\]", "", raw).strip()
            full = self.current_path / text
            if full.is_file() and full.suffix.lower() in SUPPORTED:
                self.dismiss(str(full))

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self):
        self.dismiss(None)

    def key_escape(self):
        self.dismiss(None)


if __name__ == "__main__":
    app = OCRApp()
    app.run()
