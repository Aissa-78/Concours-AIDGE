"""Interface locale pour choisir une vidéo, régler une porte et compter."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from calibrate_gate import POINT_LABELS, save_config


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR = PROJECT_DIR / "runs" / "count_video"
PREVIEW_SIZE = (840, 430)


def video_frame(path: Path):
    """Lire une image représentative, ou la première image si nécessaire."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Cette vidéo ne peut pas être lue. Essaie un fichier MP4.")
    try:
        if capture.get(cv2.CAP_PROP_FRAME_COUNT) > 50:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            raise ValueError("Aucune image lisible dans cette vidéo.")
        return frame
    finally:
        capture.release()


def points_from_config(path: Path, width: int, height: int) -> list[tuple[int, int]]:
    """Afficher un réglage existant sur la vidéo choisie."""
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        points = [*data["gate"]["line_1"], *data["gate"]["line_2"], data["inside_reference"]]
        if len(points) != 5:
            raise ValueError
        for point in points:
            if len(point) != 2 or not all(0 <= float(value) <= 1 for value in point):
                raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Ce fichier ne contient pas un réglage de porte valide.") from error
    return [(round(float(x) * width), round(float(y) * height)) for x, y in points]


def open_file(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)])


class FlowSenseApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("FlowSense — comptage de passages")
        root.geometry("950x740")
        root.minsize(650, 610)

        self.video_path: Path | None = None
        self.config_path: Path | None = None
        self.result_path: Path | None = None
        self.points: list[tuple[int, int]] = []
        self.frame_width = self.frame_height = 0
        self.display_width = self.display_height = 0
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.running = False
        self.cancel_requested = False
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, int | str, str | None]] = queue.Queue()

        self.video_text = tk.StringVar(value="Aucune vidéo sélectionnée")
        self.config_text = tk.StringVar(value="Aucun réglage de porte")
        self.instruction_text = tk.StringVar(value="Choisis d'abord une vidéo.")
        self.status_text = tk.StringVar(value="Prêt")
        self.result_text = tk.StringVar(value="Entrées : —    Sorties : —")

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="FlowSense", font=("Helvetica", 21, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Choisis une vidéo, indique le passage, puis lance le comptage.").pack(
            anchor="w", pady=(0, 12)
        )

        selection = ttk.Frame(outer)
        selection.pack(fill="x", pady=(0, 6))
        self.choose_button = ttk.Button(selection, text="1. Choisir une vidéo", command=self.choose_video)
        self.choose_button.pack(side="left")
        ttk.Label(selection, textvariable=self.video_text).pack(side="left", padx=12)

        configuration = ttk.Frame(outer)
        configuration.pack(fill="x", pady=(0, 10))
        self.load_button = ttk.Button(configuration, text="Réutiliser un réglage", command=self.load_config)
        self.load_button.pack(side="left")
        ttk.Label(configuration, textvariable=self.config_text).pack(side="left", padx=12)

        self.canvas = tk.Canvas(outer, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], bg="#20242a")
        self.canvas.pack(pady=(0, 8))
        self.canvas.create_text(
            PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2,
            text="La vidéo apparaîtra ici", fill="white",
        )
        self.canvas.bind("<Button-1>", self.on_preview_click)
        ttk.Label(outer, textvariable=self.instruction_text, wraplength=840).pack(anchor="w")

        calibration = ttk.Frame(outer)
        calibration.pack(fill="x", pady=(8, 14))
        self.undo_button = ttk.Button(calibration, text="Annuler un point", command=self.undo_point)
        self.undo_button.pack(side="left", padx=(0, 8))
        self.reset_button = ttk.Button(calibration, text="Recommencer les traits", command=self.reset_points)
        self.reset_button.pack(side="left", padx=(0, 8))
        self.save_button = ttk.Button(calibration, text="2. Enregistrer la porte", command=self.save_points)
        self.save_button.pack(side="left")

        analysis = ttk.Frame(outer)
        analysis.pack(fill="x")
        self.analyze_button = ttk.Button(analysis, text="3. Lancer l'analyse", command=self.start_analysis)
        self.analyze_button.pack(side="left", padx=(0, 8))
        self.cancel_button = ttk.Button(analysis, text="Annuler", command=self.cancel_analysis)
        self.cancel_button.pack(side="left", padx=(0, 12))
        self.progress = ttk.Progressbar(analysis, mode="indeterminate", length=210)
        self.progress.pack(side="left", padx=(0, 12))
        ttk.Label(analysis, textvariable=self.status_text).pack(side="left")

        result = ttk.Frame(outer)
        result.pack(fill="x", pady=(14, 0))
        ttk.Label(result, textvariable=self.result_text, font=("Helvetica", 16, "bold")).pack(side="left")
        self.open_button = ttk.Button(result, text="Ouvrir la vidéo résultat", command=self.open_result)
        self.open_button.pack(side="right")

        self.update_buttons()
        root.after(100, self.poll_events)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_buttons(self) -> None:
        can_edit = self.video_path is not None and not self.running
        self.choose_button.configure(state="disabled" if self.running else "normal")
        self.load_button.configure(state="normal" if can_edit else "disabled")
        self.undo_button.configure(state="normal" if can_edit and self.points else "disabled")
        self.reset_button.configure(state="normal" if can_edit and self.points else "disabled")
        self.save_button.configure(state="normal" if can_edit and len(self.points) == 5 else "disabled")
        self.analyze_button.configure(
            state="normal" if can_edit and self.config_path is not None else "disabled"
        )
        self.cancel_button.configure(state="normal" if self.running else "disabled")
        self.open_button.configure(state="normal" if self.result_path is not None else "disabled")

    def choose_video(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choisir une vidéo", initialdir=PROJECT_DIR / "videos_test",
            filetypes=[("Vidéos", "*.mp4 *.mov *.avi *.m4v"), ("Tous les fichiers", "*")],
        )
        if not filename:
            return
        path = Path(filename)
        try:
            frame = video_frame(path)
        except ValueError as error:
            messagebox.showerror("Vidéo illisible", str(error))
            return

        self.video_path = path
        self.config_path = self.result_path = None
        self.points.clear()
        self.frame_height, self.frame_width = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.display_width, self.display_height = image.size
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.configure(width=self.display_width, height=self.display_height)
        self.video_text.set(path.name)
        self.config_text.set("Aucun réglage de porte")
        self.result_text.set("Entrées : —    Sorties : —")
        self.status_text.set("Vidéo chargée")
        self.draw_preview()
        self.update_buttons()

    def on_preview_click(self, event: tk.Event) -> None:
        if self.video_path is None or self.running or len(self.points) >= 5:
            return
        x = min(self.frame_width - 1, max(0, round(event.x * self.frame_width / self.display_width)))
        y = min(self.frame_height - 1, max(0, round(event.y * self.frame_height / self.display_height)))
        self.points.append((x, y))
        self.config_path = None
        self.config_text.set("Nouveau réglage à enregistrer")
        self.draw_preview()
        self.update_buttons()

    def draw_preview(self) -> None:
        self.canvas.delete("all")
        if self.preview_photo is None:
            return
        self.canvas.create_image(0, 0, anchor="nw", image=self.preview_photo)
        shown = [
            (round(x * self.display_width / self.frame_width),
             round(y * self.display_height / self.frame_height))
            for x, y in self.points
        ]
        for start, end in ((0, 1), (2, 3)):
            if len(shown) > end:
                self.canvas.create_line(*shown[start], *shown[end], fill="#ffe05d", width=4)
        for index, (x, y) in enumerate(shown):
            color = "#54e398" if index == 4 else "#ffe05d"
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, outline="black")
            self.canvas.create_text(x + 13, y - 12, text=str(index + 1), fill="white")
        if len(self.points) < 5:
            self.instruction_text.set(POINT_LABELS[len(self.points)])
        elif self.config_path is None:
            self.instruction_text.set("Les 5 points sont placés. Clique sur « Enregistrer la porte ».")
        else:
            self.instruction_text.set("Vérifie les traits et le point vert côté intérieur, puis lance l'analyse.")

    def undo_point(self) -> None:
        if self.points:
            self.points.pop()
            self.config_path = None
            self.config_text.set("Nouveau réglage à enregistrer")
            self.draw_preview()
            self.update_buttons()

    def reset_points(self) -> None:
        self.points.clear()
        self.config_path = None
        self.config_text.set("Aucun réglage de porte")
        self.draw_preview()
        self.update_buttons()

    def save_points(self) -> None:
        if self.video_path is None or len(self.points) != 5:
            return
        path = CONFIG_DIR / f"{self.video_path.stem}.json"
        if path.exists() and not messagebox.askyesno(
            "Remplacer le réglage ?", f"Le fichier {path.name} existe déjà. Le remplacer ?"
        ):
            return
        try:
            save_config(
                argparse.Namespace(output=path, name=self.video_path.stem),
                self.points, self.frame_width, self.frame_height,
            )
        except OSError as error:
            messagebox.showerror("Enregistrement impossible", str(error))
            return
        self.config_path = path
        self.config_text.set(path.name)
        self.status_text.set("Porte enregistrée")
        self.draw_preview()
        self.update_buttons()

    def load_config(self) -> None:
        if self.video_path is None:
            return
        filename = filedialog.askopenfilename(
            title="Choisir un réglage de porte", initialdir=CONFIG_DIR,
            filetypes=[("Réglages JSON", "*.json")],
        )
        if not filename:
            return
        path = Path(filename)
        try:
            points = points_from_config(path, self.frame_width, self.frame_height)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            messagebox.showerror("Réglage illisible", str(error))
            return
        self.points = points
        self.config_path = path
        self.config_text.set(path.name)
        self.status_text.set("Réglage chargé")
        self.draw_preview()
        self.update_buttons()

    def start_analysis(self) -> None:
        if self.video_path is None or self.config_path is None:
            return
        if not self.config_path.is_file():
            messagebox.showerror("Réglage introuvable", "Enregistre ou recharge le réglage de la porte.")
            return
        previous_result = OUTPUT_DIR / f"{self.video_path.stem}_count.mp4"
        if previous_result.exists() and not messagebox.askyesno(
            "Remplacer la vidéo résultat ?",
            f"{previous_result.name} existe déjà. Relancer l'analyse remplacera ce fichier. Continuer ?",
        ):
            return
        self.running = True
        self.cancel_requested = False
        self.result_path = None
        self.result_text.set("Entrées : —    Sorties : —")
        self.status_text.set("Analyse en cours…")
        self.progress.start(12)
        self.update_buttons()
        threading.Thread(target=self.analysis_worker, daemon=True).start()

    def analysis_worker(self) -> None:
        assert self.video_path is not None and self.config_path is not None
        command = [
            sys.executable, "-u", str(PROJECT_DIR / "src" / "count_video.py"),
            "--video", str(self.video_path), "--config", str(self.config_path),
            "--output-dir", str(OUTPUT_DIR),
        ]
        try:
            self.process = subprocess.Popen(
                command, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            if self.cancel_requested:
                self.process.terminate()
            output, _ = self.process.communicate()
            self.events.put(("done", self.process.returncode, output))
        except OSError as error:
            self.events.put(("error", str(error), None))
        finally:
            self.process = None

    def cancel_analysis(self) -> None:
        self.cancel_requested = True
        self.status_text.set("Annulation en cours…")
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def poll_events(self) -> None:
        try:
            while True:
                kind, code, output = self.events.get_nowait()
                self.running = False
                self.progress.stop()
                if self.cancel_requested:
                    self.status_text.set("Analyse annulée")
                elif kind == "error" or code != 0:
                    self.status_text.set("Erreur pendant l'analyse")
                    detail = str(code) if kind == "error" else (output or "Erreur inconnue")[-1800:]
                    messagebox.showerror("Analyse impossible", detail)
                else:
                    entries = re.search(r"Entrees comptees :\s*(\d+)", output or "")
                    exits = re.search(r"Sorties comptees :\s*(\d+)", output or "")
                    assert self.video_path is not None
                    result = OUTPUT_DIR / f"{self.video_path.stem}_count.mp4"
                    if entries is None or exits is None or not result.is_file():
                        self.status_text.set("Résultat introuvable")
                        messagebox.showerror("Résultat introuvable", (output or "")[-1800:])
                    else:
                        self.result_path = result
                        self.result_text.set(f"Entrées : {entries.group(1)}    Sorties : {exits.group(1)}")
                        self.status_text.set("Analyse terminée")
                self.update_buttons()
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def open_result(self) -> None:
        if self.result_path is None:
            return
        try:
            open_file(self.result_path)
        except OSError as error:
            messagebox.showerror("Ouverture impossible", str(error))

    def on_close(self) -> None:
        if self.running:
            self.cancel_analysis()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    FlowSenseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
