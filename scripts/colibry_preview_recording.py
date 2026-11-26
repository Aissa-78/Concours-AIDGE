"""Commandes d'enregistrement pour l'aperçu Colibry sur Mac."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import tempfile
import time
import tkinter as tk


class PreviewRecording:
    def __init__(self, window, project, transport):
        self.project = Path(project)
        self.transport = transport
        self.run = None
        self.encoder = None
        self.last_video = None
        self.saved = 0
        self.status = tk.StringVar(master=window, value='Prêt à filmer')
        controls = tk.Frame(window)
        controls.pack(pady=6)
        self.toggle_button = tk.Button(controls, text='Enregistrer', command=self.toggle)
        self.toggle_button.pack(side=tk.LEFT, padx=4)
        self.open_button = tk.Button(controls, text='Ouvrir la vidéo', command=self.open_video,
                                     state=tk.DISABLED)
        self.open_button.pack(side=tk.LEFT, padx=4)
        tk.Label(window, textvariable=self.status, wraplength=620).pack(padx=8, pady=3)

    def toggle(self):
        if self.encoder is not None:
            return
        if self.run is not None:
            self.stop()
            return
        parent = self.project / 'runs/colibry_hardware/videos'
        try:
            parent.mkdir(parents=True, exist_ok=True)
            run = Path(tempfile.mkdtemp(prefix='prise_', dir=parent))
            (run / 'captures').mkdir()
        except OSError as error:
            self.status.set(f'Impossible de démarrer. {error}')
            return
        self.run = run
        self.saved = 0
        self.toggle_button.configure(text='Arrêter')
        self.status.set('● Enregistrement en cours')

    def save_frame(self, frame):
        if self.run is None:
            return
        now = time.monotonic()
        name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.png'
        try:
            frame.save(self.run / 'captures' / name, compress_level=1)
        except OSError as error:
            self.stop()
            self.status.set(f'Enregistrement interrompu. {error}')
            return
        if not self.saved:
            self.first_frame = now
        self.last_frame = now
        self.saved += 1
        if self.saved == 1 or self.saved % 8 == 0:
            self.status.set(f'● Enregistrement · {self.saved} images')

    def stop(self):
        if self.run is None:
            return
        run, self.run = self.run, None
        self.toggle_button.configure(text='Enregistrer')
        if self.saved < 2:
            self.status.set('Prise trop courte. Enregistre au moins deux images.')
            return
        period = (self.last_frame - self.first_frame) / (self.saved - 1)
        timing = {'transport': self.transport, 'frames': self.saved,
                  'elapsed_seconds': period * self.saved, 'fps': 1 / period,
                  'scope': 'Capture vidéo pendant l’aperçu'}
        try:
            (run / f'viewer_timing_{self.transport}.json').write_text(json.dumps(timing, indent=2))
            with (run / 'encodage.log').open('w') as log:
                self.encoder = subprocess.Popen([
                    str(self.project / '.venv/bin/python'),
                    str(self.project / 'scripts/enregistrer_colibry_mac.py'),
                    '--from-run', str(run), '--transport', self.transport,
                ], stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            self.status.set(f'Création du MP4 impossible. PNG dans {run}. {error}')
            return
        self.encoding_run = run
        self.toggle_button.configure(state=tk.DISABLED)
        self.status.set('Création du MP4…')

    def poll(self, wait=False):
        if self.encoder is None:
            return
        status = self.encoder.wait() if wait else self.encoder.poll()
        if status is None:
            return
        self.encoder = None
        self.toggle_button.configure(state=tk.NORMAL)
        video = self.encoding_run / 'video.mp4'
        if status == 0 and video.is_file():
            self.last_video = video
            self.open_button.configure(state=tk.NORMAL)
            self.status.set(f'Vidéo enregistrée · {self.encoding_run.name}/video.mp4')
            print(f'Vidéo : {video}', flush=True)
        else:
            self.status.set(f'Création du MP4 échouée. Voir encodage.log dans {self.encoding_run}.')

    def open_video(self):
        if self.last_video is not None:
            subprocess.Popen(['open', str(self.last_video)])

    def finish(self):
        self.stop()
        self.poll(wait=True)
