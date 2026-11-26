"""Capture une séquence Colibry puis assemble un MP4 à sa cadence moyenne."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import signal
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).absolute().parent.parent
KIT = ROOT / '.local/colibry_examples/FlowSense_Camera_Mac'


def assemble_video(run, transport, expected_frames=None):
    captures = sorted((run / 'captures').glob('*.png'))
    timing = json.loads((run / f'viewer_timing_{transport}.json').read_text())
    fps = float(timing['fps'])
    if len(captures) != timing['frames'] or not 0 < fps < 1000:
        raise ValueError('Le rapport de capture ne correspond pas aux PNG.')
    if len(captures) < 2 or (expected_frames and len(captures) != expected_frames):
        raise ValueError('Capture incomplète.')
    rate = str(Fraction(fps).limit_denominator(10000))
    output = run / 'video.mp4'
    if output.exists():
        raise FileExistsError(output)
    # Encodage après la capture pour laisser toute la cadence au transfert USB.
    with tempfile.TemporaryDirectory(dir=run) as temp_dir:
        temporary = Path(temp_dir) / 'video.mp4'
        with tempfile.TemporaryFile() as errors:
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'rawvideo',
                '-pixel_format', 'gray', '-video_size', '160x120', '-framerate', rate,
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '17',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary),
            ]
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors) as encoder:
                try:
                    for path in captures:
                        with Image.open(path) as image:
                            if (image.format, image.mode, image.size) != ('PNG', 'L', (160, 120)):
                                raise ValueError(f'Format PNG inattendu : {path.name}')
                            encoder.stdin.write(image.tobytes())
                    encoder.stdin.close()
                except BrokenPipeError:
                    encoder.wait()
                except BaseException:
                    encoder.kill()
                    raise
                if encoder.wait():
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors='replace'))
        movie = cv2.VideoCapture(str(temporary))
        try:
            ok, first_frame = movie.read()
            valid = (ok and first_frame.shape[:2] == (120, 160)
                     and int(movie.get(cv2.CAP_PROP_FRAME_COUNT)) == len(captures)
                     and abs(movie.get(cv2.CAP_PROP_FPS) - fps) < .02)
        finally:
            movie.release()
        if not valid:
            raise RuntimeError('Contrôle du MP4 échoué. Les PNG sont disponibles.')
        temporary.replace(output)
    print(f'{len(captures)} images · {fps:.2f} images/s en moyenne')
    print(f'Vidéo : {output}\nPNG originaux : {run / "captures"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', type=int, nargs='?', default=225,
                        help='Nombre d’images à enregistrer, 225 par défaut.')
    parser.add_argument('--manuel', action='store_true',
                        help='Enregistrer jusqu’à Ctrl+C puis créer le MP4.')
    parser.add_argument('--transport', choices=('fast', 'batch', 'pipeline'), default='fast')
    parser.add_argument('--from-run', type=Path, metavar='DOSSIER',
                        help='Assembler un MP4 à partir d’une capture déjà terminée.')
    args = parser.parse_args()
    if args.from_run is not None:
        assemble_video(args.from_run.absolute(), args.transport)
        return
    if not 2 <= args.frames <= 10000:
        parser.error('Indiquer entre 2 et 10 000 images.')
    viewer = KIT / 'scripts/view_colibry_camera.py'
    if not viewer.is_file():
        parser.error('Kit caméra rapide absent de .local/colibry_examples/FlowSense_Camera_Mac.')
    subprocess.run(['shasum', '-a', '256', '-c', 'SHA256SUMS'], cwd=KIT, check=True)
    parent = ROOT / 'runs/colibry_hardware/videos'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='prise_', dir=parent))
    frame_limit = 999999 if args.manuel else args.frames
    command = [
        str(ROOT / '.venv-colibry-hw/bin/python'), '-B', '-u', str(viewer),
        '--sdk-dir', str(ROOT / '.local/colibry_sdk'), '--load-sram',
        '--firmware-dir', str(KIT / '.local/colibry_camera_blanking'),
        '--transport', args.transport, '--overlap-capture', '--headless',
        '--frames', str(frame_limit), '--output-dir', str(run / 'captures'),
    ]
    if args.manuel:
        print(f'Enregistrement manuel dans {run}. Ctrl+C pour arrêter et créer le MP4.', flush=True)
        capture = None
        stop_requested = False

        def stop_capture(_signal, _frame):
            nonlocal stop_requested
            if stop_requested:
                return
            stop_requested = True
            print('\nArrêt demandé. Création du MP4 après la dernière image.', flush=True)
            if capture is not None and capture.poll() is None:
                try:
                    capture.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass

        previous = signal.signal(signal.SIGINT, stop_capture)
        try:
            with subprocess.Popen(command, cwd=ROOT, start_new_session=True) as capture:
                if stop_requested:
                    capture.send_signal(signal.SIGINT)
                status = capture.wait()
        finally:
            signal.signal(signal.SIGINT, previous)
        if status != 0:
            raise SystemExit(f'Capture interrompue (code {status}). Images disponibles dans {run}.')
        if len(list((run/'captures').glob('*.png'))) < 2:
            raise SystemExit('Prise trop courte. Attendre au moins deux lignes Capture avant Ctrl+C.')
    else:
        print(f'Enregistrement de {args.frames} images dans {run}', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    assemble_video(run, args.transport, None if args.manuel else args.frames)


if __name__ == '__main__':
    main()
