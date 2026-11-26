"""Teste le détecteur FlowSense sur une image avec OpenCV, sans installer Aidge."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "FlowSense_miroir_lumiere_v1.onnx"
sys.path.insert(0, str(ROOT / "src"))
from camera_preprocess import prepare_camera_frame
from person_detection_math import decode_boxes


def detect(net, frame, threshold=0.5):
    image, _ = prepare_camera_frame(frame)
    net.setInput(image[np.newaxis, np.newaxis].astype(np.float32) / 255)
    output = net.forward()
    if output.shape != (1, 5, 15, 20) or not np.isfinite(output).all():
        raise ValueError("Sortie inattendue : un détecteur FlowSense QQVGA est attendu.")
    return image, decode_boxes(output[0], threshold=threshold, local_maximum=True, width=160, height=120, stride=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Chemin de l’aperçu PNG.")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("Le seuil doit être compris entre 0 et 1.")
    output = args.output or ROOT / "runs" / "onnx" / f"{args.image.stem}_detection.png"
    if output.suffix.lower() != ".png" or output.resolve() in (args.image.resolve(), args.model.resolve()):
        parser.error("Choisir un chemin PNG différent de l’image et du modèle d’origine.")
    try:
        frame = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("L’image est illisible.")
        # Les versions Windows d’OpenCV ne lisent pas toujours les chemins Unicode.
        model_bytes = np.frombuffer(args.model.read_bytes(), dtype=np.uint8)
        image, boxes = detect(cv2.dnn.readNetFromONNX(model_bytes), frame, args.threshold)
        scale = 4
        preview = cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                               cv2.COLOR_GRAY2BGR)
        for confidence, box in boxes:
            x1, y1, x2, y2 = np.rint(box * scale).astype(int)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (20, 230, 40), 2)
            cv2.putText(preview, f"{confidence:.2f}", (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 230, 40), 1)
        ok, encoded = cv2.imencode(".png", preview)
        if not ok:
            raise ValueError("Impossible d’encoder l’aperçu PNG.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded.tobytes())
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Échec de la détection. {error}\n")
    print(f"{len(boxes)} rectangle(s) au seuil {args.threshold:g}")
    print(f"Image annotée : {output.resolve()}")


if __name__ == "__main__":
    main()
