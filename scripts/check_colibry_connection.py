"""Vérifie l'identifiant Colibry et écoute l'UART, sans charger de firmware."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="Durée d'écoute de l'UART, entre 0 et 30 secondes.")
    parser.add_argument("--device", default="ftdi://ftdi:4232hp",
                        help="Préfixe USB de la sonde, sans numéro de canal.")
    parser.add_argument("--report", type=Path, help="Rapport JSON local facultatif.")
    args = parser.parse_args()
    if not 0 <= args.seconds <= 30:
        parser.error("--seconds doit être compris entre 0 et 30")

    # Charge la bibliothèque USB fournie dans le venv, notamment sur macOS.
    import libusb_package
    if libusb_package.get_libusb1_backend() is None:
        raise RuntimeError("La bibliothèque libusb n'est pas disponible.")
    from pyftdi.bits import BitSequence
    from pyftdi.ftdi import Ftdi
    from pyftdi.jtag import JtagEngine
    from pyftdi.serialext import serial_for_url

    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "device": args.device, "expected_idcode": "0x249511c3",
              "jtag_ok": False, "uart_text": "", "uart_bytes": 0,
              "firmware_written": False, "errors": []}
    Ftdi.show_devices()
    jtag = JtagEngine(trst=False, frequency=100_000)
    try:
        jtag.configure(args.device.rstrip("/") + "/1")
        jtag.reset()
        # Seule l'instruction IDCODE est sélectionnée : aucun accès mémoire.
        jtag.write_ir(BitSequence("00001", msb=True, length=5))
        identifier = int(jtag.read_dr(32))
        jtag.go_idle()
        result["idcode"] = f"0x{identifier:08x}"
        result["jtag_ok"] = identifier == 0x249511C3
        print("Identifiant JTAG :", result["idcode"])
    except Exception as exc:
        result["errors"].append(f"JTAG : {exc}")
    finally:
        jtag.close()

    if args.seconds:
        try:
            with serial_for_url(args.device.rstrip("/") + "/3",
                                baudrate=115200, timeout=0.2) as port:
                deadline = time.monotonic() + args.seconds
                data = bytearray()
                while time.monotonic() < deadline and len(data) < 8192:
                    data.extend(port.read(min(256, 8192 - len(data))))
            result["uart_bytes"] = len(data)
            result["uart_text"] = data.decode("utf-8", errors="replace")
            print("Messages UART :", repr(result["uart_text"]))
        except Exception as exc:
            result["errors"].append(f"UART : {exc}")
    for error in result["errors"]:
        print(error)
    print("Connexion Colibry confirmée." if result["jtag_ok"] else
          "Connexion non confirmée. Vérifier JTAG, ON et l'alimentation.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)
                               + "\n", encoding="utf-8")
        print("Rapport :", args.report.absolute())
    return 0 if result["jtag_ok"] and not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
