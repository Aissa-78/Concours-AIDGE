"""Affiche la caméra QQVGA via le SDK local, sans modifier la flash."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import json


def read_pixel_words(link, address, count, transport='pipeline'):
    if transport == 'single':
        return [link.mmap_read(address + 4*k) for k in range(count)]
    # Une commande par échange : sa réponse arrive à l'échange suivant.
    # Ne pas concaténer plusieurs commandes sous le même chip-select.
    words = []
    for index in range(count + 1):
        if index < count:
            packet = link.SPIS_READ_MMAP_OPCODE.to_bytes(4, 'big') + (address + 4*index).to_bytes(4, 'big')
        else:
            packet = link.SPIS_NOP_OPCODE.to_bytes(4, 'big') * 2
        response = bytes(link.spi_ctrl.exchange(packet, duplex=True))
        if index:
            if len(response) != 8 or int.from_bytes(response[:4], 'big') != link.SPIS_MMAP_VAL_RSP:
                raise RuntimeError('Réponse SPI invalide : aucun pixel corrompu ne sera affiché.')
            words.append(int.from_bytes(response[4:], 'big'))
    return words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load-sram', action='store_true', help='Charger le firmware caméra déjà compilé.')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=3, help='Nombre de captures pour le test sans fenêtre.')
    parser.add_argument('--transport', choices=['pipeline', 'single'], default='pipeline')
    args = parser.parse_args()
    root = Path(__file__).absolute().parent.parent
    sdk = root / '.local/colibry_sdk'
    sys.path.insert(0, str(sdk / 'riscv_runtime/tools'))
    import libusb_package
    libusb_package.get_libusb1_backend()
    from PIL import Image, ImageTk

    if args.load_sram:
        from elftools.elf.elffile import ELFFile
        from colibry_interfaces.colibry_jtag_api import Colibry_jtag_api
        binary = root / '.local/colibry_examples/camera_mac/build/demo.bin'
        with (binary.parent / 'demo/demo').open('rb') as stream:
            for segment in ELFFile(stream).iter_segments():
                if segment['p_type'] == 'PT_LOAD' and segment['p_memsz']:
                    if not (0x10000000 <= segment['p_paddr'] and segment['p_paddr'] + segment['p_memsz'] <= 0x10020000):
                        raise RuntimeError('Programme hors SRAM.')
        data = binary.read_bytes()
        try:
            jtag = Colibry_jtag_api(freq=2e6, channel=1, p=None)
        except SystemExit as exc:
            raise RuntimeError('La puce ne répond pas au JTAG : rebrancher l’USB avant de relancer.') from exc
        try:
            jtag.cpu_halt()
            # Petits blocs : évite le gros transfert USB du chargeur d'origine.
            for offset in range(0, len(data), 4096):
                block = data[offset:offset + 4096]
                block += bytes((-len(block)) % 4)
                jtag.mmap_writeburst(0x10000000 + offset, [int.from_bytes(block[k:k+4], 'little') for k in range(0, len(block), 4)])
            jtag.mmap_write(0x200100C0, 0x10E1)
            jtag.mmap_write(0x20010000, 2)
            jtag.cpu_restart()
            jtag.mmap_write(0x20010030, 1)
        finally:
            jtag.jtag.close()
        time.sleep(2)

    from colibry_interfaces.colibry_spis_api import Colibry_spis_api
    link = Colibry_spis_api(freq=1e6, channel=2, p=None)
    message = link.SCU_CORE_MSG0_ADDR
    output = root / 'runs/colibry_hardware/captures'
    output.mkdir(parents=True, exist_ok=True)
    window = None
    try:
        link.mmap_write(message, 3)
        deadline = time.monotonic() + 5
        while link.mmap_read(message) != 3:
            if time.monotonic() > deadline:
                raise TimeoutError('Firmware caméra absent. Relancer avec --load-sram après remise sous tension.')
            time.sleep(.02)
        size = link.mmap_read(0x2001E004)
        width, height = size & 65535, size >> 16
        if (width, height) != (160, 120):
            raise RuntimeError(f'Format inattendu : {width} x {height}')
        address = 0x30001000 + link.mmap_read(0x30000018) * 16

        if not args.headless:
            import tkinter as tk
            window = tk.Tk()
            window.title('FlowSense — caméra réelle 160 × 120')
            label = tk.Label(window)
            label.pack()
            status = tk.StringVar(value='Image caméra brute. S : enregistrer, R : tourner l’aperçu, Échap : fermer.')
            tk.Label(window, textvariable=status, wraplength=620).pack(padx=12, pady=10)
            fps_label = tk.Label(window, text='Mesure de la cadence…')
            fps_label.pack(pady=4)
            state = {'running': True, 'save': False, 'rotation': 0}
            window.protocol('WM_DELETE_WINDOW', lambda: state.update(running=False))
            window.bind('<Escape>', lambda e: state.update(running=False))
            window.bind('<s>', lambda e: state.update(save=True))
            window.bind('<r>', lambda e: state.update(rotation=(state['rotation'] + 90) % 360))
        count = 0
        started = time.monotonic()
        transfers = []
        while args.headless or state['running']:
            link.mmap_write(message, 5)
            deadline = time.monotonic() + 5
            while link.mmap_read(message) != 10:
                if time.monotonic() > deadline:
                    raise TimeoutError('La caméra ne fournit pas de nouvelle image.')
                if window:
                    window.update()
                    if not state['running']:
                        return
                time.sleep(.01)
            # Lecture simple sans DMA. Les mots mémoire sont little-endian.
            transfer_start = time.monotonic()
            words = read_pixel_words(link, address, width * height // 4, args.transport)
            transfers.append(time.monotonic() - transfer_start)
            raw = [byte for word in words for byte in word.to_bytes(4, 'little')]
            pixels = link.descramble_spi_img_nnpa(raw, height, width, 1)
            frame = Image.fromarray(pixels)
            count += 1
            if args.headless or state['save']:
                destination = output / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.png')
                frame.save(destination)
                print('Capture :', destination, flush=True)
                if window:
                    status.set('Image brute enregistrée : ' + destination.name)
                    state['save'] = False
            if window:
                preview = frame.rotate(state['rotation'], expand=True)
                preview = preview.resize((preview.width * 4, preview.height * 4))
                photo = ImageTk.PhotoImage(preview)
                label.configure(image=photo)
                label.image = photo
                fps_label.configure(text=f'{count / (time.monotonic() - started):.2f} image/s affichée — {width} × {height}')
                window.update()
            if args.headless and count >= args.frames:
                break
        elapsed = time.monotonic() - started
        report = {'transport': args.transport, 'frames': count, 'elapsed_seconds': elapsed,
                  'fps': count / elapsed if elapsed else 0, 'transfer_seconds': transfers,
                  'scope': 'PC receive/decode/save or display, not embedded people detector FPS'}
        (output.parent / f'viewer_timing_{args.transport}.json').write_text(json.dumps(report, indent=2))
        print('Cadence réception :', report['fps'], 'image/s', flush=True)
    finally:
        try:
            link.mmap_write(message, 0)
        finally:
            link.cleanup()
            link.spictrl.terminate()
            if window:
                window.destroy()


if __name__ == '__main__':
    main()
