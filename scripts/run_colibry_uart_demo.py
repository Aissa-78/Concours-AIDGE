"""Compilation locale du premier exemple ASYGN, avec chargement SRAM facultatif."""
from pathlib import Path
import subprocess,os,shutil,argparse,sys,runpy,time,json,hashlib
parser=argparse.ArgumentParser(description='Compile le test UART Colibry. Option --load-sram : exécution en mémoire temporaire.')
parser.add_argument('--load-sram',action='store_true')
args=parser.parse_args()
root=Path(__file__).absolute().parent.parent
sdk=root/'.local/colibry_sdk'
app=root/'.local/colibry_examples/uart_mac'
if not app.exists():shutil.copytree(sdk/'examples/10_uart_printf_blink',app)
src=app/'main.c'
s=src.read_text()
s=s.replace("Hello, I'm Colibry!", 'FlowSense Mac SRAM OK')
src.write_text(s)
compilers=list((root/'.local/corev').rglob('riscv32-corev-elf-g++'))
if not compilers:raise SystemExit('Compiler not found')
bin=compilers[0].parent
env=dict(os.environ);env['PATH']=str(root/'.venv-colibry-hw/bin')+os.pathsep+str(bin)+os.pathsep+env['PATH']
subprocess.run([str(bin/'riscv32-corev-elf-g++'),'--version'],check=True)
(root/'runs/colibry_hardware').mkdir(parents=True,exist_ok=True)
with (root/'runs/colibry_hardware/build_uart.log').open('w') as log:
 r=subprocess.run(['make','-j4','all','board=EVBTINY','COLIBRY_SDK_PATH='+str(sdk)],cwd=app,env=env,stdout=log,stderr=subprocess.STDOUT)
print('Build status:',r.returncode)
print((root/'runs/colibry_hardware/build_uart.log').read_text()[-6500:])
if r.returncode: raise SystemExit(r.returncode)
if not args.load_sram: raise SystemExit(0)

# Vérifier tous les segments avant de charger la SRAM, sans écrire la flash.
from elftools.elf.elffile import ELFFile
binary=app/'build/test.bin'
with (app/'build/test/test').open('rb') as f:
 for segment in ELFFile(f).iter_segments():
  if segment['p_type']=='PT_LOAD' and segment['p_memsz']:
   if not (0x10000000<=segment['p_paddr'] and segment['p_paddr']+segment['p_memsz']<=0x10020000):
    raise RuntimeError('Segment ELF hors SRAM')
import libusb_package
libusb_package.get_libusb1_backend()
from pyftdi.serialext import serial_for_url
with serial_for_url('ftdi://ftdi:4232hp/3',baudrate=115200,timeout=0.2) as uart:
 uart.reset_input_buffer()
 sys.argv=['colibry-sramtool.py',str(binary),'0x10000000','--run']
 runpy.run_path(str(sdk/'riscv_runtime/tools/bin_loader/colibry-sramtool.py'),run_name='__main__')
 deadline=time.monotonic()+5
 data=bytearray()
 while time.monotonic()<deadline and len(data)<8192:data.extend(uart.read(min(256,8192-len(data))))
 text=data.decode(errors='replace')
 print('UART :',repr(text))
 result={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'loaded_into':'SRAM','flash_written':False,'uart_text':text,'success':'FlowSense Mac SRAM OK' in text}
 (root/'runs/colibry_hardware/uart_sram_test.json').write_text(json.dumps(result,indent=2)+'\n')
 if not result['success']:raise SystemExit('Message de test absent')
 print('Exécution confirmée, flash conservée.')
