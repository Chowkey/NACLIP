"""Create an isolated Linux/Colab environment without changing the notebook kernel."""
from pathlib import Path
import subprocess
import sys


def main():
    if sys.platform != 'linux' or not Path('/content').is_dir():
        raise SystemExit('Run this script on a Colab runtime, not your local computer.')
    subprocess.run(['nvidia-smi'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    env = Path('/content/naclip-env')
    python = str(env / 'bin/python')
    uv = [sys.executable, '-m', 'uv']
    if not Path(python).exists():
        subprocess.run(uv + ['venv', '--python', '3.10', str(env)], check=True)
    install = uv + ['pip', 'install', '--python', python]
    subprocess.run(install + ['torch==2.0.1', 'torchvision==0.15.2',
                             '--index-url', 'https://download.pytorch.org/whl/cu118'], check=True)
    subprocess.run(install + ['numpy==1.26.4', 'setuptools==69.5.1',
                             'opencv-python==4.10.0.84', 'mmengine==0.8.4',
                             'mmsegmentation==1.1.1', 'ftfy', 'regex',
                             'yapf==0.40.1', 'matplotlib', 'scipy', 'tqdm'], check=True)
    subprocess.run(install + ['mmcv==2.0.1', '--only-binary', 'mmcv',
                             '--find-links', 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html'], check=True)
    root = Path(__file__).resolve().parents[1]
    check = ('import torch, mmcv, mmseg; from mmcv.ops import nms; '
             'import naclip, custom_datasets; '
             'assert torch.cuda.is_available(), "No CUDA GPU attached"; '
             'print("Ready:", torch.cuda.get_device_name(0), "torch", torch.__version__, "mmcv", mmcv.__version__)')
    subprocess.run([python, '-c', check], cwd=root, check=True)
    print('Environment ready:', python)


if __name__ == '__main__':
    main()
