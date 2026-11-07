"""Evaluate with an explicit dataset path, preserving the upstream configs."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/cfg_voc21.py')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = Path(args.data_root).resolve()
    if not data.is_dir():
        parser.error(f'Dataset directory does not exist: {data}')
    if (str(Path(args.work_dir)).startswith('/content/drive/') and
            not Path('/content/drive/MyDrive').is_dir()):
        parser.error('Mount Google Drive before saving evaluation results there.')
    from mmengine.config import Config
    cfg = Config.fromfile(str(root / args.config))
    cfg.test_dataloader.dataset.data_root = str(data)
    # Colab can have limited shared memory; use the main process for loading.
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    with tempfile.TemporaryDirectory(prefix='naclip-config-') as temporary:
        config = str(Path(temporary) / 'evaluation.py')
        cfg.dump(config)
        subprocess.run([sys.executable, 'eval.py', '--config', config,
                        '--work-dir', args.work_dir], cwd=root, check=True)


if __name__ == '__main__':
    main()
