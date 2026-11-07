"""Ubuntu command workflow using Google's official CLI; no local ML packages needed."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/content/NACLIP'
REMOTE_PYTHON = '/content/naclip-env/bin/python'
# Explicit source roots prevent accidental uploads of runtime data or secrets.
SOURCE_DIRS = {'clip', 'configs', 'prompts', 'tools'}
ROOT_FILES = {'AGENTS.md', 'COLAB.md', 'README.md', 'LICENSE', '.gitignore',
              'colab_naclip.ipynb', 'custom_datasets.py', 'eval.py', 'naclip.py',
              'pamr.py', 'test_all.sh'}


def source_files(root=ROOT):
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root)
    for name in sorted(set(names.decode('utf-8').split('\0')) - {''}):
        rel = Path(name)
        if (rel.is_absolute() or '..' in rel.parts or
                any(part.startswith('.') for part in rel.parts[:-1])):
            continue
        allowed = (name in ROOT_FILES or
                   (rel.parts[0] in SOURCE_DIRS and rel.suffix in {'.py', '.txt', '.gz'}))
        # Only this upstream tokenizer asset is permitted as a compressed file.
        if rel.suffix == '.gz' and name != 'clip/bpe_simple_vocab_16e6.txt.gz':
            allowed = False
        path = root / rel
        if allowed and path.is_file() and not path.is_symlink():
            if path.stat().st_size > 10 * 1024 * 1024:
                raise RuntimeError(f'Source file unexpectedly exceeds 10 MiB: {name}')
            yield name, path


def create_bundle(destination, root=ROOT):
    files = list(source_files(root))
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files:
            if name.endswith('.ipynb'):
                notebook = json.loads(path.read_text(encoding='utf-8'))
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        cell['outputs'] = []
                        cell['execution_count'] = None
                notebook['metadata'] = {key: value for key, value in notebook['metadata'].items()
                                        if key in {'kernelspec', 'language_info'}}
                archive.writestr(name, json.dumps(notebook))
            else:
                archive.write(path, name)
    return files


def extraction_code(remote_archive):
    return f'''
from pathlib import Path
import hashlib, zipfile
root = Path({REMOTE!r}).resolve()
root.mkdir(parents=True, exist_ok=True)
updated = 0
with zipfile.ZipFile({remote_archive!r}) as archive:
    for item in archive.infolist():
        target = (root / item.filename).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError('Invalid source archive path')
        data = archive.read(item)
        if target.is_file() and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(data).digest():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        updated += 1
print(f'Synced {{updated}} changed files to {{root}}; no remote files deleted.')
'''


def execution_code(command):
    # Pipe child output back through the notebook protocol for both CLI and VS Code.
    return f'''
import subprocess
with subprocess.Popen({command!r}, cwd={REMOTE!r}, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait():
        raise RuntimeError(f'Remote process exited with status {{process.returncode}}')
'''


def run_remote_code(cli_command, session, code):
    # Some CLI releases return zero even when a kernel cell raises. Require a
    # unique marker reached only after the entire remote cell succeeds.
    marker = 'NACLIP_COMPLETE_' + uuid.uuid4().hex
    command = [*cli_command, 'exec', '-s', session, '--timeout', '600']
    completed = False
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                          errors='replace', bufsize=1) as process:
        process.stdin.write(code + f'\nprint({marker!r}, flush=True)\n')
        process.stdin.close()
        for line in process.stdout:
            if line.strip() == marker:
                completed = True
            else:
                print(line, end='', flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    if not completed:
        raise RuntimeError('Colab did not confirm successful execution; inspect the output above.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['sessions', 'new', 'sync', 'setup', 'smoke', 'eval'])
    parser.add_argument('--session', default='naclip')
    parser.add_argument('--dry-run', action='store_true', help='List sync files without connecting')
    parser.add_argument('--config', default='configs/cfg_voc21.py')
    parser.add_argument('--data-root')
    parser.add_argument('--work-dir')
    args = parser.parse_args()
    if args.dry_run:
        if args.action != 'sync':
            parser.error('--dry-run is only available for sync')
        for name, _ in source_files():
            print(name)
        return
    if args.action == 'eval' and (not args.data_root or not args.work_dir):
        parser.error('eval requires --data-root and --work-dir (remote Linux paths)')
    if sys.platform != 'linux':
        parser.error('Run this workflow on Ubuntu. You can preview sync with --dry-run here.')
    local_cli = ROOT / '.colab-cli/bin/colab'
    cli = str(local_cli) if local_cli.is_file() else shutil.which('colab')
    if not cli:
        parser.error('Run bash tools/setup_ubuntu.sh first')

    def call(*command, code=None):
        subprocess.run([cli, *command], input=code, text=True, check=True)

    def execute(code):
        run_remote_code([cli], args.session, code)

    if args.action == 'sessions':
        call('sessions')
    elif args.action == 'new':
        call('new', '-s', args.session, '--gpu', 'T4')
    elif args.action == 'sync':
        # One archive avoids slow per-file API uploads. No Git state is sent.
        remote_archive = '/content/naclip-source.zip'
        with tempfile.TemporaryDirectory(prefix='naclip-sync-') as temporary:
            bundle = Path(temporary) / 'source.zip'
            files = create_bundle(bundle)
            print(f'Uploading {len(files)} source files ({bundle.stat().st_size} bytes)', flush=True)
            # The Contents API addresses paths relative to the runtime filesystem root.
            call('upload', '-s', args.session, str(bundle), remote_archive.lstrip('/'))
            execute(extraction_code(remote_archive))
    else:
        if args.action == 'setup':
            command = ['python3', '-u', 'tools/colab_setup.py']
        elif args.action == 'smoke':
            command = [REMOTE_PYTHON, '-u', 'tools/colab_smoke.py']
        else:
            command = [REMOTE_PYTHON, '-u', 'tools/colab_eval.py', '--config', args.config,
                       '--data-root', args.data_root, '--work-dir', args.work_dir]
        execute(execution_code(command))


if __name__ == '__main__':
    main()
