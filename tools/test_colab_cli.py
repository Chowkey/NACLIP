"""Local checks for source transfer boundaries; no Google account or GPU needed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('colab_workflow', Path(__file__).with_name('colab_cli.py'))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class TransferTests(unittest.TestCase):
    def test_source_bundle_excludes_private_and_generated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            samples = {'naclip.py': 'edited locally', 'clip/model.py': 'model',
                       'clip/bpe_simple_vocab_16e6.txt.gz': 'tokenizer',
                       '.env': 'secret', 'tools/.env': 'secret',
                       'datasets/images/example.txt': 'data', 'clip/weights.pt': 'weights',
                       '.colab-cli/credentials.txt': 'secret'}
            for name, content in samples.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            notebook = {'cells': [{'cell_type': 'code', 'source': ['print(1)'],
                                   'outputs': [{'text': 'private output'}], 'execution_count': 1}],
                        'metadata': {'kernelspec': {'name': 'python3'}, 'vscode': {'machine': 'local'}}}
            (root / 'colab_naclip.ipynb').write_text(json.dumps(notebook))
            archive_path = root / 'bundle.zip'
            workflow.create_bundle(archive_path, root)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(set(archive.namelist()), {
                    'naclip.py', 'clip/model.py', 'clip/bpe_simple_vocab_16e6.txt.gz', 'colab_naclip.ipynb'})
                self.assertEqual(archive.read('naclip.py'), b'edited locally')
                clean = json.loads(archive.read('colab_naclip.ipynb'))
                self.assertEqual(clean['cells'][0]['outputs'], [])
                self.assertIsNone(clean['cells'][0]['execution_count'])
                self.assertNotIn('vscode', clean['metadata'])

    def test_extract_updates_source_preserves_other_files_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'remote'
            root.mkdir()
            (root / 'results.txt').write_text('retain')
            archive_path = Path(directory) / 'source.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('clip/model.py', 'updated')
            with patch.object(workflow, 'REMOTE', str(root)), contextlib.redirect_stdout(io.StringIO()):
                code = workflow.extraction_code(str(archive_path))
                exec(code, {})
                target = root / 'clip/model.py'
                stamp = target.stat().st_mtime_ns
                exec(code, {})
                self.assertEqual(target.stat().st_mtime_ns, stamp)
                self.assertEqual(target.read_text(), 'updated')
                self.assertEqual((root / 'results.txt').read_text(), 'retain')
                with zipfile.ZipFile(archive_path, 'w') as archive:
                    archive.writestr('../escape.py', 'bad')
                with self.assertRaises(ValueError):
                    exec(workflow.extraction_code(str(archive_path)), {})
                self.assertFalse((root.parent / 'escape.py').exists())

    def test_remote_process_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch.object(workflow, 'REMOTE', directory), contextlib.redirect_stdout(output):
                code = workflow.execution_code([sys.executable, '-c', 'print("diagnostic"); raise SystemExit(7)'])
                with self.assertRaisesRegex(RuntimeError, 'status 7'):
                    exec(code, {})
            self.assertIn('diagnostic', output.getvalue())

    def test_cli_zero_exit_without_remote_success_is_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'did not confirm'):
                workflow.run_remote_code([sys.executable, '-c', 'print("kernel error")'],
                                         'naclip', 'print("work")')
            workflow.run_remote_code([sys.executable, '-c', 'import sys; exec(sys.stdin.read())'],
                                     'naclip', 'print("work")')


if __name__ == '__main__':
    unittest.main()
