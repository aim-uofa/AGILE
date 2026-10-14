"""Protect existing local data and reject unsafe or inconsistent data archives."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
from tools.download_ho3d import checksum, extract_archive
from optimization.run_ho3d import configuration


class DownloadTests(unittest.TestCase):
    def test_optimized_uses_separate_pinned_manifest(self):
        from tools.download_ho3d import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, entry = self.fixture(root, 'HO3D/optimized/train/MC1/config.json')
            entry.update(scene='MC1', kind='optimized', path='ho3d/optimized/MC1.tar.gz')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'archives': [entry]}))
            calls = []
            def download(repo, filename, **kwargs):
                self.assertEqual(repo, 'Chrisss/AGILE')
                self.assertEqual(kwargs, {'repo_type': 'dataset', 'revision': 'resolved-commit'})
                calls.append(filename)
                return str(manifest if filename == 'ho3d/optimized/manifest.json' else archive)
            api = types.SimpleNamespace(repo_info=lambda *a, **kw: types.SimpleNamespace(sha='resolved-commit'))
            hub = types.SimpleNamespace(HfApi=lambda: api, hf_hub_download=download)
            argv = ['download_ho3d', '--kind', 'optimized', '--scenes', 'MC1', '--output', str(root / 'out')]
            with patch.dict(sys.modules, {'huggingface_hub': hub}), patch.object(sys, 'argv', argv):
                main()
            self.assertEqual(calls, ['ho3d/optimized/manifest.json', 'ho3d/optimized/MC1.tar.gz'])
            self.assertEqual((root / 'out/HO3D/optimized/train/MC1/config.json').read_bytes(), b'prepared input')

    def fixture(self, root, name='HO3D/input.txt', value=b'prepared input'):
        archive = root/'input.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            entry = tarfile.TarInfo(name); entry.size = len(value)
            stream.addfile(entry, io.BytesIO(value))
        return archive, {'sha256': checksum(archive), 'members': [dict(path=name, bytes=len(value), sha256=hashlib.sha256(value).hexdigest())]}

    def test_valid_resume_and_preserve_different_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); archive, entry=self.fixture(root); output=root/'output'
            extract_archive(archive, output, entry)
            extract_archive(archive, output, entry)
            target=output/'HO3D/input.txt'; target.write_bytes(b'user data')
            with self.assertRaises(FileExistsError): extract_archive(archive, output, entry)
            self.assertEqual(target.read_bytes(), b'user data')

    def test_path_escape_and_checksum_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); archive, entry=self.fixture(root, '../escaped')
            with self.assertRaisesRegex(ValueError, 'Unsafe'): extract_archive(archive, root/'out', entry)
            entry['sha256']='0'*64
            with self.assertRaisesRegex(ValueError, 'SHA-256'): extract_archive(archive, root/'out', entry)
            self.assertFalse((root/'escaped').exists())

    def test_preset_keeps_b_setting_and_contact_frame(self):
        values=configuration('BB13', '/tmp/data', '/tmp/mano', '/tmp/results')
        self.assertEqual(values['start_frame'], 130)
        self.assertTrue(values['fix_hand_rotation_to_wilor'])
        self.assertTrue(values['recover_object_pose_from_mask'])
        self.assertEqual(values['fixed_wilor_translation_init'], 'joints_2d')
        self.assertEqual(values['data_dir'], str(Path('/tmp/data/train/BB13').resolve()))
        with self.assertRaises(ValueError): configuration('unreleased', '/tmp/data', '/tmp/mano', '/tmp/results')


if __name__=='__main__':
    unittest.main()
