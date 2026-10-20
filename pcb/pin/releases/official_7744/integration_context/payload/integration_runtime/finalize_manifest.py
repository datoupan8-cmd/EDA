"""Build-time inventory of the newly installed, isolated OCR dependency directory."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys

ROOT = Path('/submission')
TARGET = ROOT / '.runtime/alternative_frontend'
manifest = json.loads((ROOT / 'runtime_manifest.json').read_text('utf-8'))
versions = {dist.metadata['Name'].lower(): dist.version for dist in importlib.metadata.distributions(path=[str(TARGET)])}
expected = {'rapidocr': '3.9.2', 'omegaconf': '2.3.0', 'colorlog': '6.10.1', 'antlr4-python3-runtime': '4.9.3'}
assert all(versions.get(name) == version for name, version in expected.items()), versions
for path in sorted(TARGET.rglob('*')):
    if path.is_file():
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['files'][path.relative_to(ROOT).as_posix()] = {'size': path.stat().st_size, 'sha256': digest,
                                                             'isolated_component_dependency': True}
manifest['isolated_component_dependencies'] = expected
(ROOT / 'runtime_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
sys.path.insert(0, str(TARGET))
from rapidocr import RapidOCR
assert RapidOCR is not None
print(json.dumps({'isolated_dependencies': versions, 'manifest_files': len(manifest['files'])}))
