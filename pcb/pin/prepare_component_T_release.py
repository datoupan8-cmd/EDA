"""Assemble the frozen T or integrated release context without changing source files.

Requires the accepted parent image locally. This prepares files only; it never
builds images, downloads dependencies, starts inference, or creates a ZIP.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / 'releases/official_7744'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_child(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError('Only portable relative paths are accepted')
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Path resolves outside the selected input directory')
    return target


def reconstruct_manifest(base_raw, overlay):
    if hashlib.sha256(base_raw).hexdigest() != overlay['base_sha256']:
        raise ValueError('Parent runtime manifest differs from the accepted image')
    data = json.loads(base_raw)
    data.update(copy.deepcopy(overlay['set_root']))
    for name in overlay['delete_root']:
        del data[name]
    files = data['files']
    files.update(copy.deepcopy(overlay['set_files']))
    for name in overlay['delete_files']:
        del files[name]
    if set(files) != set(overlay['file_order']):
        raise ValueError('Unexpected parent manifest file set')
    data['files'] = {name: files[name] for name in overlay['file_order']}
    if set(data) != set(overlay['root_order']):
        raise ValueError('Unexpected parent manifest root fields')
    data = {name: data[name] for name in overlay['root_order']}
    text = json.dumps(data, ensure_ascii=overlay['ensure_ascii'], indent=2, allow_nan=False)
    if overlay['trailing_newline']:
        text += '\n'
    if overlay['newline'] == 'CRLF':
        text = text.replace('\n', '\r\n')
    result = text.encode('utf-8')
    if hashlib.sha256(result).hexdigest() != overlay['result_sha256']:
        raise ValueError('Reconstructed runtime manifest differs from the accepted context')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', choices=('T', 'integrated'), default='integrated')
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--wheels-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--docker', default='docker', help='Docker CLI with the accepted parent already loaded')
    args = parser.parse_args(argv)
    release = json.loads((ASSETS / 'release.json').read_text('utf-8'))
    layer = release['layers'][args.layer]
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a new output directory; previous evidence is preserved')
    # Never create a build directory inside the source or immutable release data.
    for source in (args.repo_root.resolve(), ASSETS.resolve()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            parser.error('Build output must be separate from the source directories')
    actual = subprocess.check_output([args.docker, 'image', 'inspect', '--format', '{{.Id}}', layer['parent_tag']], text=True).strip()
    if actual != layer['parent_image_id']:
        raise ValueError('Parent image identity differs from the accepted release')
    # Legacy manifests contain machine-specific provenance. Keep it out of Git:
    # read it from the already-verified parent, and apply only our public overlay.
    parent_manifest = subprocess.check_output([
        args.docker, 'run', '--rm', '--pull=never', '--network', 'none', '--read-only',
        '--entrypoint', '/bin/cat', layer['parent_tag'], '/submission/runtime_manifest.json'])
    overlay = json.loads(safe_child(ASSETS, layer['manifest_overlay']).read_text('utf-8'))
    runtime_manifest = reconstruct_manifest(parent_manifest, overlay)
    planned = {}
    for name, record in layer['context_files'].items():
        if name == layer['runtime_manifest_path']:
            if len(runtime_manifest) != record['size'] or hashlib.sha256(runtime_manifest).hexdigest() != record['sha256']:
                raise ValueError('Manifest overlay result failed the release guard')
            planned[name] = runtime_manifest
            continue
        elif args.layer == 'integrated' and name.startswith('payload/') and name[8:] in release['component_files']:
            source = safe_child(args.repo_root, name[8:])
        elif name.startswith('wheels/'):
            if args.wheels_dir is None:
                parser.error('--wheels-dir is required for the integrated layer')
            source = safe_child(args.wheels_dir, name[7:])
        else:
            source = safe_child(ASSETS / layer['asset_directory'], name)
        if not source.is_file() or source.stat().st_size != record['size'] or sha(source) != record['sha256']:
            raise ValueError('Release input is missing, changed or still an LFS pointer: ' + name)
        planned[name] = source
    output.mkdir(parents=True, exist_ok=False)
    for name, source in planned.items():
        target = safe_child(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, bytes):
            with target.open('xb') as outgoing:
                outgoing.write(source)
        else:
            with source.open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    actual_files = {path.relative_to(output).as_posix(): {'size': path.stat().st_size, 'sha256': sha(path)}
                    for path in output.rglob('*') if path.is_file()}
    if actual_files != layer['context_files']:
        raise ValueError('Prepared context differs from the accepted release')
    print(json.dumps({'status': 'FROZEN_CONTEXT_VERIFIED', 'layer': args.layer,
                      'files': len(actual_files), 'bytes': sum(item['size'] for item in actual_files.values()),
                      'parent_image_id': actual, 'source_files_modified': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
