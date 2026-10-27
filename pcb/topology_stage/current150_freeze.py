"""Resumable current-profile image-only inference on exact 0001-0150.

Workers write disjoint cases and cache directories. Merge rejects missing cases,
different versions, changed images/outputs, or any GT access. No model tuning.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from tools.team_access_guard import Guard
from .power_comparison import IDS, discover, read, write, sha, source_inventory

ROOT = Path(__file__).resolve().parents[2]
GLOBAL_CACHE = ROOT/'runs/component_recovery_cache/updated_text'
SEEDS = ROOT/'runs/power_rails_v2_20261027/current7/predictions_sealed.json'
SELECTED = ['pcb/component', 'pcb/pin', 'pcb/core', 'pcb/wire_stage', 'pcb/network_shared',
            'pcb/schema.py', 'pcb/submission.py']


class FreezeGuard(Guard):
    def __init__(self):
        self.actual_ids = set()
        super().__init__()

    def audit(self, event, args):
        super().audit(event, args)
        if event == 'open' and args and isinstance(args[0], (str, bytes, Path)):
            parts = str(args[0]).replace('\\', '/').lower().split('/')
            if '200_train_cases' in parts:
                i = parts.index('200_train_cases')
                if i+1 < len(parts) and parts[i+1] in IDS:
                    self.actual_ids.add(parts[i+1])


def version_lock():
    from pcb.core.team_config import load_team_config
    from pcb.network_shared.selection import selected_versions
    config = load_team_config('configs/team.json')
    expected = {'text': 'recovery_rotation_vertical_v1', 'component': 'rotation_vertical_n1_n2_v1',
                'pin_localization': 'l39', 'pin_semantics': 'l39', 'wire': 'v3_frame_guard', 'topology': 'v3'}
    if any(config.as_dict()[k] != v for k, v in expected.items()) or config.enable_flying_labels:
        raise ValueError('Current profile differs from the predeclared baseline')
    versions = selected_versions()
    if versions != {'net_hypergraph': 'legacy_v2', 'net_lines': 'legacy_v2'}:
        raise ValueError('Freeze the declared network versions first')
    models = {}
    listing = subprocess.check_output(['git', 'lfs', 'ls-files', '--long'], cwd=ROOT, text=True)
    for line in listing.splitlines():
        digest, status, name = line.split(' ', 2)
        if status == '*' and (name.startswith('models/') or name.endswith('/best.pt')):
            if sha(ROOT/name) != digest:
                raise ValueError('Model changed: '+name)
            models[name] = digest
    for name in ('assets/cpntLibrary.json', 'models/component/component_selective120_candidate/model.json'):
        if (ROOT/name).is_file():
            models[name] = sha(ROOT/name)
    lock = {'config': config.as_dict(), 'network_versions': versions,
            'models_sha256': models, 'source_sha256': source_inventory(),
            'GT_opens': 0, 'OFFICIAL_SCORE': False}
    key = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    return config, lock, key


def valid_receipt(folder, cid, key, image):
    receipt = folder/'receipt.json'
    if not receipt.is_file():
        return None
    r = read(receipt)
    if r['case_id'] != cid or r['version_key'] != key or r['image_sha256'] != sha(image):
        raise ValueError('Resume version/image mismatch: '+cid)
    for k, name in (('scene', 'scene.json'), ('mask', 'wire_mask.png'), ('prediction', 'result.json')):
        if sha(folder/name) != r[k]:
            raise ValueError('Resume output changed: '+cid+'/'+name)
    return r


def seed_case(folder, cid, key, image, seed_manifest):
    if cid not in seed_manifest['ACTUAL_CASE_IDS']:
        return None
    if subprocess.call(['git', 'diff', '--quiet', 'e4682aef66e0b1eebd0c4727673926d23da7754e', '--', *SELECTED], cwd=ROOT):
        raise ValueError('Selected inference sources changed; old current7 cannot be reused')
    from .power_v2_experiment import current_folder
    source = current_folder(cid)
    proof = seed_manifest['input_provenance'][cid]
    if sha(image) != proof['image_sha256']:
        raise ValueError('Current7 seed image changed')
    for name in ('scene.json', 'wire_mask.png', 'result.json'):
        if sha(source/name) != proof[name]:
            raise ValueError('Current7 seed output changed')
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('scene.json', 'wire_mask.png', 'result.json'):
        shutil.copy2(source/name, folder/name)
    scene = read(folder/'scene.json')
    if scene['diagnostics']['deployment_recovery']['profile'] != 'rotation_vertical_n1_n2_v1':
        raise ValueError('Seed component profile mismatch')
    r = {'case_id': cid, 'version_key': key, 'image_sha256': sha(image), 'provenance': 'verified_current7',
         'source_folder': str(source), 'seconds': 0., 'scene': sha(folder/'scene.json'),
         'mask': sha(folder/'wire_mask.png'), 'prediction': sha(folder/'result.json')}
    write(folder/'receipt.json', r)
    return r


def warm_observations(worker_cache, inputs, ids):
    """Look up only allowed image+signature keys; no foreign cache enumeration."""
    import cv2
    import numpy as np
    from pcb.component.ocr_ablation import PairedPPOcrV5Backend
    from pcb.component.ocr_server import ServerPPOcrV5Backend
    mobile = PairedPPOcrV5Backend(worker_cache/'updated_text/mobile', normalization='legacy', use_cuda=True)
    server = ServerPPOcrV5Backend(worker_cache/'updated_text/server', profile='server_full', use_cuda=True)
    view_source = sha(ROOT/'pcb/component/recovery/text_candidate_views.py')
    roots = (GLOBAL_CACHE/'mobile', Path('E:/EDA/EDA/runs/component_alternative_cache/ocr_paired'))
    hits = views = 0
    for cid in ids:
        image = cv2.imdecode(np.fromfile(inputs[cid][0], np.uint8), 1)
        if image is None:
            raise ValueError('Unreadable allowed image: '+cid)
        key = hashlib.sha256(image.tobytes()+mobile.signature).hexdigest()[:24]+'.paired.json'
        dst = worker_cache/'updated_text/mobile'/key
        if not dst.is_file():
            for root in roots:
                source = root/key
                if source.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dst)
                    hits += 1
                    break
        vk = hashlib.sha256(image.tobytes()+server.signature+view_source.encode()).hexdigest()[:24]
        src = GLOBAL_CACHE/'views/rotation'/vk
        target = worker_cache/'updated_text/views/rotation'/vk
        if src.is_dir() and not target.exists():
            shutil.copytree(src, target)
            views += 1
    print(json.dumps({'event': 'observation_cache_ready', 'paired_hits': hits, 'rotation_image_hits': views}), flush=True)


def worker(args, inputs, guard):
    import torch
    from pcb.io import read_image, write_json, write_image
    from pcb.core.context import PipelineContext
    from tools.run_team_pipeline import build_runtime
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config, lock, key = version_lock()
    ids = IDS[args.shard::args.shards]
    args.output.mkdir(parents=True, exist_ok=True)
    version_path = args.output/f'version_worker{args.shard}.json'
    if version_path.exists() and read(version_path)['version_key'] != key:
        raise ValueError('Do not resume with changed source/config/model versions')
    write(version_path, {'version_key': key, **lock})
    print(json.dumps({'event': 'start', 'ACTUAL_CASE_IDS': ids, 'version_key': key, 'GT_opens': 0}), flush=True)
    seed_manifest = read(SEEDS)
    records, pending = [], []
    for cid in ids:
        folder = args.output/cid
        r = valid_receipt(folder, cid, key, inputs[cid][0])
        if r is None and folder.exists():
            raise ValueError('Incomplete case preserved; move it to an explicitly checked backup before resuming: '+cid)
        if r is None:
            r = seed_case(folder, cid, key, inputs[cid][0], seed_manifest)
        if r is None:
            pending.append(cid)
        else:
            records.append(r)
    cache = args.output/f'component_cache_worker{args.shard}'
    with guard.inferring():
        warm_observations(cache, inputs, pending)
        pipeline, ocr, detector, device = build_runtime(config, 'cuda', 'cuda', ROOT/'runs/team_cache')
    for cid in pending:
        start = time.perf_counter()
        print(json.dumps({'event': 'start_case', 'case_id': cid, 'GT_opens': guard.target_opens}), flush=True)
        def progress(message, **detail):
            print(json.dumps({'event': 'stage_progress', 'case_id': cid, 'message': message, **detail}), flush=True)
        with guard.inferring():
            image = read_image(inputs[cid][0])
            h, w = image.shape[:2]
            context = PipelineContext(inputs[cid][0].name, w, h, config, ocr, detector,
                resources={'pin': {'device': device, 'words_cache': ROOT/'runs/team_cache/pin_words'},
                           'component': {'cache_dir': cache}, 'progress': progress})
            result = pipeline.run(image, context)
            folder = args.output/cid
            write_json(folder/'scene.json', asdict(result.scene))
            write_json(folder/'result.json', result.result)
            write_image(folder/'wire_mask.png', result.wire.mask)
            r = {'case_id': cid, 'version_key': key, 'image_sha256': sha(inputs[cid][0]),
                 'provenance': 'current_source_image', 'seconds': time.perf_counter()-start,
                 'scene': sha(folder/'scene.json'), 'mask': sha(folder/'wire_mask.png'),
                 'prediction': sha(folder/'result.json')}
            write(folder/'receipt.json', r)
        records.append(r)
        print(json.dumps({'event': 'completed_case', 'case_id': cid, 'seconds': r['seconds'],
                          'GT_opens': guard.target_opens}), flush=True)
        write(args.output/f'progress_worker{args.shard}.json', {'cases': records, 'GT_opens': guard.target_opens,
              'actual_dataset_case_ids_opened': sorted(guard.actual_ids)})
    if source_inventory() != lock['source_sha256']:
        raise ValueError('Source/config changed during frozen inference')
    write(args.output/f'manifest_worker{args.shard}.json', {'ACTUAL_CASE_IDS': list(ids), 'cases': records,
          'version_key': key, 'GT_opens': guard.target_opens, 'actual_dataset_case_ids_opened': sorted(guard.actual_ids)})
    print(json.dumps({'event': 'worker_complete', 'shard': args.shard, 'case_count': len(records), 'GT_opens': guard.target_opens}), flush=True)


def merge(args, inputs, guard):
    versions, cases = [], {}
    for shard in range(args.shards):
        v = read(args.output/f'version_worker{shard}.json')
        m = read(args.output/f'manifest_worker{shard}.json')
        if m['ACTUAL_CASE_IDS'] != list(IDS[shard::args.shards]) or m['GT_opens'] or v['GT_opens']:
            raise ValueError('Incomplete/unsafe shard')
        versions.append(v)
        for r in m['cases']:
            cid = r['case_id']
            if cid in cases or cid not in IDS:
                raise ValueError('Duplicate/invalid case')
            cases[cid] = valid_receipt(args.output/cid, cid, v['version_key'], inputs[cid][0])
    if list(sorted(cases)) != list(IDS) or any(v != versions[0] for v in versions):
        raise ValueError('Different versions or missing current cases')
    manifest = {**versions[0], 'ACTUAL_CASE_IDS': list(IDS), 'cases': [cases[c] for c in IDS], 'GT_opens': 0}
    write(args.output/'manifest.json', manifest)
    print(json.dumps({'event': 'current150_sealed', 'ACTUAL_CASE_IDS': IDS, 'version_key': manifest['version_key'], 'GT_opens': 0}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('worker', 'merge'), required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=2)
    args = p.parse_args()
    if not 0 <= args.shard < args.shards <= 2:
        raise ValueError('Require one or two disjoint shards')
    guard = FreezeGuard()
    inputs = discover(args.dataset)
    (worker if args.stage == 'worker' else merge)(args, inputs, guard)


if __name__ == '__main__':
    main()
