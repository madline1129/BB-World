"""Original nuScenes instance labels; no future annotation enters model inputs."""
import argparse
import ast
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle']


def rotation(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose(frame):
    t = np.eye(4)
    t[:3, :3] = rotation(frame['ego2global_rotation'])
    t[:3, 3] = frame['ego2global_translation']
    return t


def actor_class(name):
    if name in ('human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.police_officer', 'human.pedestrian.construction_worker'):
        return CLASSES.index('pedestrian')
    if name == 'vehicle.construction':
        return CLASSES.index('construction_vehicle')
    for c in CLASSES:
        if name == 'vehicle.' + c or (c == 'bus' and name.startswith('vehicle.bus.')):
            return CLASSES.index(c)
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--splits-source', required=True)
    args = p.parse_args()
    root, out = Path(args.metadata), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    def table(name):
        return json.loads((root / (name+'.json')).read_text())
    cat = {x['token']: x['name'] for x in table('category')}
    inst = {x['token']: actor_class(cat[x['category_token']]) for x in table('instance')}
    samples = {x['token']: x for x in table('sample')}
    scenes = {x['name']: x for x in table('scene')}
    official = {}
    for node in ast.parse(Path(args.splits_source).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    official[target.id] = ast.literal_eval(node.value)
    official['train'] = official['train_detect'] + official['train_track']
    anns = defaultdict(dict)
    for a in table('sample_annotation'):
        c = inst[a['instance_token']]
        if c is not None:
            anns[a['sample_token']][a['instance_token']] = (a['translation'], c, a['num_lidar_pts']+a['num_radar_pts'])
    report = {'classes': CLASSES, 'history': 5, 'future': 6, 'coordinate_frame': 'fifth input ego, full SE(3)',
              'labels': 'original global sample_annotation.translation joined by instance_token and exact sample_token',
              'selection': 'current center in Occ XYZ range and current lidar+radar points>0; future missing annotations masked; no future ROI clipping',
              'scope': 'current actors only, no future entrants', 'sources': {}, 'splits': {}}
    scene_sets, token_sets = [], []
    for split, name in [('train', 'm0.bin'), ('val', 'm1.bin')]:
        source = Path(args.cache)/name
        with source.open('rb') as f:
            infos = pickle.load(f)['infos']
        scene_sets.append(set(infos))
        assert set(infos) == set(official[split]), 'Scene split differs from official nuScenes train/val'
        token_sets.append({f['token'] for frames in infos.values() for f in frames})
        report['sources'][name] = sha256(source)
        windows, counts, class_counts = [], [], Counter()
        frame_count = 0
        intervals = []
        for scene, frames in sorted(infos.items()):
            frames = sorted(frames, key=lambda x: x['timestamp'])
            assert frames[0]['token'] == scenes[scene]['first_sample_token']
            assert frames[-1]['token'] == scenes[scene]['last_sample_token']
            assert len(frames) == scenes[scene]['nbr_samples']
            frame_count += len(frames)
            assert len({f['token'] for f in frames}) == len(frames)
            for a, b in zip(frames, frames[1:]):
                assert samples[a['token']]['next'] == b['token'], (scene, 'nonconsecutive frames')
                dt = (b['timestamp']-a['timestamp'])/1e6
                assert dt > 0
                intervals.append(dt)
            for i in range(4, len(frames)-6):
                window = frames[i-4:i+7]
                tokens = [f['token'] for f in window]
                paths = [str(Path(args.cache)/'s0/gts'/scene/t/'labels.npz') for t in tokens]
                assert all(Path(x).is_file() for x in paths), (scene, tokens)
                current = frames[i]
                inv = np.linalg.inv(pose(current))
                ids, classes, centers, future, valid = [], [], [], [], []
                for instance, (xyz, cls, npts) in sorted(anns[current['token']].items()):
                    center = (inv @ np.r_[xyz, 1])[:3]
                    if npts == 0 or not (-40 <= center[0] < 40 and -40 <= center[1] < 40 and -1 <= center[2] < 5.4):
                        continue
                    trajectory, mask = np.zeros((6, 3), np.float32), np.zeros(6, bool)
                    for t, token in enumerate(tokens[5:]):
                        if instance in anns[token]:
                            trajectory[t] = (inv @ np.r_[anns[token][instance][0], 1])[:3]
                            mask[t] = True
                    ids.append(instance); classes.append(cls); centers.append(center); future.append(trajectory); valid.append(mask)
                    class_counts[CLASSES[cls]] += 1
                counts.append(len(ids))
                windows.append({'scene': scene, 'token': current['token'], 'history_paths': paths[:5],
                                'history_tokens': tokens[:5], 'future_tokens': tokens[5:],
                                'current_to_history': np.stack([np.linalg.inv(pose(f)) @ pose(current) for f in window[:5]]).astype(np.float32),
                                'history_times': np.array([(f['timestamp']-current['timestamp'])/1e6 for f in window[:5]], np.float32),
                                'instance_ids': ids, 'classes': np.array(classes, np.int64),
                                'centers': np.array(centers, np.float32).reshape(-1, 3),
                                'future': np.array(future, np.float32).reshape(-1, 6, 3),
                                'valid': np.array(valid, bool).reshape(-1, 6)})
        assert max(counts) <= 256, 'Increase query capacity; never truncate actors silently'
        with (out/(split+'.pkl')).open('wb') as f:
            pickle.dump(windows, f, protocol=4)
        report['splits'][split] = {'scenes': len(infos), 'frames': frame_count, 'windows': len(windows),
                                  'max_actors': max(counts), 'mean_actors': float(np.mean(counts)), 'class_counts': dict(class_counts),
                                  'keyframe_interval_range_s': [min(intervals), max(intervals)],
                                  'manifest_sha256': sha256(out/(split+'.pkl'))}
        print(json.dumps({split: report['splits'][split]}), flush=True)
    assert not scene_sets[0] & scene_sets[1]
    assert not token_sets[0] & token_sets[1]
    report['sources']['sample_annotation.json'] = sha256(root/'sample_annotation.json')
    report['sources']['official_splits.py'] = sha256(args.splits_source)
    report['protocol_note'] = 'Official 700/150 scenes, all original sample chains verified, complete 5+6 windows (21130/4519). Brain actor metrics are auxiliary; no end-to-end Occ/II-World performance claim.'
    (out/'audit.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
