"""Validate every official image/CSV and save immutable stratified split + caches."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    data = ROOT / 'data'
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    cache = data / 'cache'
    cache.mkdir(exist_ok=True)
    report = {'seed': 2026, 'files': {}, 'datasets': {}}
    tables = {}
    for name, fields, count in [('train_labels.csv',['id','filename','label'],50000), ('test.csv',['id','filename'],2000), ('sample_submission.csv',['id','label'],2000)]:
        actual, rows = read_csv(data/name)
        assert actual == fields, (name, actual)
        assert len(rows) == count
        ids = [int(r['id']) for r in rows]
        assert len(set(ids)) == count and set(ids) == set(range(count))
        report['files'][name] = {'sha256':digest(data/name),'rows':count,'columns':actual}
        tables[name] = rows
    assert read_csv(ROOT/'sample_submission.csv') == read_csv(data/'sample_submission.csv')
    assert {r['id'] for r in tables['test.csv']} == {r['id'] for r in tables['sample_submission.csv']}
    for split, table in [('train','train_labels.csv'),('test','test.csv')]:
        rows = tables[table]
        folder = data / (split+'_images')
        names = [r['filename'] for r in rows]
        assert len(set(names)) == len(rows)
        assert {p.name for p in folder.iterdir() if p.is_file()} == set(names)
        images = np.lib.format.open_memmap(cache/(split+'.npy'), mode='w+', dtype=np.uint8, shape=(len(rows),32,32,3))
        manifest = hashlib.sha256()
        hashes = []
        for i,r in enumerate(rows):
            path = folder / r['filename']
            assert path.parent == folder
            with Image.open(path) as im:
                assert im.mode == 'RGB' and im.size == (32,32), str(path)
                a = np.array(im)
                images[i] = a
            h = hashlib.sha256(a.tobytes()).hexdigest()
            hashes.append(h)
            manifest.update((r['id']+','+r['filename']+','+h+'\n').encode())
            if (i+1)%10000==0: print(split, i+1, flush=True)
        images.flush()
        np.save(cache/(split+'_ids.npy'), np.array([int(r['id']) for r in rows]))
        report['datasets'][split] = {'count':len(rows),'shape':[32,32,3],'mode':'RGB','manifest_sha256':manifest.hexdigest(),'duplicate_pixel_images':len(hashes)-len(set(hashes))}
        if split == 'train': train_hashes = hashes
    labels = np.array([int(r['label']) for r in tables['train_labels.csv']],dtype=np.int64)
    assert set(labels.tolist()) == set(range(10))
    counts = np.bincount(labels,minlength=10)
    assert (counts == 5000).all()
    np.save(cache/'labels.npy',labels)
    rng = np.random.default_rng(2026)
    train, val = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(labels == c))
        val.extend(idx[:500].tolist()); train.extend(idx[500:].tolist())
    train, val = np.array(sorted(train)), np.array(sorted(val))
    assert not set(train).intersection(val)
    assert len(train)==45000 and len(val)==5000
    overlap = set(train_hashes[i] for i in train) & set(train_hashes[i] for i in val)
    report['split'] = {'train':len(train),'validation':len(val),'train_class_counts':np.bincount(labels[train]).tolist(),'validation_class_counts':np.bincount(labels[val]).tolist(),'exact_pixel_overlap_count':len(overlap)}
    assert not overlap, 'Exact duplicate images cross validation boundary; use grouped split.'
    np.savez(out/'split_seed2026.npz',train=train,val=val)
    report['split']['sha256'] = digest(out/'split_seed2026.npz')
    report['status'] = 'passed'
    (out/'data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__': main()
