import hashlib
import json
from collections import Counter
from PIL import Image
from clip_explorer.config import ROOT


def test_prepared_assets_are_present_and_hashes_match():
    data = json.loads((ROOT/'assets/manifest.json').read_text())
    records = data['gallery']+data['probes']
    assert len({r['id'] for r in records})==len(records)
    assert len(data['gallery'])>=30
    assert Counter(r['experiment'] for r in data['probes'])=={'Counting':30,'Spatial relationships':30,'Attribute binding':30}
    for r in records:
        path = ROOT/'assets'/r['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        assert r['source'] and r['license']
        with Image.open(path) as image: image.verify()


def test_probe_labels_are_balanced_and_distractors_differ():
    data = json.loads((ROOT/'assets/manifest.json').read_text())
    counts = [r['count'] for r in data['probes'] if r['experiment']=='Counting']
    assert Counter(counts)=={2:10,3:10,4:10}
    for r in data['probes']: assert r['correct']!=r['incorrect']


def test_main_demo_gallery_contains_real_photographs_with_credits():
    from clip_explorer.data import photographic_gallery,photo_for_label
    from clip_explorer.embedding_space import default_records
    photos=photographic_gallery()
    assert len(photos)>=30
    assert all(r['medium']=='photograph' and r.get('author') and r.get('license') and r.get('source') for r in photos)
    for label in ['dog','cat','car','bicycle','coffee','robot']:
        assert photo_for_label(label)['medium']=='photograph'
    paired=default_records(photos)
    assert len(paired)==12
    assert all(r['medium']=='photograph' for r in paired)
    assert len({r['id'] for r in paired})==len(paired)
    assert {r['category'] for r in paired}=={'Animals','Vehicles','Food'}
