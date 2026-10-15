"""Validate the prepared 50 × 50 collection rather than silently padding counts."""
from collections import Counter
import hashlib
from clip_explorer.collections import collection_photos
from clip_explorer.config import ROOT


def test_large_collection_has_fifty_distinct_photos_per_category():
    photos=collection_photos('Caltech · 50 × 50')
    counts=Counter(r['category'] for r in photos)
    assert len(counts)==50
    assert set(counts.values())=={50}
    assert len({r['id'] for r in photos})==2500
    assert len({r['sha256'] for r in photos})==2500
    for record in photos:
        assert hashlib.sha256((ROOT/'assets'/record['file']).read_bytes()).hexdigest()==record['sha256']
        assert record['source']=='https://data.caltech.edu/records/mzrjq-6wc02'
        assert record['license']=='CC BY 4.0'
