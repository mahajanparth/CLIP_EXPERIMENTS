"""Offline camera-facing photo markers at unchanged joint projection coordinates."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .data import open_asset
from .process_visuals import image_uri
from .embedding_space import COLORS


def thumbnail_document(points: pd.DataFrame, records: list[dict], method: str,
                       connect_pairs: bool = True) -> str:
    dimensions = 3 if 'Z' in points else 2
    photos = {}
    for record in records:
        photo = open_asset(record)
        photo.thumbnail((160,120))
        photos[record['id']] = image_uri(photo)
    payload = dict(dimensions=dimensions, method=method, pairCount=len(records),
                   connect=connect_pairs, colors=COLORS, photos=photos,
                   credits=[{k:r.get(k, "") for k in ("id", "author", "license", "source")} for r in records],
                   points=points.to_dict(orient='records'))
    # Escape HTML script terminators even when captions contain arbitrary user text.
    data = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('thumbnail_space.html').read_text()
    return template.replace('__PAYLOAD__', data)
