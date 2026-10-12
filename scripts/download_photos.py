"""Optional small CC0/public-domain photographic supplements (download once)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
from clip_explorer.config import ROOT

PHOTO_SOURCES = [
    ('chelsea.png','cat','Animals','Stefan van der Walt','CC0-1.0'),
    ('coffee.png','coffee','Food','Rachel Michetti; courtesy of Pikolo Espresso Bar','CC0-1.0'),
    ('astronaut.png','astronaut','People','NASA','Public domain'),
    ('rocket.jpg','rocket','Vehicles','SpaceX','Public domain'),
]


def download() -> None:
    manifest_path = ROOT/'assets/manifest.json'
    data = json.loads(manifest_path.read_text())
    for filename,label,category,author,license_name in PHOTO_SOURCES:
        url = f'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/{filename}'
        content = urlopen(url,timeout=30).read()
        # Validate file content before storing the upstream bytes.
        with Image.open(BytesIO(content)) as image: image.verify()
        target = ROOT/'assets/gallery'/filename
        target.write_bytes(content)
        record = {'id':f'photo_{label}','file':f'gallery/{filename}','label':label,'category':category,'caption':f'a photo of a {label}','medium':'photograph','source':url,'author':author,'license':license_name,'license_evidence':'https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/_fetchers.py','sha256':hashlib.sha256(content).hexdigest()}
        data['gallery'] = [r for r in data['gallery'] if r['id']!=record['id']] + [record]
        print(f'Downloaded {filename} ({license_name})',flush=True)
    manifest_path.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__': download()
