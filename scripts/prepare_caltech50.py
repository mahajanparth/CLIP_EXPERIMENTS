"""Prepare a reproducible 50 × 50 photographic subset from official Caltech-101."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import tarfile
import zipfile
from io import BytesIO
from urllib.request import urlretrieve
from PIL import Image
from clip_explorer.config import ROOT

SOURCE='https://data.caltech.edu/records/mzrjq-6wc02'
URL=SOURCE+'/files/caltech-101.zip?download=1'
MD5='3138e1922a9193bfa496528edbbc45d0'
AUTHOR="Fei-Fei Li; Marco Andreeto; Marc'Aurelio Ranzato; Pietro Perona (Caltech-101 dataset creators)"
# Exclude background, redundant face collections, cartoons, symbols and scans.
EXCLUDE={'BACKGROUND_Google','Faces','Faces_easy','garfield','yin_yang','brain','dollar_bill','snoopy','stegosaurus'}
LABELS={'airplanes':'airplane','Motorbikes':'motorbike','Leopards':'leopard','accordion':'accordion',
        'cougar_body':'cougar','cougar_face':'cougar face','crab':'crab','crocodile_head':'crocodile head',
        'elephant':'elephant','ewer':'ewer pitcher','helicopter':'helicopter','inline_skate':'inline skate',
        'nautilus':'nautilus shell','pyramid':'pyramid','schooner':'schooner sailboat'}


def main():
    cache=ROOT/'.cache/datasets';cache.mkdir(parents=True,exist_ok=True)
    archive=cache/'caltech-101.zip'
    if not archive.exists(): urlretrieve(URL,archive)
    if hashlib.md5(archive.read_bytes()).hexdigest()!=MD5:
        raise ValueError('Caltech archive checksum mismatch; delete cached archive and download again.')
    with zipfile.ZipFile(archive) as zipped:
        name=next(n for n in zipped.namelist() if n.endswith('101_ObjectCategories.tar.gz'))
        with tarfile.open(fileobj=BytesIO(zipped.read(name)),mode='r:gz') as tar:
            grouped={}
            for member in tar.getmembers():
                parts=Path(member.name).parts
                if member.isfile() and len(parts)==3 and parts[-1].lower().endswith('.jpg'):
                    grouped.setdefault(parts[-2],[]).append(member)
            eligible=sorted(c for c,items in grouped.items() if len(items)>=50 and c not in EXCLUDE)
            if len(eligible)<50:raise RuntimeError(f'Only {len(eligible)} eligible categories.')
            categories=eligible[:50];records=[]
            target=ROOT/'assets/caltech50';target.mkdir(exist_ok=True)
            for category in categories:
                selected=sorted(grouped[category],key=lambda m:m.name)[:50]
                label=LABELS.get(category,category.replace('_',' ').lower())
                for member in selected:
                    ident='caltech50_'+category+'_'+Path(member.name).stem
                    payload=tar.extractfile(member).read()
                    with Image.open(BytesIO(payload)) as image:
                        image.verify()
                    destination=target/(ident+'.jpg');destination.write_bytes(payload)
                    records.append(dict(id=ident,file='caltech50/'+destination.name,label=label,
                        category=label.title(),caption='a photo of a '+label,medium='photograph',
                        source=SOURCE,author=AUTHOR,license='CC BY 4.0',
                        license_url='https://creativecommons.org/licenses/by/4.0/',license_evidence=SOURCE,
                        dataset='Caltech-101',dataset_member=member.name,sha256=hashlib.sha256(payload).hexdigest()))
    metadata=dict(dataset='Caltech-101',source=SOURCE,archive_url=URL,archive_md5=MD5,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),license='CC BY 4.0',
        license_evidence='https://data.caltech.edu/api/records/mzrjq-6wc02',
        selection='First 50 alphabetically sorted eligible categories, first 50 filenames each; no repetitions or augmentation.',
        excluded_categories=sorted(EXCLUDE),categories=[LABELS.get(c,c.replace('_',' ').lower()).title() for c in categories],
        image_count=len(records),gallery=records)
    (ROOT/'assets/caltech50_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (ROOT/'assets/CALTECH50_CREDITS.md').write_text(
        '# Caltech-101 · 50 categories × 50 photographs\n\n'
        f'Dataset creators: {AUTHOR}. [Official source]({SOURCE}); [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). '
        'Original JPEG bytes are preserved. Each manifest record identifies the original archive member and SHA-256. '
        'No repeated images or synthetic augmentation are used to reach the count. This educational subset is not a benchmark split.\n\n'
        'Archive MD5: `'+MD5+'`. Categories with fewer than 50 files are excluded, along with background, redundant face collections, cartoons, symbols, and scans.\n\n'
        +'\n'.join('- '+c for c in metadata['categories'])+'\n')
    print(f'Prepared {len(records)} real images in {len(categories)} categories.',flush=True)

if __name__=='__main__':main()
