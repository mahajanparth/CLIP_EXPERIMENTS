"""One-time Wikimedia photographic gallery; retain exact sources and attribution.

Running again uses the locked source list in assets/photo_sources.json rather
than searching again, so an existing prepared gallery is reproducible.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from io import BytesIO
import hashlib
import html
import json
import re
from PIL import Image
from clip_explorer.config import ROOT

API='https://commons.wikimedia.org/w/api.php'
USER_AGENT='CLIPExplorer/1.0 (local educational image-text demonstration)'
SEARCHES=[
 ('dog','Animals','intitle:"Golden retriever"',4),
 ('cat','Animals','intitle:"Cat" -intitle:painting -intitle:sculpture',4),
 ('car','Vehicles','intitle:"Toyota Corolla"',4),
 ('bicycle','Vehicles','intitle:"Bicycle" -intitle:map -intitle:logo -intitle:museum',4),
 ('coffee','Food','intitle:"Coffee cup"',2),
 ('pizza','Food','intitle:"Pizza" -intitle:restaurant -intitle:logo',2),
 ('apple','Food','intitle:"Apples" -intitle:painting',2),
 ('mountains','Landscapes','intitle:"Mountain landscape"',2),
 ('forest','Landscapes','intitle:"Forest landscape"',2),
 ('beach','Landscapes','intitle:"Beach landscape"',2),
 ('chair','Household','intitle:"Chair" -intitle:painting -intitle:sculpture',2),
 ('clock','Household','intitle:"Wall clock"',2),
 ('robot','Robotics','intitle:"Robot" -intitle:logo -intitle:diagram -intitle:art',2),
]


def plain(value: str) -> str:
    return html.unescape(re.sub('<[^>]*>','',value)).strip()


def fetch(url: str) -> bytes:
    return urlopen(Request(url,headers={'User-Agent':USER_AGENT}),timeout=40).read()


def discover() -> list[dict]:
    selected=[];used=set()
    for label,category,term,count in SEARCHES:
        params={'action':'query','format':'json','generator':'search','gsrsearch':term+' filetype:bitmap',
                'gsrnamespace':6,'gsrlimit':30,'prop':'imageinfo','iiprop':'url|extmetadata|mime|size','iiurlwidth':720}
        response=json.loads(fetch(API+'?'+urlencode(params)))
        pages=sorted(response.get('query',{}).get('pages',{}).values(),key=lambda p:p.get('index',1000))
        found=0
        for page in pages:
            info=page.get('imageinfo',[{}])[0];meta=info.get('extmetadata',{})
            license_name=plain(meta.get('LicenseShortName',{}).get('value',''))
            if not (license_name.startswith('CC BY') or license_name in ['CC0','Public domain']):continue
            if info.get('mime')!='image/jpeg' or min(info.get('width',0),info.get('height',0))<350:continue
            if page['pageid'] in used:continue
            author=plain(meta.get('Artist',{}).get('value',''))
            if not author:continue
            selected.append({'id':f'commons_{label}_{page["pageid"]}','file':f'gallery/commons_{label}_{page["pageid"]}.jpg',
                'label':label,'category':category,'caption':f'a photo of a {label}', 'medium':'photograph',
                'source':info['descriptionurl'],'download_url':info.get('thumburl',info['url']),
                'original_url':info['url'],'author':author,'license':license_name,
                'license_url':meta.get('LicenseUrl',{}).get('value',''),
                'license_evidence':info['descriptionurl'],'commons_title':page['title'],
                'source_description':plain(meta.get('ImageDescription',{}).get('value',''))})
            used.add(page['pageid']);found+=1
            if found==count:break
        print(f'Selected {found}/{count} photographs for {label}',flush=True)
        if found<count:raise RuntimeError(f'Insufficient licensed JPEG photographs for {label}; inspect search results.')
    return selected


def main():
    source_path=ROOT/'assets/photo_sources.json'
    if source_path.exists():sources=json.loads(source_path.read_text())
    else:
        raise FileNotFoundError('The locked assets/photo_sources.json source list is required to reproduce this curated gallery.')
    manifest_path=ROOT/'assets/manifest.json'
    data=json.loads(manifest_path.read_text())
    for source in sources:
        record=dict(source);path=ROOT/'assets'/record['file']
        if not path.exists():
            content=fetch(record['download_url'])
            with Image.open(BytesIO(content)) as image:image.verify()
            path.write_bytes(content)
        record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        data['gallery']=[r for r in data['gallery'] if r['id']!=record['id']]+[record]
        manifest_path.write_text(json.dumps(data,indent=2)+'\n')
        print(f'Prepared {record["id"]} · {record["license"]}',flush=True)
    # Machine-readable source list and a visible human-readable attribution sheet.
    credits=['# Real photograph credits','',
        'Wikimedia Commons photographic assets are downloaded once as resized JPEGs. Original file pages, authors, license links, and hashes are preserved in manifest.json. Photo pixels are not edited by this project. Source selection is locked in photo_sources.json.','',
        '| Image | Author | License | Source |','|---|---|---|---|']
    for r in data['gallery']:
        if r['medium']!='photograph':continue
        author=r['author'].replace('|',' / ').replace('\n',' ')
        license_text=f'[{r["license"]}]({r["license_url"]})' if r.get('license_url') else r['license']
        credits.append(f'| {r["id"]} | {author} | {license_text} | [Original file]({r["source"]}) |')
    (ROOT/'assets/PHOTO_CREDITS.md').write_text('\n'.join(credits)+'\n')
    print(f'Gallery now has {sum(r["medium"]=="photograph" for r in data["gallery"])} real photographs.')

if __name__=='__main__':main()
