"""Deterministic Pillow drawings: original project artwork, released CC0-1.0."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import random
from PIL import Image, ImageDraw
from clip_explorer.config import ROOT, SEED

SIZE = 512
BG = '#f5f0e8'


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new('RGB', (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 488, 488), radius=30, fill='#e6e9e7')
    draw.ellipse((100, 390, 410, 434), fill='#cbd0ca')
    return image, draw


def cube(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, size: int = 88) -> None:
    draw.rectangle((x, y, x + size, y + size), fill=color, outline='#354252', width=3)
    draw.polygon([(x,y), (x+22,y-22), (x+size+22,y-22), (x+size,y)], fill=color, outline='#354252')
    draw.polygon([(x+size,y), (x+size+22,y-22), (x+size+22,y+size-22), (x+size,y+size)], fill=color, outline='#354252')


def sphere(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, size: int = 92) -> None:
    draw.ellipse((x,y,x+size,y+size), fill=color, outline='#354252', width=3)
    draw.ellipse((x+18,y+15,x+38,y+35), fill='#efefef')


def apple(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 70, color: str = '#d13b43') -> None:
    draw.ellipse((x,y,x+size,y+size), fill=color)
    draw.line((x+size//2,y+3,x+size//2+4,y-14), fill='#68452c', width=5)
    draw.ellipse((x+size//2+3,y-14,x+size//2+23,y-4), fill='#4c8150')


def illustration(kind: str, variation: int) -> Image.Image:
    image, d = canvas()
    accent = ['#db8851', '#88755e'][variation]
    if kind in ['dog', 'cat', 'rabbit', 'bear', 'fox']:
        d.ellipse((145,185,365,395), fill=accent)
        if kind == 'rabbit':
            d.ellipse((155,75,220,235), fill=accent); d.ellipse((290,75,350,235), fill=accent)
        elif kind in ['cat', 'fox']:
            d.polygon([(145,230),(155,110),(240,210)], fill=accent); d.polygon([(280,210),(350,110),(370,230)], fill=accent)
        else:
            d.ellipse((110,165,185,300), fill=accent); d.ellipse((335,165,405,300), fill=accent)
        d.ellipse((196,251,211,269), fill='#273440'); d.ellipse((296,251,311,269), fill='#273440')
        d.ellipse((220,285,290,350), fill='#f6dfba'); d.ellipse((245,292,266,310), fill='#273440')
        if kind == 'cat':
            for dy in [0,15]:
                d.line((230,320+dy,160,310+dy),fill='#273440',width=3); d.line((282,320+dy,354,310+dy),fill='#273440',width=3)
    elif kind in ['car', 'bus', 'truck', 'bicycle', 'boat']:
        if kind == 'bicycle':
            for x in [100,300]: d.ellipse((x,290,x+105,395),outline='#273440',width=10)
            d.line([(152,342),(225,235),(310,342),(152,342),(210,285),(310,342),(290,225)],fill='#137c75',width=9)
            d.line((195,230,238,230),fill='#273440',width=8)
        elif kind == 'boat':
            d.polygon([(100,310),(410,310),(350,395),(165,395)],fill=accent)
            d.line((255,130,255,310),fill='#273440',width=7); d.polygon([(265,140),(265,290),(375,290)],fill='#fafafa')
            for yy in [420,444]: d.line((65,yy,450,yy),fill='#619ab0',width=5)
        else:
            top = 175 if kind == 'bus' else 230
            d.rounded_rectangle((95,top,420,365), radius=25, fill=accent)
            if kind == 'car': d.polygon([(160,230),(200,165),(315,165),(365,230)],fill=accent)
            for x in [140,290]: d.ellipse((x,330,x+65,395), fill='#273440'); d.ellipse((x+18,348,x+47,377),fill='#d9e2e9')
            d.rectangle((195,top+12,310,top+60),fill='#a1ccd3')
    elif kind in ['mountains', 'forest', 'ocean', 'desert', 'lake']:
        d.rectangle((35,65,477,300),fill='#b8dce3'); d.ellipse((375,90,430,145),fill='#f4c76c')
        if kind in ['mountains', 'lake']:
            d.polygon([(40,320),(180,120),(315,320)],fill='#768d88'); d.polygon([(200,320),(350,145),(477,320)],fill='#92a79c')
            d.polygon([(142,175),(180,120),(218,175)],fill='#f7faf9')
        if kind == 'forest':
            for x in [80,175,285,390]:
                d.rectangle((x+20,250,x+35,410),fill='#7b6143'); d.polygon([(x-20,300),(x+25,140),(x+70,300)],fill='#477968')
        if kind in ['ocean', 'lake']: d.rectangle((35,330,477,470), fill='#729faf')
        if kind == 'desert':
            d.polygon([(35,375),(150,255),(320,390),(477,270),(477,470),(35,470)],fill='#dfb880')
    elif kind in ['apple', 'pizza', 'cake', 'banana', 'coffee']:
        d.ellipse((105,180,410,410),fill='#fafafa',outline='#c2cbc8',width=6)
        if kind == 'apple': apple(d,205,235,110)
        elif kind == 'pizza':
            d.ellipse((150,200,365,400),fill='#d99354'); d.ellipse((165,215,350,385),fill='#f2d374')
            for x,y in [(205,250),(290,260),(250,320),(200,335)]: d.ellipse((x,y,x+28,y+28),fill='#b94d4a')
        elif kind == 'cake':
            d.rectangle((170,250,350,370),fill='#cf986f'); d.rectangle((170,240,350,270),fill='#f0dddd')
            d.line((258,240,258,185),fill='#137c75',width=8); d.ellipse((248,160,270,189),fill='#edb649')
        elif kind == 'banana':
            d.arc((170,170,360,370),20,165,fill='#e8bf47',width=40)
        else:
            d.ellipse((280,245,390,345),outline='#fafafa',width=18); d.rounded_rectangle((175,215,325,360),radius=22,fill='#fafafa')
            d.ellipse((175,205,325,250),fill='#78513c',outline='#fafafa',width=10)
    elif kind in ['robot', 'robot arm', 'robot vacuum', 'drone', 'rover']:
        if kind == 'robot vacuum':
            d.ellipse((145,225,375,390),fill='#627a89'); d.ellipse((165,235,355,350),fill='#a9b5bc'); d.ellipse((245,245,285,270),fill='#137c75')
        elif kind == 'drone':
            for x,y in [(120,170),(315,170),(120,320),(315,320)]:
                d.line((255,275,x+30,y+20),fill='#627a89',width=14); d.ellipse((x-15,y,x+80,y+35),fill='#364656')
            d.ellipse((205,230,305,310),fill='#a9b5bc')
        elif kind == 'robot arm':
            d.rectangle((155,375,375,410),fill='#627a89'); d.line([(255,375),(180,265),(295,170),(365,245)],fill=accent,width=35)
            for x,y in [(255,375),(180,265),(295,170)]: d.ellipse((x-22,y-22,x+22,y+22),fill='#627a89')
            d.line([(340,260),(365,245),(390,260)],fill='#364656',width=9)
        else:
            d.rounded_rectangle((150,120,360,250),radius=20,fill='#a9b5bc'); d.rectangle((175,270,335,375),fill='#627a89')
            for x in [195,285]: d.ellipse((x,165,x+30,195),fill='#137c75'); d.rectangle((x,375,x+30,420),fill='#a9b5bc')
            d.line((190,220,310,220),fill='#364656',width=7)
    else:
        if kind == 'chair':
            d.rectangle((160,150,350,290),fill=accent); d.rectangle((155,290,355,320),fill=accent)
            for x in [175,325]: d.rectangle((x,315,x+15,425),fill='#694f3f')
        elif kind == 'lamp':
            d.line((255,200,255,395),fill='#627a89',width=12); d.polygon([(185,130),(325,130),(375,260),(135,260)],fill='#e2bb6b'); d.ellipse((190,390,325,420),fill='#627a89')
        elif kind == 'clock':
            d.ellipse((130,140,385,395),fill='#fafafa',outline='#627a89',width=14); d.line([(255,190),(255,270),(320,305)],fill='#273440',width=10)
        elif kind == 'bottle':
            d.rounded_rectangle((195,210,315,405),radius=25,fill='#137c75'); d.rectangle((225,140,285,240),fill='#137c75'); d.rectangle((220,135,290,160),fill='#627a89')
        else:
            d.rectangle((155,170,360,410),fill=accent,outline='#694f3f',width=6); d.rectangle((175,185,345,395),fill='#f6dfba'); d.line((185,200,185,380),fill='#694f3f',width=5)
    return image


def record(image: Image.Image, file: str, **metadata: object) -> dict:
    path = ROOT / 'assets' / file
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {'file': file, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source': 'Original deterministic Pillow artwork, scripts/generate_assets.py', 'license': 'CC0-1.0', **metadata}


def generate() -> dict:
    rng = random.Random(SEED)
    manifest = {'schema': 1, 'seed': SEED, 'gallery': [], 'probes': []}
    categories = {
        'Animals': ['dog','cat','rabbit','bear','fox'],
        'Vehicles': ['car','bicycle','bus','truck','boat'],
        'Landscapes': ['mountains','forest','ocean','desert','lake'],
        'Food': ['apple','pizza','cake','banana','coffee'],
        'Robotics': ['robot','robot arm','robot vacuum','drone','rover'],
        'Household': ['chair','lamp','clock','bottle','book'],
    }
    for category, kinds in categories.items():
        for kind in kinds:
            for variant in range(2):
                ident = f'{kind.replace(" ", "_")}_{variant}'
                manifest['gallery'].append(record(illustration(kind,variant),f'gallery/{ident}.png',id=ident,category=category,label=kind,caption=f'a drawing of a {kind}',medium='illustration'))
    words = {2:'Two',3:'Three',4:'Four'}
    for idx in range(30):
        count = [2,3,4][idx % 3]
        wrong = [n for n in [2,3,4] if n != count][(idx//3)%2]
        image,d = canvas()
        d.rounded_rectangle((45,235,467,430),radius=12,fill='#bf9c76')
        size = 60 + rng.randrange(8)
        for i in range(count): apple(d,80+i*90,270+rng.randrange(-12,13),size)
        manifest['probes'].append(record(image,f'probes/counting_{idx:02}.png',id=f'counting_{idx:02}',experiment='Counting',correct=f'{words[count]} apples on a table.',incorrect=f'{words[wrong]} apples on a table.',count=count))
    for experiment in ['Spatial relationships','Attribute binding']:
        for idx in range(30):
            image,d = canvas()
            dy = (idx//2 % 5)*9
            if experiment == 'Spatial relationships':
                left = idx%2 == 0
                cube(d,110 if left else 305,260+dy,'#d13b43')
                sphere(d,305 if left else 110,260+dy,'#3676c9')
                correct = f'A red cube to the {"left" if left else "right"} of a blue sphere.'
                incorrect = f'A red cube to the {"right" if left else "left"} of a blue sphere.'
                ident = f'spatial_{idx:02}'
            else:
                red = idx%2 == 0
                # Match paired examples in geometry; counterbalance cube position across pairs.
                x1,x2 = (110,305) if idx//2%2 == 0 else (305,110)
                cube(d,x1,260+dy,'#d13b43' if red else '#3676c9')
                sphere(d,x2,260+dy,'#3676c9' if red else '#d13b43')
                correct = f'A {"red" if red else "blue"} cube beside a {"blue" if red else "red"} sphere.'
                incorrect = f'A {"blue" if red else "red"} cube beside a {"red" if red else "blue"} sphere.'
                ident = f'binding_{idx:02}'
            # Five vertical offsets and three background colors; 15 matched pairs.
            background = ['#e6e9e7','#e8e1d5','#dfe8ef'][idx//10]
            # Replace only the canvas panel color, preserving all object pixels.
            pixels = image.load()
            for y in range(25,488):
                for x in range(25,488):
                    if pixels[x,y] == (230,233,231): pixels[x,y] = tuple(bytes.fromhex(background[1:]))
            manifest['probes'].append(record(image,f'probes/{ident}.png',id=ident,experiment=experiment,correct=correct,incorrect=incorrect,pair_id=idx//2))
    path = ROOT / 'assets' / 'manifest.json'
    # Preserve separately downloaded photos when regenerating original artwork.
    if path.exists():
        previous = json.loads(path.read_text())
        manifest['gallery'].extend(x for x in previous['gallery'] if x.get('medium') == 'photograph')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__ == '__main__':
    data = generate()
    print(f'Prepared {len(data["gallery"])} gallery images and {len(data["probes"])} labeled probes.')
