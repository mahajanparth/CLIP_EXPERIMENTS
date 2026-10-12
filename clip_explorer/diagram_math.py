"""Render equations as self-contained SVG paths; no browser LaTeX dependency."""
from __future__ import annotations
from functools import lru_cache
from io import BytesIO
import threading
import xml.etree.ElementTree as ET

_LOCK=threading.RLock()


@lru_cache(maxsize=256)
def _formula_svg(expression: str, color: str) -> tuple[str,float,float]:
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    buffer=BytesIO()
    with _LOCK,matplotlib.rc_context({'svg.fonttype':'path','svg.hashsalt':'clip-explorer',
                                    'figure.facecolor':'none','savefig.facecolor':'none'}):
        math_to_image('$'+expression+'$',buffer,prop=FontProperties(size=18),format='svg',color=color)
    ET.register_namespace('','http://www.w3.org/2000/svg')
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    root=ET.fromstring(buffer.getvalue())
    x,y,width,height=map(float,root.attrib['viewBox'].split())
    root.attrib.pop('width',None);root.attrib.pop('height',None)
    return ET.tostring(root,encoding='unicode'),width,height


def math_svg(expression: str, x: float, y: float, width: float, height: float, color: str) -> str:
    """Center and fit vector typeset math without clipping a fraction or index."""
    source,natural_width,natural_height=_formula_svg(expression,color)
    scale=min(width/natural_width,height/natural_height)
    rendered_width,rendered_height=natural_width*scale,natural_height*scale
    root=ET.fromstring(source)
    root.set('x',str(x+(width-rendered_width)/2))
    root.set('y',str(y+(height-rendered_height)/2))
    root.set('width',str(rendered_width));root.set('height',str(rendered_height))
    title=ET.Element('{http://www.w3.org/2000/svg}title');title.text=expression;root.insert(0,title)
    return ET.tostring(root,encoding='unicode')
