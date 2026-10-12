"""Export an offline stage viewer and complete real CLIP text activation arrays."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from html import escape
import argparse
import json
import numpy as np
import plotly.graph_objects as go
from clip_explorer.config import ROOT,MODEL_NAME,MODEL_REVISION
from clip_explorer.text_activations import inspect_text_prompt
from clip_explorer.activation_lab import activation_heatmap


def export(prompt: str = 'a photo of a golden retriever') -> None:
    trace=inspect_text_prompt(prompt)
    names=list(trace['stages'])
    figures=[activation_heatmap(trace,name) for name in names]
    figure=figures[0]
    buttons=[]
    for name,observed in zip(names,figures):
        heat=observed.data[0]
        buttons.append(dict(label=name,method='update',args=[
            {'z':[list(heat.z)],'x':[list(heat.x)],'y':[list(heat.y)],'zmin':heat.zmin,'zmax':heat.zmax},
            {'title':{'text':observed.layout.title.text},'yaxis.autorange':'reversed'}]))
    figure.update_layout(updatemenus=[dict(type='dropdown',x=0,y=1.18,buttons=buttons)],margin=dict(l=190,r=20,t=110,b=60))
    plot=figure.to_html(full_html=False,include_plotlyjs=True)
    target=ROOT/'artifacts'
    np.savez_compressed(target/'text_activations.npz',**trace['stages'])
    metadata={'model':MODEL_NAME,'model_revision':MODEL_REVISION,'prompt':prompt,'token_ids':trace['token_ids'].tolist(),
              'token_pieces':trace['token_pieces'],'eos_index':trace['eos_index'],'stages':{name:list(values.shape) for name,values in trace['stages'].items()},
              'source':'Actual frozen forward pass; temporary hooks; no gradients','display':'First 64 coordinates; padding omitted; full tensors in NPZ'}
    (target/'text_activations_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLIP · Text activation inspector</title><style>body{margin:0;background:#f5f7fa;color:#172b42;font:17px/1.6 Arial,sans-serif}main{max-width:1300px;margin:auto;padding:32px}h1{letter-spacing:-1px}.note{padding:16px;background:#e6f1ef;border-left:4px solid #137c75}</style></head><body><main><h1>Actual CLIP text feature activations</h1>'''
    document+=f'<p><strong>Prompt:</strong> {escape(prompt)}</p><p>{trace["eos_index"]+1} tokens including start/end; 77 context positions; 512 hidden features. Use the stage dropdown to inspect token lookup, positions, every Transformer block, and EOS readout. The heatmap displays the first 64 coordinates; the NPZ file contains all coordinates and padding positions.</p>'
    document+=plot+'<p class="note">Measured original OpenAI CLIP ViT-B/32 values. The displayed color range changes with the stage. Activations are not attention weights, saliency, or causal token importance. Only the end-of-text state is projected into the shared image–text space. BPE pieces are byte-level fragments; &lt;/w&gt; denotes a word boundary. Use the local Streamlit Text Activation Inspector to enter another prompt, inspect all coordinates, or compare prompts.</p></main></body></html>'
    (target/'text_activations.html').write_text(document)
    print('Exported actual text activations, metadata, and offline stage viewer.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',default='a photo of a golden retriever')
    export(parser.parse_args().prompt)
