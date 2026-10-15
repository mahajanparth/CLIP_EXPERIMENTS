"""Visual inspection of image patch features, class readout, and text matching."""
from io import BytesIO
import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import ImageDraw
from .image_activations import inspect_image_payload,stage_labels,patch_values,preprocessed_image
from .model import image_bytes,encode_text
from .collections import collection_names,collection_photos
from .ui import intro,takeaway,select_image,photo_credits


def image_heatmap(trace,stage,start=0,count=64):
    values=trace['stages'][stage][:,start:start+count]
    limit=max(float(np.abs(values).max()),1e-8)
    fig=go.Figure(go.Heatmap(z=values.tolist(),x=list(range(start,start+values.shape[1])),
        y=stage_labels(trace,stage),colorscale='RdBu',zmin=-limit,zmax=limit,
        hovertemplate='%{y}<br>Feature %{x}<br>Activation %{z:.6f}<extra></extra>'))
    fig.update_layout(title=f'{stage} · {list(trace["stages"][stage].shape)}',height=650,
        xaxis_title='Feature coordinate (zero-based)',yaxis_title='Patch or class token',yaxis_autorange='reversed',margin=dict(l=150))
    return fig



def image_document(trace,image,record=None,prompt='',cosine=None):
    """Self-contained snapshot with synchronized stage heatmaps and patch maps."""
    from html import escape
    from plotly.subplots import make_subplots
    from .process_visuals import image_uri
    names=list(trace['stages']);observed=[image_heatmap(trace,name) for name in names]
    fig=make_subplots(rows=1,cols=2,column_widths=[.72,.28],subplot_titles=['Token × feature activations','Spatial patch RMS'])
    fig.add_trace(observed[0].data[0],row=1,col=1)
    maps=[]
    for name in names:
        maps.append(patch_values(trace,name).tolist() if trace['stages'][name].shape[0]>1 else [[None]*7 for _ in range(7)])
    fig.add_trace(go.Heatmap(z=maps[0],x=list(range(7)),y=list(range(7)),colorscale='Teal',showscale=False),row=1,col=2)
    buttons=[]
    for name,chart,spatial in zip(names,observed,maps):
        heat=chart.data[0]
        buttons.append(dict(label=name,method='update',args=[
            {'z':[list(heat.z),spatial],'x':[list(heat.x),list(range(7))],'y':[list(heat.y),list(range(7))],
             'zmin':[heat.zmin,None],'zmax':[heat.zmax,None]},
            {'title':{'text':chart.layout.title.text}}]))
    fig.update_layout(height=820,title=observed[0].layout.title.text,updatemenus=[dict(buttons=buttons,x=0,y=1.16)],margin=dict(l=150,t=130))
    fig.update_yaxes(autorange='reversed');fig.update_xaxes(title_text='Feature coordinate',row=1,col=1)
    fig.update_xaxes(title_text='Patch column',row=1,col=2);fig.update_yaxes(title_text='Patch row',row=1,col=2)
    crop=preprocessed_image(trace);photo=image.copy();photo.thumbnail((440,320))
    credits='' if record is None else f'<p>Image: {escape(record["id"])} · {escape(record.get("author",""))} · <a href="{escape(record["source"],quote=True)}">{escape(record["license"])}</a></p>'
    match='' if cosine is None else f'<p>Text prompt: <strong>{escape(prompt)}</strong> · full 512D image–text cosine: {cosine:.4f}</p>'
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLIP image activations</title><style>body{background:#f5f7fa;color:#172b42;font:16px/1.6 Arial;margin:32px}img{max-width:45%;max-height:320px;object-fit:contain;margin:12px}a{color:#137c75}</style><h1>Actual CLIP image activations</h1><p>Original photograph → actual 224 × 224 center crop → 49 patches × 768 features → 50 vision tokens → 12 blocks → 512D class projection.</p><img alt="Original photograph" src="'+image_uri(photo)+'"><img alt="Actual model center crop" src="'+image_uri(crop)+'">'+credits+match+fig.to_html(full_html=False,include_plotlyjs=True)+'<p>The stage menu updates both plots. Spatial RMS is empty for pooled class readouts. Heatmaps show the first 64 feature coordinates; the NPZ file contains complete tensors. Patch locations are row,column in the actual model crop. Activations are not attention or saliency; a prompt does not change the independent image encoder. Use the live Image Activation Inspector for uploads, prompt comparison and feature trajectories.</p></html>'


def render():
    intro('09','Inspect image activations','What does CLIP compute for each patch of an image?',
          'Follow the actual frozen ViT-B/32 forward pass: 32 × 32 pixel patches, 768-wide vision states, 12 Transformer blocks, and a 512D class-token projection. Select a real photograph or upload your own, then compare the final vector with a text prompt.')
    names=collection_names()
    collection=st.segmented_control('Inspector image collection',names,default=names[-1])
    gallery=collection_photos(collection or names[-1])
    categories=list(dict.fromkeys(r['category'] for r in gallery))
    category=st.selectbox('Inspector image category',categories)
    image,record=select_image('image_activation',[r for r in gallery if r['category']==category])
    prompt=st.text_input('Text prompt to compare','a photo of a cat',key='image_activation_prompt')
    with st.spinner('Observing the frozen image encoder…'):
        trace=inspect_image_payload(image_bytes(image))
    crop=preprocessed_image(trace);grid_image=crop.copy();draw=ImageDraw.Draw(grid_image)
    h,w=trace['grid'];ph,pw=trace['patch_size']
    for r in range(h):
        for c in range(w):
            draw.rectangle((c*pw,r*ph,(c+1)*pw-1,(r+1)*ph-1),outline='#00ffff',width=1)
            draw.text((c*pw+2,r*ph+2),f'{r},{c}',fill='white',stroke_width=1,stroke_fill='black')
    left,right=st.columns(2)
    left.image(image,caption='Original image',width=360)
    right.image(grid_image.resize((448,448)),caption='Actual CLIP center crop · 224 × 224 · row,column patch indices',width=448)
    if record: photo_credits([record])
    st.latex(r'X\in\mathbb{R}^{1\times3\times224\times224}\to E\in\mathbb{R}^{1\times49\times768}\to H\in\mathbb{R}^{50\times1\times768}\to v\in\mathbb{R}^{1\times512}')
    st.latex(r'v=\mathrm{LN}(H^{(12)}_{\mathrm{CLS}})W_I,\quad W_I\in\mathbb{R}^{768\times512},\quad \hat v=v/\|v\|_2')
    metrics=st.columns(3);metrics[0].metric('Spatial patches',h*w);metrics[1].metric('Features per vision token',trace['hidden_width']);metrics[2].metric('Joint embedding dimensions',trace['output_width'])
    controls=st.columns([2,1,1])
    stage=controls[0].selectbox('Image activation stage',list(trace['stages']))
    count=controls[1].selectbox('Image features shown',[32,64,128,512],index=1)
    maximum=trace['stages'][stage].shape[1]-count
    start=controls[2].slider('First image feature',0,maximum,0) if maximum>0 else 0
    st.plotly_chart(image_heatmap(trace,stage,start,count),width='stretch')
    st.caption('A symmetric color range is fitted to this displayed slice; numeric values remain available on hover. Patch projection has 49 rows; later residual states have 50 rows including the class token. Pooled outputs have one row.')
    spatial=trace['stages'][stage].shape[0]>1
    if spatial:
        controls=st.columns(2)
        mode=controls[0].radio('Patch map statistic',['RMS magnitude','One feature'])
        feature=int(controls[1].number_input('Spatial feature coordinate',0,trace['hidden_width']-1,0))
        values=patch_values(trace,stage,feature if mode=='One feature' else None)
        fig=go.Figure(go.Heatmap(z=values.tolist(),x=list(range(w)),y=list(range(h)),
            colorscale='RdBu' if mode=='One feature' else 'Teal',
            **({'zmin':-max(float(np.abs(values).max()),1e-8),'zmax':max(float(np.abs(values).max()),1e-8)} if mode=='One feature' else {}),
            hovertemplate='Patch (%{y},%{x})<br>Value %{z:.6f}<extra></extra>'))
        fig.update_layout(title=f'7 × 7 patch map · {stage} · {mode}',height=500,xaxis_title='Patch column',yaxis_title='Patch row',yaxis_autorange='reversed',yaxis_scaleanchor='x')
        st.plotly_chart(fig,width='stretch')
    else: st.info('The final readout uses only the class token. This pooled vector has no 7 × 7 spatial layout.')
    stages=[name for name,values in trace['stages'].items() if values.shape==(h*w+1,trace['hidden_width'])]
    token=st.selectbox('Follow a vision token',range(h*w+1),format_func=lambda i:'Class token' if i==0 else f'Patch ({(i-1)//w},{(i-1)%w})')
    feature=int(st.number_input('Vision feature to follow',0,trace['hidden_width']-1,0))
    fig=go.Figure(go.Scatter(x=stages,y=[float(trace['stages'][name][token,feature]) for name in stages],mode='lines+markers'))
    fig.update_layout(title=f'Feature {feature} across vision depth',height=350,yaxis_title='Activation',xaxis_title='Forward stage')
    st.plotly_chart(fig,width='stretch')
    rms=np.stack([np.sqrt(np.mean(trace['stages'][name]**2,axis=1)) for name in stages],axis=1)
    depth=go.Figure(go.Heatmap(z=rms.tolist(),x=stages,y=stage_labels(trace,stages[0]),colorscale='Teal'))
    depth.update_layout(title='Vision-token RMS across depth',height=650,yaxis_autorange='reversed')
    st.plotly_chart(depth,width='stretch')
    cosine=None
    if prompt.strip():
        text=encode_text([prompt]).numpy()[0];vector=trace['stages']['Normalized CLIP vector'][0]
        cosine=float(vector@text)
        st.metric('Image–prompt cosine · original 512D space',f'{cosine:.4f}')
        products=vector*text
        matching=go.Figure(go.Bar(x=list(range(512)),y=products.tolist(),marker_color='#137c75'))
        matching.update_layout(title='Coordinate contributions to the cosine sum',height=350,xaxis_title='Joint feature coordinate',yaxis_title='Normalized image × text feature')
        st.plotly_chart(matching,width='stretch')
        st.latex(r'\cos(v,t)=\hat v^\top\hat t=\sum_{d=1}^{512}\hat v_d\hat t_d')
    st.info('Patch maps show activations, not attention, saliency, or prompt-conditioned importance. A text prompt does not change the image encoder states: the two encoders run independently and meet only in the joint embedding space. Large activations do not automatically identify semantic concepts.')
    buffer=BytesIO();np.savez_compressed(buffer,preprocessed=trace['preprocessed'],**trace['stages'])
    st.download_button('Download full image activation tensors · NPZ',buffer.getvalue(),file_name='image_activations.npz')
    metadata={k:trace[k] for k in ['grid','patch_size','input_shape','hidden_width','output_width']}
    metadata.update(image_id=record['id'] if record else 'uploaded',prompt=prompt,stages={name:list(v.shape) for name,v in trace['stages'].items()})
    st.download_button('Download image activation metadata · JSON',json.dumps(metadata,indent=2).encode(),file_name='image_activations_metadata.json')
    st.download_button('Download offline image activation view · HTML',image_document(trace,image,record,prompt,cosine).encode(),file_name='image_activations.html',mime='text/html')
    takeaway('Inspect real spatial patch features, follow their contextual transformation, and distinguish them from the pooled image vector used for text matching.')
