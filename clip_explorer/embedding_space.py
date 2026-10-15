"""Joint projections and figures for inspecting real multimodal embeddings."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .config import SEED

COLORS = {'Animals':'#137C75','Vehicles':'#5279C1','Landscapes':'#649447',
          'Food':'#D08036','Robotics':'#9666B5','Household':'#BB617D','People':'#617888'}


def category_color(category: str) -> str:
    if category in COLORS:return COLORS[category]
    import colorsys
    import hashlib
    hue=int(hashlib.sha256(category.encode()).hexdigest()[:8],16)/0xffffffff
    return '#'+''.join(f'{round(c*255):02x}' for c in colorsys.hsv_to_rgb(hue,.65,.72))


def project_joint_embeddings(embeddings: np.ndarray, method: str = 'PCA', perplexity: float = 8.0, dimensions: int = 3) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit one projection to all modalities together; preserve input row order.

    PCA is centered, unwhitened, and uses deterministic full SVD. t-SNE is a
    seeded exploratory view using Euclidean distances of unit CLIP vectors;
    it does not preserve global distances or define semantic axes.
    """
    vectors = np.asarray(embeddings,dtype=np.float64)
    if dimensions not in (2,3):
        raise ValueError('Choose 2 or 3 projection dimensions.')
    if vectors.ndim!=2 or min(vectors.shape)<dimensions or not np.isfinite(vectors).all():
        raise ValueError(f'A {dimensions}D projection requires at least {dimensions} finite vectors and features.')
    if method=='PCA':
        projection = PCA(n_components=dimensions,svd_solver='full')
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=8):
            coordinates = projection.fit_transform(vectors)
        return coordinates, projection.explained_variance_ratio_
    if method=='t-SNE':
        if not 0<perplexity<len(vectors):
            raise ValueError('t-SNE perplexity must be positive and smaller than the point count.')
        from threadpoolctl import threadpool_limits
        estimator = TSNE(n_components=dimensions,perplexity=perplexity,random_state=SEED,
                           init='pca',learning_rate='auto',max_iter=1000,method='barnes_hut' if len(vectors)>500 else 'exact',
                           metric='euclidean',n_jobs=1)
        with threadpool_limits(limits=8):
            coordinates=estimator.fit_transform(vectors)
        return coordinates, None
    raise ValueError('Select PCA or t-SNE.')


def embedding_figure(points: pd.DataFrame, method: str, pair_count: int,
                     connect_pairs: bool = True, show_labels: bool = False,
                     selected_pair: int | None = None) -> go.Figure:
    """Plot modality by symbol and category by color, with optional pair links."""
    dimensions=3 if 'Z' in points.columns else 2
    trace_class=go.Scatter3d if dimensions==3 else go.Scatter
    figure = go.Figure()
    if connect_pairs:
        xs,ys,zs = [],[],[]
        for idx in range(pair_count):
            image,text = points.iloc[idx],points.iloc[pair_count+idx]
            xs.extend([image.X,text.X,None]);ys.extend([image.Y,text.Y,None])
            if dimensions==3:zs.extend([image.Z,text.Z,None])
        figure.add_trace(trace_class(x=xs,y=ys,**({'z':zs} if dimensions==3 else {}),mode='lines',line=dict(color='#BEC9D3',width=2),
                                     name='Prepared image–caption pairs',hoverinfo='skip'))
    for (category,modality),rows in points.groupby(['Category','Modality'],sort=False):
        query = modality=='Query'
        normal_size,selected_size=(7,13) if dimensions==3 else (12,20)
        sizes = [selected_size if selected_pair is not None and i in [selected_pair,pair_count+selected_pair] else (normal_size+3 if query else normal_size) for i in rows.index]
        figure.add_trace(trace_class(x=rows.X,y=rows.Y,**({'z':rows.Z} if dimensions==3 else {}),
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=sizes,color='#172B42' if query else category_color(category),
                        symbol='diamond' if modality!='Image' else 'circle',opacity=.9,
                        line=dict(color='#FFFFFF',width=1)),
            text=rows.Label,textposition='top center',
            customdata=rows[['ID','Description','Modality']].to_numpy(),
            hovertemplate='<b>%{customdata[2]} · %{text}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>',
            name=f'{category} · {modality}'))
    axes = ['PC 1','PC 2','PC 3'] if method=='PCA' else ['t-SNE 1','t-SNE 2','t-SNE 3']
    figure.update_layout(title=f'Joint image + text embeddings · {dimensions}D {method}',height=720,
        legend=dict(orientation='h',y=-.08),margin=dict(l=25,r=25,t=55,b=80),
        paper_bgcolor='rgba(0,0,0,0)',font=dict(size=14))
    if points.Category.nunique()>10:
        figure.update_layout(legend=dict(orientation='v',x=1.02,y=1),margin=dict(r=190))
    if dimensions==3:
        figure.update_layout(scene=dict(xaxis_title=axes[0],yaxis_title=axes[1],zaxis_title=axes[2],aspectmode='data'))
    else:
        figure.update_layout(xaxis=dict(title=axes[0],zeroline=False),
                            yaxis=dict(title=axes[1],zeroline=False,scaleanchor='x',scaleratio=1),
                            dragmode='pan',plot_bgcolor='#FFFFFF')
    return figure


def point_table(records: list[dict], descriptions: list[str], coordinates: np.ndarray,
                query: str = '') -> pd.DataFrame:
    """Describe image rows followed by text rows and optional query row."""
    rows = []
    for modality in ['Image','Text']:
        for record,description in zip(records,descriptions):
            rows.append({'ID':record['id'],'Label':record['label'],'Category':record['category'],
                         'Modality':modality,'Description':description})
    if query: rows.append({'ID':'custom_query','Label':'Custom query','Category':'Query','Modality':'Query','Description':query})
    if len(rows)!=len(coordinates):
        raise ValueError('Point metadata and projection rows must agree.')
    frame = pd.DataFrame(rows)
    if coordinates.ndim!=2 or coordinates.shape[1] not in (2,3):
        raise ValueError('Expected two or three projection coordinates per point.')
    frame[['X','Y','Z'][:coordinates.shape[1]]] = coordinates
    return frame


def default_records(gallery: list[dict]) -> list[dict]:
    """Twelve real photographs balanced across three semantic categories."""
    from .data import balanced_photos
    return balanced_photos(gallery,['Animals','Vehicles','Food'],4)


def embedding_document(figure: go.Figure, records: list[dict], points: pd.DataFrame | None = None,
                       method: str = "PCA", connect_pairs: bool = True) -> str:
    """Offline plot with actual photographic thumbnails and hover-linked preview."""
    from html import escape
    from .data import open_asset
    from .thumbnail_space import photo_uri as image_uri
    cards=[]
    for idx,record in enumerate(records):
        image=open_asset(record);image.thumbnail((300,220))
        cards.append(f'<article {"hidden" if idx>=60 else ""}><img id="photo-{escape(record["id"],quote=True)}" src="{image_uri(image)}" alt="{escape(record["label"],quote=True)}" data-caption="{escape(record["caption"],quote=True)}"><strong>{escape(record["label"].title())}</strong><small>{escape(record["id"])}</small><small>{escape(record.get("author",""))} · <a href="{escape(record["source"],quote=True)}">{escape(record["license"])}</a></small></article>')
    if len(records)>60:cards.insert(0,f'<p>Contact sheet previews 60 of {len(records)} images; all photographs are embedded and represented in the plots. Hover any point to inspect its photograph.</p>')
    plot=figure.to_html(full_html=False,include_plotlyjs=True)
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLIP · Real images and text in 3D</title><style>body{background:#f5f7fa;color:#172b42;font:16px/1.5 Arial,sans-serif;margin:0}main{max-width:1300px;margin:auto;padding:32px}h1{letter-spacing:-1px}.photos{display:grid;grid-template-columns:repeat(6,1fr);gap:15px}article{padding:10px;background:white;border-radius:12px}article img{width:100%;height:130px;object-fit:contain}strong,small{display:block}small{color:#617888;font-size:11px;overflow-wrap:anywhere}a{color:#137c75}.preview{display:flex;gap:20px;align-items:center;background:white;border-radius:12px;padding:16px;margin:20px 0}.preview img{width:220px;height:160px;object-fit:contain}.note{background:#e6f1ef;border-left:4px solid #137c75;padding:16px}@media(max-width:800px){.photos{grid-template-columns:repeat(3,1fr)}}</style></head><body><main><h1>Real photographs + text in the CLIP space</h1><p>Circles are image vectors; diamonds are text vectors. Rotate, zoom, hover, and toggle the legend. Hover an image or paired caption to display the corresponding photograph below.</p><div class="photos">'''+''.join(cards)+'''</div>'''+plot+'''<div class="preview"><img id="hover-photo" alt="Hovered photograph" hidden><div><strong id="hover-caption">Hover a point to inspect its photograph.</strong><small>Matching image/text points reference the same gallery asset.</small></div></div><p class="note">One shared projection is fitted to measured normalized CLIP vectors. 3D positions are a lossy view of the original 512D space; plotted distances should not be used as cosine scores. This view contains real local photographs, with source credits beside each thumbnail. No external script or image download is needed.</p></main><script>const plot=document.querySelector('.js-plotly-plot');plot.on('plotly_hover',event=>{const data=event.points[0].customdata;if(!data)return;const image=document.getElementById('photo-'+data[0]);if(!image)return;const preview=document.getElementById('hover-photo');preview.src=image.src;preview.hidden=false;document.getElementById('hover-caption').textContent=image.dataset.caption;});</script></body></html>'''

    dimensions=3 if any(trace.type=='scatter3d' for trace in figure.data) else 2
    if dimensions==2:
        document=document.replace('in 3D','in 2D').replace('3D positions','2D positions').replace('Rotate, zoom','Pan, zoom')
    if points is not None:
        from .thumbnail_space import thumbnail_document
        thumbnail_html = thumbnail_document(points, records, method, connect_pairs)
        embedded = '<h2>A separate view with photographs at their coordinates</h2><iframe title="Photographic embedding plot" style="width:100%;height:850px;border:0" srcdoc="' + escape(thumbnail_html, quote=True) + '"></iframe>'
        document = document.replace('</main>', embedded + '</main>')
    return document
