"""Interactive 2D/3D views; original-space scores remain the source of rankings."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from .thumbnail_space import thumbnail_document
from .collections import collection_names,collection_photos
import torch
from .data import manifest, asset_embeddings, open_asset, photographic_gallery, balanced_photos
from .model import encode_text
from .math_ops import cosine_similarity, format_prompts
from .embedding_space import project_joint_embeddings, embedding_figure, point_table, embedding_document
from .ui import intro, takeaway, downloadable, photo_gallery


@st.cache_data(show_spinner=False,max_entries=32)
def _project(vectors: np.ndarray, method: str, perplexity: float, dimensions: int):
    return project_joint_embeddings(vectors,method,perplexity,dimensions)


def render() -> None:
    intro('06','Explore the embedding space','Where do images and their descriptions sit in the same representation?',
          'CLIP produces 512-dimensional unit vectors for images and text. Fit one shared projection to both modalities and explore a 2D or 3D view. Circles represent images; diamonds represent descriptions; colors identify gallery categories.')
    collections=collection_names()
    collection=st.segmented_control('Image collection',collections,default=collections[-1])
    gallery=collection_photos(collection or collections[-1])
    controls = st.columns([2,1,1,1])
    categories = controls[0].multiselect('Gallery categories',list(dict.fromkeys(r['category'] for r in gallery)),default=list(dict.fromkeys(r['category'] for r in gallery)))
    per_category = controls[1].slider('Images per category',1,50,50)
    method = controls[2].selectbox('Projection',['PCA','t-SNE'])
    dimensions=controls[3].select_slider('Plot dimensions',options=[2,3],value=3,format_func=lambda n:f'{n}D')
    records = balanced_photos(gallery,categories,per_category)
    availability = pd.DataFrame([
        {'Category':category,
         'Available photographs':sum(r['category']==category for r in gallery),
         'Shown':sum(r['category']==category for r in records)}
        for category in categories
    ])
    st.caption(f'{len(categories)} categories selected · {len(records)} real photographs shown · up to {per_category} per category.')
    if any(row['Available photographs']<per_category for row in availability.to_dict('records')):
        st.info('Some categories contain fewer photographs than the requested limit. All available photographs are shown once; additional images must be added to reach the limit.')
    with st.expander('Photographs available per category'):
        st.dataframe(availability,hide_index=True,width='stretch')
    if len(records)<2:
        st.info('Select at least two images in total to compare image and text points.')
        return
    a,b = st.columns([1,2])
    source = a.radio('Caption source',['Prepared captions','Shared prompt template'])
    if source=='Shared prompt template':
        template = b.text_input('Text prompt template','a photo of a {}',key='space_template')
        descriptions = format_prompts([r['label'] for r in records],template)
    else:
        descriptions = [r['caption'] for r in records]
        b.caption('Every point uses a real photograph and its caption. Changing the prompt changes the text vector.')
    query = st.text_input('Optional extra text point','',placeholder='e.g. a pet animal or a vehicle with two wheels').strip()
    if method=='t-SNE':
        max_perplexity = min(30,2*len(records)-1+bool(query))
        perplexity = float(st.slider('t-SNE perplexity',1,max_perplexity,min(8,max_perplexity),help='A neighborhood-size control. Changing it can substantially change the view.'))
    else:
        perplexity = 8.0
    options = st.columns(2)
    connect = options[0].checkbox('Connect prepared image–caption pairs',len(records)<=60)
    show_labels = options[1].checkbox('Show point labels',False)
    selected = st.selectbox('Inspect an image and its caption',range(len(records)),format_func=lambda i:f'{records[i]["label"].title()} · {records[i]["id"]}')
    with st.spinner('Projecting the measured image and text embeddings together…'):
        images = asset_embeddings(records)
        texts = encode_text(descriptions)
        vectors = torch.cat([images,texts]+([encode_text([query])] if query else []))
        coordinates,variance = _project(vectors.numpy(),method,perplexity,dimensions)
    frame = point_table(records,descriptions,coordinates,query)
    figure = embedding_figure(frame,method,len(records),connect,show_labels,selected)
    st.subheader('The photographs represented in this view')
    preview_count=min(60,len(records))
    if len(records)>preview_count:st.caption(f'Contact sheet shows the first {preview_count} of {len(records)} photographs. Every selected photo appears in the embedding plots below.')
    photo_gallery(records[:preview_count],columns=6)
    st.plotly_chart(figure,width='stretch',config={'scrollZoom':True})
    st.caption(('Drag to rotate' if dimensions==3 else 'Drag to pan')+' · scroll to zoom · hover to inspect · click legend entries to toggle groups. Larger markers identify the selected image–caption pair. Connecting lines indicate known pairings, not a claim that every pair is the nearest match.')
    st.subheader('Photographs at their embedding coordinates')
    st.caption('A separate view of the same projection: photo centers locate image vectors; diamonds locate text vectors. Adjust thumbnail size to resolve overlaps. In 3D, photos face the camera while their positions and depth follow the rotated coordinates.')
    thumbnail_html = thumbnail_document(frame, records, method, connect)
    components.html(thumbnail_html, height=850, scrolling=True)
    st.download_button('Download thumbnail plot · HTML', thumbnail_html.encode(),
                       file_name=f'embedding_thumbnails_{dimensions}d.html', mime='text/html')
    if variance is not None:
        st.metric(f'Variance retained by the {dimensions} PCA axes',f'{variance.sum():.1%}')
        st.caption('Variance is computed across this joint image/text sample. PCA axes are data-derived directions, not named semantic attributes.')
    else:
        st.caption('Seeded t-SNE emphasizes local neighborhoods. Its axis directions and distances between distant clusters have no direct semantic interpretation.')
    st.info(f'This is a lossy {dimensions}D projection of 512D vectors. Visual distance can distort the original geometry. All similarities and rankings below use normalized vectors in the original CLIP space. Adding points or changing prompts refits the projection and can move existing points.')
    left,right = st.columns([1,2])
    with left:
        st.image(open_asset(records[selected]),width='stretch')
        st.write(f'**Paired text:** {descriptions[selected]}')
    with right:
        scores = cosine_similarity(images[selected:selected+1],texts)[0]
        paired = float(scores[selected])
        nearest = int(scores.argmax())
        metrics = st.columns(2)
        metrics[0].metric('Paired caption cosine · 512D',f'{paired:.4f}')
        metrics[1].metric('Highest-ranked caption',records[nearest]['label'])
        ranking = pd.DataFrame({'Caption':descriptions,'Image ID': [r['id'] for r in records],'Cosine similarity':scores.numpy()}).sort_values('Cosine similarity',ascending=False,kind='stable')
        st.dataframe(ranking.head(8),hide_index=True,width='stretch')
        if query:
            query_scores = cosine_similarity(images,vectors[-1:]).flatten()
            best = int(query_scores.argmax())
            st.write(f'**Custom query’s nearest image in 512D:** {records[best]["label"]} · {records[best]["id"]} · cosine {float(query_scores[best]):.4f}')
    exports = st.columns(2)
    with exports[0]: downloadable(frame,f'embedding_coordinates_{dimensions}d.csv')
    with exports[1]:
        st.download_button(f'Download interactive {dimensions}D view · HTML',lambda: embedding_document(figure,records,frame,method,connect).encode(),file_name=f'embedding_space_{dimensions}d.html',mime='text/html')
    with st.expander('What exactly gets projected?'):
        st.latex(rf'X=[\hat V;\hat T]\in\mathbb{{R}}^{{{len(vectors)}\times512}},\quad Z\in\mathbb{{R}}^{{{len(vectors)}\times{dimensions}}}')
        if method=='PCA':
            st.latex(rf'Z=(X-\bar X)W,\qquad W\in\mathbb{{R}}^{{512\times{dimensions}}}')
        else:
            st.latex(r'Z=\operatorname{t-SNE}(X)')
        st.write('One projection is fitted to the concatenated image and text vectors, optionally including the extra text query. Projecting the modalities separately would produce unrelated coordinate systems. PCA keeps the selected number of directions with the most joint variance, without whitening. t-SNE uses Euclidean distances between unit vectors (squared distance = 2 − 2 × cosine). Neither view replaces cosine matching in the full embedding space.')
    takeaway('Images and words inhabit the same learned space. Explore alignment in 2D or 3D, then verify the impression with original-space cosine similarities.')
