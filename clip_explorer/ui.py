"""Shared presentation components and input validation."""
from __future__ import annotations
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
from .data import manifest, open_asset, asset_embeddings, photographic_gallery
from .model import encode_image


def intro(number: str, title: str, question: str, explanation: str) -> None:
    st.caption(f'CLIP EXPLORER  /  EXPERIMENT {number}  /  RADFORD ET AL., 2021')
    st.title(title)
    st.markdown(f'**{question}**')
    st.write(explanation)
    st.divider()


def takeaway(text: str) -> None:
    st.success(f'Key takeaway · {text}')


def lines(text: str, max_count: int = 40) -> list[str]:
    entries = list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
    if not entries or len(entries) > max_count:
        raise ValueError(f'Enter between 1 and {max_count} unique, nonempty lines.')
    return entries


def select_image(key: str) -> tuple[Image.Image, dict | None]:
    gallery = photographic_gallery()
    # Prefer the public-domain cat photo, when downloaded, for the default demo.
    preferred = 'dog' if key == 'prompt' else 'cat'
    gallery = sorted(gallery,key=lambda r:r['label']!=preferred)
    record = st.selectbox('Prepared image', gallery, format_func=lambda r:f'{r["label"].title()} · {r["medium"]} · {r["id"]}',key=f'{key}_asset')
    upload = st.file_uploader('Or upload your own image',type=['png','jpg','jpeg','webp'],key=f'{key}_upload')
    if upload:
        try:
            with Image.open(BytesIO(upload.getvalue())) as image:
                if image.width*image.height > 30_000_000:
                    raise ValueError('Please use an image smaller than 30 megapixels.')
                return ImageOps.exif_transpose(image).convert('RGB'),None
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f'Cannot read this image: {exc}')
    return open_asset(record),record


def image_embedding(image: Image.Image, record: dict | None):
    return asset_embeddings([record]) if record else encode_image(image)


def bars(frame: pd.DataFrame, label: str, value: str, title: str, probability: bool = False) -> None:
    data = frame.sort_values(value,ascending=True)
    figure = go.Figure(go.Bar(x=data[value],y=data[label],orientation='h',marker_color='#137C75',text=[f'{v:.1%}' if probability else f'{v:.3f}' for v in data[value]],textposition='auto',hovertemplate='%{y}<br>%{x:.4f}<extra></extra>'))
    figure.update_layout(title=title,height=max(330,len(data)*42+110),xaxis_title='Candidate softmax probability' if probability else 'Cosine similarity',yaxis_title=None,margin=dict(l=10,r=25,t=60,b=40),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(size=18 if st.session_state.get('presenter') else 14))
    if probability: figure.update_xaxes(range=[0,1],tickformat='.0%')
    st.plotly_chart(figure,width='stretch')


def downloadable(frame: pd.DataFrame, filename: str) -> None:
    st.download_button('Download measured results · CSV',frame.to_csv(index=False).encode(),file_name=filename,mime='text/csv')


def photo_gallery(records: list[dict], columns: int = 4) -> None:
    """Display photographs with matching captions, and accessible source credits."""
    grid=st.columns(columns)
    for idx,record in enumerate(records):
        with grid[idx%columns]:
            st.image(open_asset(record),caption=record['label'].title(),width='stretch')
            st.caption(record['caption'])
    photo_credits(records)


def photo_credits(records: list[dict]) -> None:
    photos=[r for r in records if r.get('medium')=='photograph']
    if not photos:return
    with st.expander('Photo credits and source licenses'):
        st.dataframe(pd.DataFrame([{'Image':r['id'],'Photographer':r.get('author',''),'License':r['license'],'Source':r['source']} for r in photos]),hide_index=True,width='stretch',column_config={'Source':st.column_config.LinkColumn('Original file')})
