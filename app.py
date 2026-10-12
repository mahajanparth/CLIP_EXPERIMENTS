"""Launch with: streamlit run app.py"""
import streamlit as st
from clip_explorer.config import CHECKPOINT
from clip_explorer.model import ModelUnavailable
from clip_explorer import labs, embedding_lab, process_lab, activation_lab

st.set_page_config(page_title='CLIP Explorer',page_icon='↔',layout='wide',initial_sidebar_state='expanded')
with st.sidebar:
    st.markdown('## CLIP Explorer')
    st.caption('FROM NATURAL LANGUAGE TO VISUAL REPRESENTATIONS')
    presenter = st.toggle('Presenter Mode',key='presenter')
    st.divider()
    st.caption('Original OpenAI CLIP · ViT-B/32')
    st.caption('Local weights ready' if CHECKPOINT.exists() else 'Synthetic lab ready · weights needed for inference')
    st.markdown('[Paper ↗](https://arxiv.org/abs/2103.00020) · [Implementation ↗](https://github.com/openai/CLIP)')

font_size = 22 if presenter else 17
st.markdown(f'''<style>
.block-container {{max-width: 1440px; padding-top: 3.5rem;}}
h1 {{letter-spacing:-.035em;}}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{font-size:{font_size}px;}}
[data-testid="stMetricValue"] {{font-size:2rem;}}
[data-testid="stSidebar"] {{border-right:1px solid #dbe3e9;}}
</style>''',unsafe_allow_html=True)

pages = [
    st.Page(process_lab.render,title='CLIP Process Walkthrough',url_path='clip_process',icon=':material/account_tree:'),
    st.Page(labs.contrastive,title='Contrastive Learning',icon=':material/swap_horiz:',default=True),
    st.Page(labs.zero_shot,title='Zero-Shot Classification',icon=':material/category:'),
    st.Page(labs.prompts,title='Prompt Engineering',icon=':material/edit:'),
    st.Page(labs.retrieval,title='Image Retrieval',icon=':material/search:'),
    st.Page(labs.failures,title='Failure Laboratory',icon=':material/science:'),
    st.Page(activation_lab.render,title='Text Activation Inspector',url_path='text_activations',icon=':material/neurology:'),
    st.Page(embedding_lab.render,title='Embedding Space · 2D / 3D',url_path='embedding_space',icon=':material/view_in_ar:'),
]
try:
    st.navigation(pages).run()
except ModelUnavailable as exc:
    st.warning(str(exc))
    st.code('python scripts/prepare.py --download-model',language='bash')
    st.info('The synthetic Contrastive Learning lab is available now. No model scores are invented or substituted when the checkpoint is missing.')
except (ValueError, RuntimeError) as exc:
    st.error(f'Experiment could not run: {exc}')
    st.caption('Check your candidate descriptions and templates. CLIP accepts at most 77 tokens per description. If CUDA memory is exhausted, restart with CUDA_VISIBLE_DEVICES="" streamlit run app.py.')
