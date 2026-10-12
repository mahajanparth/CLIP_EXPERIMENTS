"""Five classroom demonstrations, each with a local fallback asset set."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from .config import TEMPLATES
from .data import manifest, open_asset, asset_embeddings, photographic_gallery, photo_for_label
from .math_ops import (cosine_similarity, contrastive_loss, directional_probabilities, temperature_logits, format_prompts, ensemble_embeddings, retrieval_ranking, similarity_margins)
from .model import encode_text, compute_logits, compute_probabilities, load_clip_model
from .ui import intro, takeaway, select_image, image_embedding, lines, bars, downloadable, photo_gallery, photo_credits


def contrastive() -> None:
    intro('01','Learn to match','How does a batch of image–caption pairs become a learning signal?',
          'Two encoders map images and text into the same embedding space. Each matched pair is a positive; the other pairings in this batch are negatives. This lab visualizes the objective without training a model.')
    a,b,c = st.columns([2,1,1])
    mode = a.radio('Similarity source',['Synthetic teaching example','Pretrained CLIP'],horizontal=True)
    n = b.selectbox('Batch size',[4,6])
    tau = c.slider('Temperature τ',0.01,1.0,0.07,0.01)
    st.caption('The temperature control is a teaching override: logits = cosine / τ. Classification pages always use the checkpoint’s learned scale.')
    labels = ['dog','cat','car','bicycle','coffee','robot'][:n]
    records = [photo_for_label(label) for label in labels]
    captions = [r['caption'] for r in records]
    images = st.columns(n)
    for idx,(column,record) in enumerate(zip(images,records)):
        column.image(open_asset(record),caption=f'Image {idx+1} · {record["label"]}',width='stretch')
    photo_credits(records)
    if mode == 'Synthetic teaching example':
        st.info('SYNTHETIC MATRIX · Constructed unit-vector similarities, not measured CLIP outputs. The slider changes alignment; no optimizer is run.')
        strength = st.slider('Positive-pair alignment',0.0,1.0,0.65,0.01)
        # Unit-vector construction makes these legitimate cosine similarities.
        images_emb = torch.eye(n,2*n)
        text_emb = strength*torch.eye(n,2*n)
        text_emb[:,n:] = (1-strength**2)**0.5*torch.eye(n)
        similarities = cosine_similarity(images_emb,text_emb)
    else:
        with st.spinner('Encoding the prepared pairs…'):
            similarities = cosine_similarity(asset_embeddings(records),encode_text(captions))
        learned_tau = float(load_clip_model().model.logit_scale.detach().exp().reciprocal().cpu())
        st.caption(f'Measured pretrained cosine similarities. Checkpoint’s learned τ ≈ {learned_tau:.4f}; the slider above overrides τ only in this lab.')
    shuffled = st.checkbox('Rotate caption order to move positives off the diagonal')
    permutation = list(range(n))
    if shuffled: permutation = permutation[1:]+permutation[:1]
    similarities = similarities[:,permutation]
    captions = [captions[j] for j in permutation]
    targets = torch.tensor([permutation.index(i) for i in range(n)])
    logits = temperature_logits(similarities,tau)
    rows,columns = directional_probabilities(logits)
    view = st.radio('Heatmap',['Cosine similarity','Image → text probability','Text → image probability'],horizontal=True)
    values = {'Cosine similarity':similarities,'Image → text probability':rows,'Text → image probability':columns}[view]
    selection = st.columns(2)
    selected_row = selection[0].selectbox('Highlight image',list(range(n)),format_func=lambda i:f'Image {i+1} · {labels[i]}')
    selected_col = selection[1].selectbox('Highlight caption',list(range(n)),format_func=lambda i:f'{i+1}: {captions[i]}')
    fig = go.Figure(go.Heatmap(z=values.numpy(),x=list(range(n)),y=list(range(n)),colorscale='Teal',zmin=-1 if view=='Cosine similarity' else 0,zmax=1,text=values.numpy(),texttemplate='%{text:.3f}',hovertemplate='Image %{y}<br>Caption %{x}<br>Value %{z:.4f}<extra></extra>'))
    for i,j in enumerate(targets.tolist()):
        fig.add_shape(type='rect',x0=j-.47,x1=j+.47,y0=i-.47,y1=i+.47,line=dict(color='#DB8A2C',width=4))
    fig.add_shape(type='rect',x0=selected_col-.48,x1=selected_col+.48,y0=selected_row-.48,y1=selected_row+.48,line=dict(color='#24384B',width=6))
    fig.update_layout(height=500,xaxis=dict(tickmode='array',tickvals=list(range(n)),ticktext=[f'{i+1}. {captions[i]}' for i in range(n)],side='top'),yaxis=dict(tickmode='array',tickvals=list(range(n)),ticktext=[f'Image {i+1}: {label}' for i,label in enumerate(labels)],autorange='reversed'),margin=dict(t=100,b=30,l=80,r=30),font=dict(size=16),paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,width='stretch')
    st.caption('Gold outlines = true matched pairs. Dark outline = selected pair. Image thumbnails above are numbered to match the rows.')
    positive = selected_col==targets[selected_row].item()
    st.write(f'Selected pair is a **{"positive" if positive else "negative"}**. Cosine = {similarities[selected_row,selected_col]:.3f}; image → text P = {rows[selected_row,selected_col]:.3f}; text → image P = {columns[selected_row,selected_col]:.3f}.')
    li,lt,loss = contrastive_loss(logits,targets)
    m = st.columns(3)
    for col,label,value in zip(m,['Image → text loss','Text → image loss','Symmetric loss'],[li,lt,loss]): col.metric(label,f'{value.item():.4f}')
    with st.expander('The mathematics of matching'):
        st.latex(r'S_{ij}=\frac{v_i^\top u_j}{\|v_i\|\|u_j\|},\qquad L_{ij}=S_{ij}/\tau')
        st.latex(r'\mathcal{L}=\tfrac12[\mathrm{CE}(L,y)+\mathrm{CE}(L^\top,y^{-1})]')
        st.write('Image → text probabilities sum to one along each row. Text → image probabilities sum to one down each column. The diagonal contains positives only when both orders are aligned. Lower temperature sharpens rankings; it can also amplify incorrect matches. The original CLIP training batch is much larger than this teaching batch.')
    takeaway('Alignment determines the positives. The symmetric objective asks both modalities to recover their matching partner.')


def zero_shot() -> None:
    intro('02','Build a classifier with words','Can a text encoder supply the weights of an image classifier?',
          'Describe the categories, encode them, and compare each description with the image. The pretrained encoders stay frozen; no task-specific gradient updates are performed.')
    left,right = st.columns([1,1.7])
    with left:
        image,record = select_image('zero')
        st.image(image,caption='Original input image',width='stretch')
        if record: photo_credits([record])
    with right:
        preset = st.selectbox('Candidate experiment',['Broad categories','Dog breeds'])
        defaults = 'dog\ncat\ncar\nbicycle' if preset=='Broad categories' else 'golden retriever\nLabrador retriever\nGerman shepherd\nSiberian husky'
        text = st.text_area('Candidate classes · one per line',defaults,key=f'zero_classes_{preset}',height=145)
        template = st.text_input('Prompt template','a photo of a {}',key='zero_template')
        classes = lines(text)
        prompts = format_prompts(classes,template)
        with st.spinner('Comparing the image and candidate descriptions…'):
            v = image_embedding(image,record)
            u = encode_text(prompts)
            scores = cosine_similarity(v,u)[0]
            probs = compute_probabilities(compute_logits(v,u))[0]
        frame = pd.DataFrame({'Class':classes,'Description':prompts,'Cosine similarity':scores.numpy(),'Probability':probs.numpy()})
        top = int(scores.argmax())
        st.metric('Highest-ranked candidate',classes[top])
        measure = st.radio('Show scores',['Cosine similarity','Probability'],horizontal=True,key='zero_measure')
        bars(frame,'Class',measure,'Zero-shot candidate ranking',probability=measure=='Probability')
    st.info('Softmax probabilities are relative to these candidates, not calibrated confidence. Adding, removing, or rewording candidates changes the distribution, even for the same image.')
    st.dataframe(frame,width='stretch',hide_index=True)
    downloadable(frame,'zero_shot_results.csv')
    with st.expander('From text embeddings to classifier weights'):
        st.latex(r'\hat c=\arg\max_c\cos(v,u_c),\quad P(c\mid v,\mathcal C)=\mathrm{softmax}_c(e^\alpha\cos(v,u_c))')
        st.write('The text embeddings act as classifier weight vectors. α is the actual learned logit_scale parameter in the checkpoint. The original CLIP preprocessing resizes and center-crops a copy of the image for inference; the image shown above is the original.')
    takeaway('Zero-shot classification is matching against a supplied vocabulary. Try removing the best category and watch where the probability mass moves.')


def prompts() -> None:
    intro('03','Words change the experiment','How much do phrasing and context change the same class?',
          'Compare descriptions of one image, then compare three classifier strategies using exactly the same images and labels. Ensembling averages normalized text embeddings before scoring.')
    left,right = st.columns([1,1.7])
    with left:
        image,record = select_image('prompt')
        st.image(image,width='stretch')
        if record: photo_credits([record])
    with right:
        raw = st.text_area('Descriptions · one per line','dog\na dog\na photo of a dog\na photograph of a dog\na sketch of a dog',height=150)
        descriptions = lines(raw)
        v = image_embedding(image,record)
        scores = cosine_similarity(v,encode_text(descriptions))[0]
        frame = pd.DataFrame({'Description':descriptions,'Cosine similarity':scores.numpy()})
        bars(frame,'Description','Cosine similarity','Prompt sensitivity · same image')
    st.caption('These are cosine similarities, not probabilities. A differently phrased description is a different vector in the shared space.')
    st.divider()
    st.subheader('Three ways to construct the same classifier')
    a,b = st.columns(2)
    classes = lines(a.text_area('Shared class labels','dog\ncat\ncar\nbicycle',key='ensemble_classes'))
    templates = lines(b.text_area('Ensemble templates · one {} each','\n'.join(TEMPLATES),height=150))
    for template in templates: format_prompts(classes,template)
    single = st.text_input('Single contextual template','a photo of a {}',key='single_context')
    sample_scope = st.radio('Evaluate on',['Selected image','Prepared labeled photographs'],horizontal=True)
    if sample_scope=='Prepared labeled photographs':
        records = [r for r in manifest()['gallery'] if r['label'] in classes and r['medium']=='photograph']
        if not records:
            st.warning('No prepared photograph labels match these classes. Select dog, cat, car, or bicycle, or use the selected image.')
            return
        st.caption('These are the actual images used for the strategy comparison.')
        photo_gallery(records,columns=4)
        vectors = asset_embeddings(records)
    else:
        records = [record] if record else [None]
        vectors = v
    raw_embeddings = encode_text(classes)
    contextual = encode_text(format_prompts(classes,single))
    prompt_embeddings = encode_text([template.format(label) for label in classes for template in templates]).reshape(len(classes),len(templates),-1)
    strategies = {'Raw class names':raw_embeddings,'Single contextual prompt':contextual,'Template ensemble':ensemble_embeddings(prompt_embeddings)}
    rows = []
    all_labeled = all(r is not None and r['label'] in classes for r in records)
    for name,u in strategies.items():
        matrix = cosine_similarity(vectors,u)
        for idx,similarities in enumerate(matrix):
            pred = classes[int(similarities.argmax())]
            r = records[idx]
            row = {'Strategy':name,'Image':r['id'] if r else 'Uploaded image','Prediction':pred,'Highest cosine':float(similarities.max())}
            if all_labeled:
                row.update({'Label':r['label'],'Correct':pred==r['label'],'Label cosine':float(similarities[classes.index(r['label'])])})
            rows.append(row)
    results = pd.DataFrame(rows)
    if all_labeled:
        metrics = st.columns(3)
        for col,name in zip(metrics,strategies):
            subset = results[results.Strategy==name]
            col.metric(name,f'{subset.Correct.mean():.0%}',help=f'Accuracy on these {len(subset)} labeled examples only.')
        st.caption('Descriptive accuracy on this small photographic set; this is not a benchmark estimate. An ensemble is not guaranteed to improve results.')
    else:
        st.caption('No usable ground-truth labels for this candidate set. Reporting similarities and rankings only.')
    # Same selected image/classes across all three strategies, regardless of evaluation scope.
    fig = go.Figure()
    for name,u in strategies.items():
        fig.add_trace(go.Bar(name=name,x=classes,y=cosine_similarity(v,u)[0].numpy()))
    fig.update_layout(barmode='group',height=410,yaxis_title='Cosine similarity',title='Strategy comparison · selected image',font=dict(size=16),paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,width='stretch')
    st.dataframe(results,hide_index=True,width='stretch')
    downloadable(results,'prompt_strategy_results.csv')
    with st.expander('Embedding-based prompt ensembling'):
        st.latex(r'\bar u_c=\mathrm{normalize}\left(\frac1M\sum_{m=1}^M\mathrm{normalize}(f_T(t_m(c)))\right)')
        st.write('Normalize every prompt vector, average the vectors for each class, then normalize the mean. We compare images to these class vectors. We do not average softmax probabilities.')
    takeaway('Context can move text embeddings. A normalized embedding ensemble combines several phrasings while preserving one vector per class.')


def retrieval() -> None:
    intro('04','Search the shared space','Can a sentence retrieve an image without category-specific training?',
          'Encode a natural-language query and rank local gallery images by cosine similarity. The gallery contains real photographs with documented sources and licenses.')
    gallery = photographic_gallery()
    a,b = st.columns([3,1])
    query = a.text_input('Search with a description','a bicycle')
    k = b.slider('Top K',1,min(20,len(gallery)),6)
    method = st.radio('Query modality',['Text → image','Image → image'],horizontal=True)
    vectors = asset_embeddings(gallery)
    if method=='Text → image':
        if not query.strip():
            st.info('Enter a description to search the gallery.')
            return
        u = encode_text([query])
    else:
        image,record = select_image('retrieval')
        st.image(image,width=220)
        u = image_embedding(image,record)
        st.caption('When querying with a gallery image, its identical self-match is included and should rank first (ties may occur).')
    indices,scores = retrieval_ranking(vectors,u,k)
    rows = []
    columns = st.columns(3)
    for rank,(idx,score) in enumerate(zip(indices.tolist(),scores.tolist()),1):
        record = gallery[idx]
        with columns[(rank-1)%3]:
            st.image(open_asset(record),caption=f'#{rank} · {record["label"]} · cosine {score:.3f}',width='stretch')
            st.caption(f'{record["category"]} / {record["medium"]}')
        rows.append({'Rank':rank,'ID':record['id'],'Label':record['label'],'Cosine similarity':score})
    photo_credits([gallery[idx] for idx in indices.tolist()])
    downloadable(pd.DataFrame(rows),'retrieval_results.csv')
    with st.expander('Ranking and provenance'):
        st.latex(r's_i=\hat v_i^\top\hat u,\qquad\mathrm{results}=\mathrm{argsort}(s,\mathrm{descending})[:K]')
        st.write(f'{len(gallery)} prepared gallery assets. Embeddings are content-cached and persisted during preparation; retrieval uses ordinary matrix multiplication. Asset origins, licenses, and hashes are documented in assets/manifest.json and assets/PROVENANCE.md.')
    takeaway('One shared embedding space supports classification and retrieval. A high rank reflects relative semantic similarity within this gallery.')


def failures() -> None:
    intro('05','Probe the limits','Does semantic similarity preserve counts, spatial relations, and attribute binding?',
          'These controlled probes test actual pretrained CLIP predictions. They are experimental questions, not guaranteed failures. Each experiment contains 30 labeled examples with known correct captions.')
    experiment = st.selectbox('Controlled experiment',['Counting','Spatial relationships','Attribute binding'])
    records = [r for r in manifest()['probes'] if r['experiment']==experiment]
    with st.spinner('Scoring all labeled examples…'):
        vectors = asset_embeddings(records)
        correct_text = encode_text([r['correct'] for r in records])
        wrong_text = encode_text([r['incorrect'] for r in records])
        correct = (vectors*correct_text).sum(-1)
        incorrect = (vectors*wrong_text).sum(-1)
        margins = similarity_margins(correct,incorrect)
    threshold = st.slider('Near-zero margin threshold',0.0,0.05,0.005,0.001,help='A display threshold, not a statistical significance or calibration claim.')
    frame = pd.DataFrame({'ID':[r['id'] for r in records],'Correct description':[r['correct'] for r in records],'Incorrect description':[r['incorrect'] for r in records],'Correct cosine':correct.numpy(),'Incorrect cosine':incorrect.numpy(),'Margin':margins.numpy(),'Correct ranked first':(margins>0).numpy()})
    metrics = st.columns(3)
    metrics[0].metric('Pairwise accuracy',f'{(margins>0).float().mean():.1%}')
    metrics[1].metric('Mean similarity margin',f'{margins.mean():+.4f}')
    metrics[2].metric('Near-zero margins',f'{int((margins.abs()<=threshold).sum())} / {len(records)}')
    st.caption('Accuracy counts strictly positive margins; exact ties count as unsuccessful. Near-zero describes score proximity and does not change the accuracy rule.')
    selected = st.selectbox('Inspect a labeled example',list(range(len(records))),format_func=lambda i:records[i]['id'])
    left,right = st.columns([1,1.5])
    with left:
        st.image(open_asset(records[selected]),width='stretch')
    with right:
        st.write(f'**Correct:** {records[selected]["correct"]}')
        st.write(f'**Distractor:** {records[selected]["incorrect"]}')
        pair = pd.DataFrame({'Description':['Correct','Distractor'],'Cosine similarity':[float(correct[selected]),float(incorrect[selected])]})
        bars(pair,'Description','Cosine similarity','Measured caption scores')
        margin = float(margins[selected])
        interpretation = 'Descriptions received similar scores' if abs(margin)<=threshold else ('Correct description ranked higher' if margin>0 else 'Incorrect description ranked higher')
        st.write(f'**Margin: {margin:+.4f}** · {interpretation}.')
    fig = go.Figure(go.Bar(x=frame.ID,y=frame.Margin,marker_color=['#137C75' if m>0 else '#C96948' for m in frame.Margin],hovertemplate='%{x}<br>Margin %{y:+.4f}<extra></extra>'))
    fig.add_hline(y=0,line_color='#172B42')
    fig.update_layout(title='Correct minus distractor · every prepared example',height=370,yaxis_title='Cosine similarity margin',xaxis_title='Labeled example',font=dict(size=15),paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,width='stretch')
    st.dataframe(frame,hide_index=True,width='stretch')
    downloadable(frame,f'{experiment.lower().replace(" ","_")}_results.csv')
    with st.expander('How to interpret these probes'):
        st.latex(r'm=\cos(v,u_{\mathrm{correct}})-\cos(v,u_{\mathrm{incorrect}})')
        st.write('A positive margin ranks the correct caption first; a negative margin ranks the distractor first. Counts are balanced over 2, 3, and 4 apples. Spatial and binding tests counterbalance both caption alternatives in matched pairs, varying position or background. There are only a few templates and nuisance variations; examples are correlated. These drawings are not a representative benchmark and the results do not establish broad model capabilities. Surface cues and rendering style can affect scores.')
    takeaway('Measure the model’s behavior before calling something a failure. Compare margins across controlled alternatives and examine the images.')
