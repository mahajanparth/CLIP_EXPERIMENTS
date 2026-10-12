"""Visual end-to-end CLIP walkthrough: conceptual architecture, toy SGD, real testing."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import torch
from .data import manifest, open_asset, asset_embeddings, photo_for_label
from .model import encode_text, compute_logits, compute_probabilities
from . import model as clip_model
from .math_ops import cosine_similarity, format_prompts
from .training_demo import toy_training
from .process_visuals import (TRAIN_STAGES,TEST_STAGES,LABELS,pipeline_svg,svg_html,
                             matrix_figure,loss_animation,preprocess_svg)
from .ui import intro,takeaway,select_image,image_embedding,lines,bars,downloadable,photo_credits,photo_gallery


@st.cache_data(show_spinner=False,max_entries=12)
def _toy_history(learning_rate: float, temperature: float):
    return toy_training(40,learning_rate,temperature)


def _diagram(training: bool, active: int | None = None, toy: bool=False, image_count: int | None=None, text_count: int | None=None):
    components.html(svg_html(pipeline_svg(training,active,toy,image_count,text_count)),height=740,scrolling=False)
    if training:
        st.latex(r'P^{I\to T}_{ij}=\frac{\exp(L_{ij})}{\sum_k\exp(L_{ik})},\qquad P^{T\to I}_{ij}=\frac{\exp(L_{ij})}{\sum_k\exp(L_{kj})}')
    else:
        st.latex(r'P_{ic}=\frac{\exp(L_{ic})}{\sum_{k=1}^{C}\exp(L_{ik})},\qquad\mathrm{accuracy}=\frac1B\sum_{i=1}^{B}\mathbf{1}[\hat y_i=y_i^*]')


def _stage_control(stages: list[str], key: str) -> int:
    selected=st.select_slider('Walkthrough stage',options=stages,value=stages[0],key=key)
    index=stages.index(selected)
    def move(offset):
        st.session_state[key]=stages[max(0,min(len(stages)-1,stages.index(st.session_state[key])+offset))]
    previous,_,next_button=st.columns([1,4,1])
    previous.button('← Previous',on_click=move,args=(-1,),disabled=index==0,key=key+'_prev')
    next_button.button('Next →',on_click=move,args=(1,),disabled=index==len(stages)-1,key=key+'_next')
    return index


def _vectors(values: np.ndarray, labels: list[str], title: str, limit: int | None = None):
    visible=values[:,:limit] if limit else values
    figure=go.Figure(go.Heatmap(z=visible,x=list(range(1,visible.shape[1]+1)),y=labels,
        colorscale='RdBu',zmid=0,hovertemplate='%{y}<br>Coordinate %{x}: %{z:.4f}<extra></extra>'))
    figure.update_layout(title=title,height=max(240,len(labels)*38+120),xaxis_title='Embedding coordinate',
        yaxis_autorange='reversed',font=dict(size=14),paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=80,r=10,t=50,b=45))
    st.plotly_chart(figure,width='stretch')


def _training():
    st.info('TOY TRAINING · Actual gradients on two tiny 4→8 linear encoders and a learned scale, using synthetic one-hot feature inputs. The pictures illustrate pair identities. No pretrained CLIP parameters or real image encoders are trained.')
    controls=st.columns(3)
    rate=controls[0].select_slider('Toy learning rate',options=[.01,.03,.08,.2],value=.08)
    temperature=controls[1].select_slider('Initial toy temperature',options=[.1,.2,.5,1.],value=.2)
    iteration=controls[2].slider('Optimizer step to inspect',0,40,0)
    history=_toy_history(rate,temperature)
    snapshot=history[iteration]
    stage=_stage_control(TRAIN_STAGES,'train_cycle_stage')
    _diagram(True,stage,toy=True)
    metrics=st.columns(3)
    for col,title,key in zip(metrics,['Image → text CE','Text → image CE','Symmetric loss'],['loss_i2t','loss_t2i','loss']):
        col.metric(title,f'{snapshot[key]:.4f}')
    st.subheader(TRAIN_STAGES[stage])
    records=[photo_for_label(label) for label in LABELS]
    if stage==0:
        columns=st.columns(4)
        for idx,(col,record) in enumerate(zip(columns,records)):
            col.image(open_asset(record),caption=f'Pair {idx+1} · {record["caption"]}',width='stretch')
            col.code('Feature '+str(np.eye(4,dtype=int)[idx].tolist()),language=None)
        photo_credits(records)
        st.write('In real CLIP pretraining, a minibatch contains images and their accompanying natural-language descriptions. Each image must identify its own caption among the batch. The toy substitutes one-hot features for both modalities so the entire optimization can be inspected.')
    elif stage==1:
        left,right=st.columns(2)
        with left:_vectors(snapshot['image_weights'],LABELS,'Image linear encoder · current weights')
        with right:_vectors(snapshot['text_weights'],LABELS,'Text linear encoder · current weights')
        st.latex(r'v_i=f_I(x_i;\theta_I),\quad u_i=f_T(t_i;\theta_T)')
        st.write('Each toy encoder maps a four-dimensional feature vector to eight dimensions. They have independent weights. Original CLIP instead uses an image encoder and text Transformer followed by learned projections to a common embedding dimension.')
    elif stage==2:
        a,b=st.columns(2)
        with a:_vectors(snapshot['images'],LABELS,'Normalized image vectors · all 8 coordinates')
        with b:_vectors(snapshot['texts'],LABELS,'Normalized text vectors · all 8 coordinates')
        norms=pd.DataFrame({'Pair':LABELS,'Raw image norm':np.linalg.norm(snapshot['raw_images'],axis=1),
            'Unit image norm':np.linalg.norm(snapshot['images'],axis=1),'Raw text norm':np.linalg.norm(snapshot['raw_texts'],axis=1),
            'Unit text norm':np.linalg.norm(snapshot['texts'],axis=1)})
        st.dataframe(norms,hide_index=True,width='stretch')
        st.latex(r'\hat v_i=v_i/\|v_i\|_2,\qquad\hat u_i=u_i/\|u_i\|_2')
    elif stage==3:
        st.plotly_chart(matrix_figure(snapshot['similarity'],'Every image is compared with every caption'),width='stretch')
        st.latex(r'S_{ij}=\hat v_i^\top\hat u_j,\qquad L=e^\alpha S')
        st.write(f'Gold outlines are the four matching pairs. The other 12 cells are in-batch negatives. The current learned toy multiplier is {snapshot["scale"]:.3f}; its equivalent temperature is {snapshot["temperature"]:.4f}.')
    elif stage==4:
        a,b=st.columns(2)
        with a:st.plotly_chart(matrix_figure(snapshot['image_to_text'],'Image → text · each ROW sums to 1',True),width='stretch')
        with b:st.plotly_chart(matrix_figure(snapshot['text_to_image'],'Text → image · each COLUMN sums to 1',True),width='stretch')
        selected=st.selectbox('Follow a matching pair',range(4),format_func=lambda i:LABELS[i])
        st.write(f'For **{LABELS[selected]}**, its positive image → text probability is **{snapshot["image_to_text"][selected,selected]:.4f}**, and positive text → image probability is **{snapshot["text_to_image"][selected,selected]:.4f}**. Both directions must recover the correct partner.')
    elif stage==5:
        figure=go.Figure()
        figure.add_bar(name='Image → text: −log P',x=LABELS,y=snapshot['row_nll'],marker_color='#137C75')
        figure.add_bar(name='Text → image: −log P',x=LABELS,y=snapshot['column_nll'],marker_color='#5279C1')
        figure.update_layout(title='Per-positive cross-entropy contributions',height=400,barmode='group',yaxis_title='Negative log probability',font=dict(size=16))
        st.plotly_chart(figure,width='stretch')
        st.latex(r'\mathcal L_{I\to T}=-\frac1N\sum_i\log P^{I\to T}_{ii},\quad\mathcal L_{T\to I}=-\frac1N\sum_i\log P^{T\to I}_{ii}')
        st.latex(r'\mathcal L_{\mathrm{CLIP}}=\tfrac12(\mathcal L_{I\to T}+\mathcal L_{T\to I})')
        st.write('Cross-entropy penalizes low probability assigned to the known positive. Average across the batch within each direction, then average the two directional losses. Targets are pair indices; the diagonal is correct here because both lists are aligned.')
    else:
        gradients=pd.DataFrame({'Parameter':['Image encoder weights','Text encoder weights','Logit scale α'],'Gradient norm':snapshot['gradient_norms']})
        fig=go.Figure(go.Bar(x=gradients.Parameter,y=gradients['Gradient norm'],marker_color=['#137C75','#5279C1','#BB617D']))
        fig.update_layout(title='Gradients computed from the symmetric loss',height=320,yaxis_title='Gradient norm',font=dict(size=16))
        st.plotly_chart(fig,width='stretch')
        st.latex(r'\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L,\qquad\alpha\leftarrow\alpha-\eta\nabla_\alpha\mathcal L')
        if iteration<40:
            next_snapshot=history[iteration+1]
            st.metric('Loss after the next actual toy SGD update',f'{next_snapshot["loss"]:.4f}',delta=f'{next_snapshot["loss"]-snapshot["loss"]:+.4f}',delta_color='inverse')
        st.write('Gradients flow through the scale, dot products, normalization, and both toy encoders. Move the optimizer-step slider to inspect the changed weights and matrix. Original CLIP jointly learns its encoders and logit scale with its own optimizer and large minibatches; this tiny SGD example illustrates the mechanism only.')
    st.divider()
    st.subheader('Play the complete learning cycle')
    st.caption('Each animation frame is an actual recorded toy optimizer step. It has its own playback slider; the inspection controls above remain independent. The same batch is reused, so this curve does not demonstrate generalization.')
    animation=loss_animation(history)
    st.plotly_chart(animation,width='stretch')
    frame=pd.DataFrame([{key:s[key] for key in ['iteration','loss_i2t','loss_t2i','loss','scale','temperature']} for s in history])
    downloadable(frame,'toy_training_losses.csv')
    st.download_button('Download animated training cycle · HTML',animation.to_html(include_plotlyjs=True,full_html=True).encode(),file_name='toy_training_cycle.html',mime='text/html')
    takeaway('Both encoders receive gradients from the same symmetric matching loss. A toy optimizer makes the update visible without retraining pretrained CLIP.')


def _testing():
    st.success('REAL PRETRAINED INFERENCE · Original OpenAI CLIP ViT-B/32. Encoders and learned scale are frozen; no backward pass or optimizer step is performed.')
    setup_left,setup_right=st.columns([1,2])
    with setup_left:
        image,record=select_image('process_test')
        st.image(image,caption='Original test image',width='stretch')
        if record:photo_credits([record])
    with setup_right:
        classes=lines(st.text_area('Testing candidates · one per line','dog\ncat\ncar\nbicycle',height=130))
        template=st.text_input('Testing prompt template','a photo of a {}')
        descriptions=format_prompts(classes,template)
    with st.spinner('Tracing the actual frozen forward pass…'):
        images=image_embedding(image,record)
        texts=encode_text(descriptions)
        similarity=cosine_similarity(images,texts)[0]
        logits=compute_logits(images,texts)[0]
        probability=compute_probabilities(logits.unsqueeze(0))[0]
        runtime=clip_model.load_clip_model()
        scale=float(runtime.model.logit_scale.detach().float().exp().cpu())
    results=pd.DataFrame({'Class':classes,'Prompt':descriptions,'Cosine similarity':similarity.numpy(),'Scaled logit':logits.numpy(),'Probability':probability.numpy()})
    stage=_stage_control(TEST_STAGES,'test_cycle_stage')
    evaluation_count=sum(r['medium']=='photograph' and r['label'] in classes for r in manifest()['gallery'])
    diagram_batch=evaluation_count if stage==6 and evaluation_count else 1
    _diagram(False,stage,image_count=diagram_batch,text_count=len(classes))
    if stage==6:
        st.caption(f'The evaluation diagram uses B = {diagram_batch} labeled photographs; earlier stages use the single selected image.')
    st.subheader(TEST_STAGES[stage])
    if stage==0:
        st.write('The test image and candidate descriptions form two input streams. A supplied vocabulary determines what the model can predict; it does not require a fitted classification head.')
        columns=st.columns(min(4,len(classes)))
        for idx,description in enumerate(descriptions):columns[idx%len(columns)].info(description)
    elif stage==1:
        # Derive the displayed crop from this checkpoint's actual preprocessing.
        tensor=runtime.preprocess(image)
        normalization=runtime.preprocess.transforms[-1]
        mean=torch.tensor(normalization.mean).view(3,1,1)
        std=torch.tensor(normalization.std).view(3,1,1)
        pixels=((tensor*std+mean).clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
        left,right=st.columns([1,2])
        with left:
            components.html(preprocess_svg(Image.fromarray(pixels)),height=290)
            st.caption('Actual resized/center-cropped 224×224 input. The overlay shows 49 patches, each 32×32 pixels. The model also uses a class token and pixel normalization.')
        with right:
            import clip
            tokens=clip.tokenize(descriptions)
            chosen=st.selectbox('Inspect tokenizer output',range(len(classes)),format_func=lambda i:descriptions[i])
            count=int(tokens[chosen].argmax())+1
            actual=tokens[chosen,:count].tolist()
            pieces=[clip.clip._tokenizer.decoder[token] for token in actual]
            st.code(' → '.join(pieces),language=None)
            st.caption('Actual original CLIP BPE token pieces, including start/end tokens. “</w>” denotes a word boundary; the context is padded to 77 tokens.')
            st.dataframe(pd.DataFrame({'Position':range(count),'BPE piece':pieces,'Token ID':actual}),hide_index=True,width='stretch')
    elif stage==2:
        a,b=st.columns(2)
        with a:_vectors(images.numpy(),['Test image'],'Image embedding · first 32 of 512 coordinates',32)
        with b:_vectors(texts.numpy(),classes,'Text embeddings · first 32 of 512 coordinates',32)
        st.write('The encoders use different architectures but output vectors in the same 512-dimensional space. This display shows a coordinate slice; all scoring uses complete normalized vectors.')
        st.metric('Image embedding L2 norm',f'{images.norm():.6f}')
        st.caption(f'Text vector norms range from {float(texts.norm(dim=1).min()):.6f} to {float(texts.norm(dim=1).max()):.6f}.')
    elif stage==3:
        bars(results,'Class','Cosine similarity','Match the test image with every description')
        st.latex(r's_c=\hat v^\top\hat u_c')
        st.write('These dot products are cosine similarities, not probabilities. Text vectors serve as the candidate classifier weights.')
    elif stage==4:
        a,b=st.columns(2)
        with a:
            fig=go.Figure(go.Bar(x=logits.numpy(),y=classes,orientation='h',marker_color='#9666B5'))
            fig.update_layout(title='1 · Learned scale → logits',height=370,xaxis_title='Scaled logit',font=dict(size=16))
            st.plotly_chart(fig,width='stretch')
        with b:bars(results,'Class','Probability','2 · Softmax over supplied candidates',True)
        st.latex(r'L_c=e^\alpha s_c,\qquad P(c\mid I,\mathcal C)=\frac{e^{L_c}}{\sum_{k\in\mathcal C}e^{L_k}}')
        st.write(f'The actual checkpoint multiplier exp(α) is **{scale:.4f}**. Probabilities sum to **{float(probability.sum()):.6f}** across this candidate set. They are not calibrated confidence; try adding or removing a candidate above.')
    elif stage==5:
        best=int(similarity.argmax())
        metrics=st.columns(3)
        metrics[0].metric('Highest-ranked candidate',classes[best])
        metrics[1].metric('Cosine similarity',f'{float(similarity[best]):.4f}')
        metrics[2].metric('Candidate-dependent probability',f'{float(probability[best]):.1%}')
        st.dataframe(results.sort_values('Cosine similarity',ascending=False),hide_index=True,width='stretch')
        st.write('Argmax of cosine, scaled logit, and softmax gives the same candidate because the scale is positive. Without a usable ground-truth label, this is a ranking, not a verified correct answer. There is no model update after prediction.')
        downloadable(results,'testing_forward_pass.csv')
    else:
        labeled=[r for r in manifest()['gallery'] if r['medium']=='photograph' and r['label'] in classes]
        if not labeled:
            st.info('No prepared photograph ground-truth labels match these candidates. Use dog, cat, car, and bicycle for the default photographic evaluation.')
            return
        scores=cosine_similarity(asset_embeddings(labeled),texts)
        predictions=[classes[int(row.argmax())] for row in scores]
        truths=[r['label'] for r in labeled]
        accuracy=np.mean([prediction==truth for prediction,truth in zip(predictions,truths)])
        st.caption('The evaluation uses these real photographs:')
        photo_gallery(labeled,columns=4)
        cols=st.columns(2)
        cols[0].metric('Accuracy on prepared labeled photographs',f'{accuracy:.1%}')
        cols[1].metric('Labeled examples',len(labeled))
        confusion=np.zeros((len(classes),len(classes)),dtype=int)
        for truth,prediction in zip(truths,predictions):confusion[classes.index(truth),classes.index(prediction)]+=1
        fig=go.Figure(go.Heatmap(z=confusion,x=classes,y=classes,colorscale='Teal',text=confusion,texttemplate='%{text}',hovertemplate='True %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>'))
        fig.update_layout(title='Measured evaluation · true vs predicted',xaxis_title='Predicted class',yaxis_title='Known true label',yaxis_autorange='reversed',height=450,font=dict(size=16))
        st.plotly_chart(fig,width='stretch')
        st.caption('Small, curated photographic set; not a benchmark estimate or the paper’s evaluation. Only prepared photographs whose known label belongs to this candidate set are evaluated. No training occurs on these examples.')
    takeaway('Testing is a frozen forward pass: image + class prompts → normalized vectors → learned-scale scores → candidate ranking. Labels are used only to evaluate the predictions.')


def render():
    intro('07','CLIP, from inputs to decisions','What changes between learning image–text alignment and using it?',
          'Follow the two data streams, inspect the losses that join them, and trace an actual zero-shot prediction. The visual guide separates the conceptual CLIP architecture, an inspectable toy training loop, and measured pretrained inference.')
    mode=st.radio('Visual walkthrough',['Architecture overview','Training cycle','Testing cycle'],horizontal=True)
    if mode=='Training cycle':_training()
    elif mode=='Testing cycle':_testing()
    else:
        st.subheader('Real image–caption examples')
        photo_gallery([photo_for_label(label) for label in LABELS],columns=4)
        st.subheader('Learning: paired supervision connects both encoders')
        _diagram(True)
        st.latex(r'(N\times512)(512\times N)\to N\times N,\quad L=e^{\alpha}\hat V\hat T^\top')
        st.latex(r'\mathcal L=\frac12[\mathrm{CE}(L,y)+\mathrm{CE}(L^\top,y)],\qquad y=(0,\ldots,N-1)')
        st.write('A batch of matching images and descriptions creates a square similarity matrix. Two cross-entropies supervise the two matching directions. Their average supplies gradients to both encoders and the learned scale. The optimizer updates the parameters before the next batch.')
        st.subheader('Testing: words construct a classifier')
        _diagram(False)
        st.latex(r'(B\times512)(512\times C)\to B\times C,\quad P=\mathrm{softmax}_{\mathrm{classes}}(e^{\alpha}S)')
        st.write('A new image and candidate class prompts pass through the trained encoders. Normalized dot products, the learned scale, and a candidate softmax give a ranking. Ground-truth labels can evaluate predictions, but are not inputs to classification.')
        st.download_button('Download training architecture · SVG',pipeline_svg(True).encode(),file_name='clip_training_architecture.svg',mime='image/svg+xml')
        st.download_button('Download testing architecture · SVG',pipeline_svg(False).encode(),file_name='clip_testing_architecture.svg',mime='image/svg+xml')
        takeaway('Learning closes a loop through gradients. Testing reuses the learned alignment with frozen parameters and a supplied vocabulary.')
