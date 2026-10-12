"""Text token and feature activation inspector for frozen original CLIP."""
from __future__ import annotations
from io import BytesIO
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .text_activations import inspect_text_prompt,token_labels,contextual_stages
from .ui import intro,takeaway,downloadable


def activation_heatmap(trace: dict, stage: str, start: int=0, count: int=64, show_padding: bool=False) -> go.Figure:
    values=trace['stages'][stage]
    pooled=stage in ['Projected EOS','Normalized CLIP vector']
    labels=['EOS → whole-prompt vector'] if pooled else token_labels(trace,show_padding)
    values=values[:len(labels),start:start+count]
    limit=max(float(np.abs(values).max()),1e-8)
    figure=go.Figure(go.Heatmap(z=values.tolist(),x=list(range(start,start+values.shape[1])),y=labels,
        colorscale='RdBu',zmin=-limit,zmax=limit,
        hovertemplate='%{y}<br>Feature %{x}<br>Activation %{z:.6f}<extra></extra>'))
    figure.update_layout(title=f'{stage} · measured feature activations',height=max(330,min(1000,len(labels)*30+130)),
        xaxis_title='Feature coordinate (zero-based)',yaxis_title='Token position and BPE piece',yaxis_autorange='reversed',
        font=dict(size=15),margin=dict(l=190,r=20,t=60,b=60),paper_bgcolor='rgba(0,0,0,0)')
    return figure


def render() -> None:
    intro('08','Inspect a text prompt','What activations does CLIP compute for each token in your prompt?',
          'Inspect the actual token lookup vectors, learned positional embeddings, residual-stream outputs from all 12 text Transformer blocks, and final EOS readout. All values come from the frozen local ViT-B/32 checkpoint.')
    a,b=st.columns(2)
    prompt=a.text_input('Text prompt','a photo of a golden retriever',key='activation_prompt')
    compare=b.text_input('Optional comparison prompt','',placeholder='e.g. a photo of a cat',key='activation_compare')
    with st.spinner('Observing the frozen text encoder…'):
        trace=inspect_text_prompt(prompt)
        other=inspect_text_prompt(compare) if compare.strip() else None
    metrics=st.columns(3)
    metrics[0].metric('Tokens including start/end',trace['eos_index']+1)
    metrics[1].metric('Hidden features per token',trace['hidden_width'])
    metrics[2].metric('Projected text-vector dimensions',trace['output_width'])
    st.latex(r'E=\mathrm{Embedding}(\mathrm{token\ IDs}),\quad H^{(0)}=E+P,\quad H^{(\ell)}=\mathrm{Block}_{\ell}(H^{(\ell-1)})')
    st.latex(r'u=\mathrm{LN}(H^{(12)})_{\mathrm{EOS}}W_T,\qquad \hat u=u/\|u\|_2')
    st.caption('Forward-pass shapes: token IDs [1,77] → embeddings [1,77,512] → Transformer states [77,1,512] → selected EOS projection [1,512]. The charts omit the batch dimension and use token-row layout.')
    show_padding=st.checkbox('Include zero-filled positions after EOS',False)
    labels=token_labels(trace,show_padding)
    tokens=pd.DataFrame({'Position':range(len(labels)),'BPE piece':[trace['token_pieces'][i] if i<=trace['eos_index'] else '[zero-filled padding]' for i in range(len(labels))],
                         'Token ID':trace['token_ids'][:len(labels)],'Used for final readout':[i==trace['eos_index'] for i in range(len(labels))]})
    st.dataframe(tokens,hide_index=True,width='stretch')
    controls=st.columns([2,1,1])
    stage=controls[0].selectbox('Activation stage',list(trace['stages']),key='activation_stage')
    count=controls[1].selectbox('Features shown',[32,64,128,512],index=1)
    maximum=trace['stages'][stage].shape[1]-count
    start=controls[2].slider('First feature coordinate',0,maximum,0) if maximum>0 else 0
    pooled=stage in ['Projected EOS','Normalized CLIP vector']
    full=trace['stages'][stage]
    st.write(f'**Measured tensor:** {list(full.shape)} · **Displayed slice:** {1 if pooled else len(labels)} tokens × {count} features.')
    st.plotly_chart(activation_heatmap(trace,stage,start,count,show_padding),width='stretch')
    st.caption('Colors use a symmetric range fitted to the current displayed slice. Compare numeric values when switching stages; the color range can change.')
    if pooled:st.info('CLIP projects only the EOS hidden state into the image–text space. This stage has one whole-prompt vector, rather than one projected output per word.')
    selected=st.selectbox('Follow a token through the layers',range(len(labels)),index=trace['eos_index'],format_func=lambda i:labels[i],key='activation_token')
    feature=st.number_input('Feature coordinate to follow',min_value=0,max_value=trace['hidden_width']-1,value=0,step=1)
    stages=contextual_stages(trace)
    figure=go.Figure(go.Scatter(x=stages,y=[trace['stages'][name][selected,int(feature)] for name in stages],mode='lines+markers',line=dict(color='#137C75',width=3),name=labels[selected]))
    figure.update_layout(title=f'Actual feature {feature} for {labels[selected]}',height=350,yaxis_title='Activation value',xaxis_title='Forward stage',font=dict(size=14))
    st.plotly_chart(figure,width='stretch')
    # RMS summarizes magnitude, not attention or token importance.
    rms=np.stack([np.sqrt(np.mean(trace['stages'][name][:len(labels)]**2,axis=1)) for name in stages],axis=1)
    depth=go.Figure(go.Heatmap(z=rms.tolist(),x=stages,y=labels,colorscale='Teal',hovertemplate='%{y}<br>%{x}<br>RMS %{z:.5f}<extra></extra>'))
    depth.update_layout(title='Token activation magnitude across depth · RMS',height=max(350,min(1000,len(labels)*30+130)),xaxis_title='Forward stage',yaxis_autorange='reversed',font=dict(size=14),margin=dict(l=190,r=20,t=60,b=80))
    st.plotly_chart(depth,width='stretch')
    if other is not None:
        similarity=float(trace['stages']['Normalized CLIP vector'][0]@other['stages']['Normalized CLIP vector'][0])
        st.metric('Full 512D cosine between the two prompt vectors',f'{similarity:.4f}')
        comparison=go.Figure()
        for observed,color in [(trace,'#137C75'),(other,'#5279C1')]:
            comparison.add_scatter(x=contextual_stages(observed),y=[float(np.sqrt(np.mean(observed['stages'][name][observed['eos_index']]**2))) for name in contextual_stages(observed)],
                                   name=observed['prompt'],mode='lines+markers',line=dict(color=color,width=3))
        comparison.update_layout(title='Each prompt’s own EOS magnitude across depth',height=380,yaxis_title='EOS RMS activation',font=dict(size=14))
        st.plotly_chart(comparison,width='stretch')
        st.caption('Each curve follows its own EOS position, even when token counts differ. This is a magnitude comparison, not a measure of importance or model performance.')
    st.info('These are feature activations, not attention weights, saliency maps, or causal word importance. Large values do not automatically identify semantic concepts. CLIP uses causal text attention: a token can attend to earlier positions; EOS can summarize the whole prompt. The token lookup alone is context-independent; later hidden states are contextual.')
    exports=st.columns(2)
    buffer=BytesIO();np.savez_compressed(buffer,**{name:values for name,values in trace['stages'].items()})
    exports[0].download_button('Download full activation tensors · NPZ',buffer.getvalue(),file_name='text_prompt_activations.npz',mime='application/octet-stream')
    exports[1].download_button('Download prompt and token metadata · JSON',json.dumps({'prompt':trace['prompt'],'token_ids':trace['token_ids'].tolist(),'token_pieces':trace['token_pieces'],'eos_index':trace['eos_index'],
        'context_length':trace['context_length'],'stages':{name:list(values.shape) for name,values in trace['stages'].items()}},indent=2).encode(),file_name='text_prompt_metadata.json',mime='application/json')
    downloadable(tokens,'text_prompt_tokens.csv')
    with st.expander('Interpreting the input features'):
        st.write('Token IDs index a learned vocabulary table. Positional embeddings add information about token location. Each residual block combines causal self-attention and an MLP update; the displayed block outputs are the resulting residual states, not isolated neuron outputs from each MLP. Final layer normalization is applied, then the end-of-text state is projected and normalized for cosine matching. BPE pieces are byte-level fragments; “</w>” marks a word boundary. Positions after EOS are zero-filled, not a separate learned PAD token, and are not used for the final readout.')
    takeaway('You can inspect every token’s measured feature vector across depth, while distinguishing input lookup features from contextual states and the final whole-prompt embedding.')
