"""Code-native SVG architecture diagrams and linked Plotly learning animations."""
from __future__ import annotations
import base64
from html import escape
from io import BytesIO
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

LABELS = ['dog','cat','car','bicycle']
PALETTE = ['#137C75','#5279C1','#D08036','#9666B5']
TRAIN_STAGES = ['1 · Paired inputs','2 · Two encoders','3 · Normalize vectors','4 · All pairwise scores',
                '5 · Two probability directions','6 · Symmetric loss','7 · Backpropagation + update']
TEST_STAGES = ['1 · Image + candidate classes','2 · Preprocess + tokenize','3 · Frozen encoders',
               '4 · Cosine similarity','5 · Scale + softmax','6 · Prediction','7 · Labeled evaluation']


def image_uri(image: Image.Image) -> str:
    buffer = BytesIO();image.convert('RGB').save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def pipeline_svg(training: bool, active: int | None = None, toy: bool = False,
                 image_count: int | None = None, text_count: int | None = None) -> str:
    """Dimensioned dual-encoder graph; equations are embedded vector glyphs.

    Original ViT-B/32: 224px images, 49 patches plus CLS, visual width 768,
    text width 512, context 77, joint dimension 512. Training has N aligned
    pairs; inference has B images and C candidate descriptions. The toy has
    four 4D feature inputs, independent 4×8 maps, and an 8D joint space.
    """
    from .diagram_math import math_svg
    if toy and not training:
        raise ValueError('Toy encoders are only used in the training demonstration.')
    if any(n is not None and n<1 for n in [image_count,text_count]):
        raise ValueError('Image and text counts must be positive.')
    if training and image_count is not None and text_count is not None and image_count!=text_count:
        raise ValueError('Training requires the same number of images and matched texts.')
    n=str(image_count or text_count or 4) if toy else str(image_count or text_count or 'N')
    b=str(image_count or 'B');c=str(text_count or 'C')
    ni,nt=(n,n) if training else (b,c)
    d=8 if toy else 512
    mode='TOY TRAINING · synthetic features' if toy else ('CLIP TRAINING · aligned pairs' if training else 'CLIP TESTING · frozen encoders')
    pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="740" viewBox="0 0 1280 740" role="img" aria-label="{mode}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#72849a"/></marker></defs>',
        '<rect width="1280" height="740" rx="22" fill="#f5f7fa"/>',
        '<style>text{font-family:Arial,sans-serif}.wire{stroke:#72849a;stroke-width:2.5;fill:none;marker-end:url(#arrow)}.pulse{stroke-dasharray:8 10;animation:flow 1.5s linear infinite}@keyframes flow{to{stroke-dashoffset:-36}}</style>']
    def text(x,y,value,size=16,color='#617888',weight='normal',anchor='middle'):
        pieces.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(value)}</text>')
    def wire(path,animate=False):pieces.append(f'<path d="{path}" class="wire {"pulse" if animate else ""}"/>')
    def card(x,y,w,h,title,lines,color,stage):
        focus=active is None or active==stage
        fg='#ffffff' if focus else '#172B42'
        pieces.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{color if focus else "#ffffff"}" stroke="{color if focus else "#d6e0e8"}" stroke-width="{3 if focus else 1.5}"/>')
        text(x+w/2,y+29,title,19,fg,'bold')
        for index,(kind,value) in enumerate(lines):
            row_y=y+43+index*(36 if len(lines)==2 else 29)
            if kind=='math':pieces.append(math_svg(value,x+12,row_y,w-24,34 if r'\frac' in value or r'\sum' in value else 26,fg))
            else:text(x+w/2,row_y+19,value,14,fg)
    text(640,30,mode,23,'#172B42','bold')
    legend=(f'N = {n} pairs; d = {d}. Shapes follow [batch, features].' if toy else
            ('N = aligned image–caption pairs; d = 512 shared coordinates.' if training else
             f'B = {b} images; C = {c} candidate descriptions; d = 512.'))
    text(640,58,legend,16)
    # Forward streams: input → encoder/projection → normalized vectors.
    for yy in [157,344]:
        wire(f'M226 {yy} H260',active in [1,2]);wire(f'M526 {yy} H560',active==2)
    wire('M766 157 H800 V229 H834',active==3)
    wire('M766 344 H800 V270 H834',active==3)
    wire('M1010 250 H1042',active==4)
    input_stage=1 if not training and active==1 else 0
    if toy:
        image_lines=[('text','One-hot synthetic inputs'),('math',rf'X_I\in\mathbb{{R}}^{{{n}\times4}}')]
        text_lines=[('text','One-hot synthetic inputs'),('math',rf'X_T\in\mathbb{{R}}^{{{n}\times4}}')]
        encoder_i=[('math',r'W_I\in\mathbb{R}^{4\times8}'),('math',r'V=X_IW_I'),('text',f'Output: {n} × 8')]
        encoder_t=[('math',r'W_T\in\mathbb{R}^{4\times8}'),('math',r'T=X_TW_T'),('text',f'Output: {n} × 8')]
    else:
        image_lines=[('text','Resize + sampled square crop' if training else 'Resize + center crop'),('math',rf'X_I\in\mathbb{{R}}^{{{ni}\times3\times224\times224}}')]
        text_lines=[('text','Prompt + BPE token IDs'),('math',rf'X_T\in\mathbb{{Z}}^{{{nt}\times77}}')]
        encoder_i=[('text',f'49 patches + CLS: {ni} × 50 × 768'),('math',r'W_I\in\mathbb{R}^{768\times512}'),('math',rf'V=H_{{\mathrm{{CLS}}}}W_I\in\mathbb{{R}}^{{{ni}\times512}}')]
        encoder_t=[('text',f'Hidden sequence: {nt} × 77 × 512'),('math',r'W_T\in\mathbb{R}^{512\times512}'),('math',rf'T=H_{{\mathrm{{EOS}}}}W_T\in\mathbb{{R}}^{{{nt}\times512}}')]
    card(16,91,210,133,'Image features' if toy else 'Image tensor',image_lines,'#137C75',input_stage)
    card(16,278,210,133,'Text features' if toy else 'Text tokens',text_lines,'#5279C1',input_stage)
    card(260,91,266,133,'Image encoder + projection',encoder_i,'#137C75',1 if training else 2)
    card(260,278,266,133,'Text encoder + projection',encoder_t,'#5279C1',1 if training else 2)
    card(560,91,206,133,'Unit image vectors',[
        ('math',rf'\hat V\in\mathbb{{R}}^{{{ni}\times{d}}}'),('math',r'\hat v_i=v_i/\|v_i\|_2')],'#137C75',2)
    card(560,278,206,133,'Unit text vectors',[
        ('math',rf'\hat T\in\mathbb{{R}}^{{{nt}\times{d}}}'),('math',r'\hat t_j=t_j/\|t_j\|_2')],'#5279C1',2)
    shape=f'{n}\\times {n}' if training else f'{b}\\times {c}'
    card(834,177,176,147,'Cosine matrix',[
        ('math',r'S=\hat V\hat T^{\mathsf{T}}'),('math',rf'S\in\mathbb{{R}}^{{{shape}}}'),('text','Contract the feature axis')],'#9666B5',3)
    card(1042,177,222,147,'Scaled logits',[
        ('math',r'L=e^{\alpha}S'),('math',rf'L\in\mathbb{{R}}^{{{shape}}}'),('math',r'\alpha\in\mathbb{R},\quad\tau=e^{-\alpha}')],'#D08036',4)
    # Dimensions of the contraction are shown directly below the junction.
    pieces.append(math_svg(rf'({ni}\times{d})\;({d}\times {nt})\;\longrightarrow\;({shape})',814,350,448,31,'#172B42'))
    wire('M1153 324 V464 H1025',active==4)
    if training:
        card(785,425,480,125,'Two matching directions',[
            ('math',r'P^{I\to T}_{ij}=\frac{e^{L_{ij}}}{\sum_k e^{L_{ik}}},\quad P^{T\to I}_{ij}=\frac{e^{L_{ij}}}{\sum_k e^{L_{kj}}}'),
            ('text',f'Both {n} × {n}; I→T rows sum to 1; T→I columns sum to 1')],'#D08036',4)
        wire('M785 488 H733',active==5)
        card(257,425,476,125,'Symmetric cross-entropy · scalar',[
            ('math',r'\mathcal{L}_{I\to T}=\mathrm{CE}(L,y),\quad\mathcal{L}_{T\to I}=\mathrm{CE}(L^{\mathsf{T}},y)'),
            ('math',r'\mathcal{L}=\frac{1}{2}(\mathcal{L}_{I\to T}+\mathcal{L}_{T\to I})')],'#D08036',5)
        wire('M495 550 V596',active==6)
        card(257,596,476,106,'Backpropagation + SGD' if toy else 'Backpropagation + AdamW',[
            ('math',r'\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L},\quad\alpha\leftarrow\alpha-\eta\nabla_{\alpha}\mathcal{L}' if toy else r'(\theta,\alpha)\leftarrow\mathrm{AdamWStep}(\theta,\alpha,\nabla\mathcal{L})'),
            ('text','θ includes both encoders and their projections')],'#BB617D',6)
        wire('M257 649 H239 V245 H392 V224',active==6)
        wire('M239 245 V413 H392 V411',active==6)
        text(1025,588,f'Known pair targets: y = (0, …, {n} − 1)',16,'#172B42')
        text(1025,615,'Diagonal targets require aligned ordering.',14)
        text(1025,645,'Gradient shape = corresponding parameter shape.',14)
        text(640,727,'Toy: actual SGD on one reused batch. Original CLIP: joint optimization over paired minibatches.',14)
    else:
        card(785,425,480,125,'Candidate softmax · normalize each image row',[
            ('math',r'P=\mathrm{softmax}_{\mathrm{classes}}(L)'),
            ('math',rf'P\in\mathbb{{R}}^{{{shape}}},\quad\sum_cP_{{ic}}=1')],'#D08036',4)
        wire('M785 488 H733',active==5)
        card(257,425,476,125,'Prediction · one class index per image',[
            ('math',r'\hat y_i=\operatorname{argmax}_c S_{ic}'),('math',rf'\hat y\in\{{0,\ldots,{c}-1\}}^{{{b}}}')],'#D08036',5)
        wire('M495 550 V596',active==6)
        card(257,596,476,106,'Optional evaluation · scalar accuracy',[
            ('math',r'\mathrm{accuracy}=n_{\mathrm{correct}}/B'),
            ('text','Ground-truth labels are not classifier inputs')],'#9666B5',6)
        text(1025,588,'Encoders, projections, and α stay frozen.',16,'#137C75','bold')
        text(1025,617,'Probabilities depend on supplied candidates.',14)
        text(1025,646,'They are not calibrated confidence.',14)
        text(640,727,'Testing uses a forward pass only. No training loss, backward pass, or parameter update.',14)
    pieces.append('</svg>')
    return ''.join(pieces)


def svg_html(svg: str, height: int = 740) -> str:
    responsive=svg.replace('width="1280"','width="100%"')
    return f'<div style="width:100%;overflow:auto">{responsive}</div>'


def matrix_figure(values: np.ndarray, title: str, probability: bool=False, labels: list[str] | None=None) -> go.Figure:
    labels = labels or LABELS
    fig = go.Figure(go.Heatmap(z=values,x=labels,y=labels,colorscale='Teal',
        zmin=0 if probability else -1,zmax=1,text=values,texttemplate='%{text:.3f}',
        hovertemplate='Image: %{y}<br>Text: %{x}<br>%{z:.4f}<extra></extra>'))
    for i in range(len(labels)):
        fig.add_shape(type='rect',x0=i-.46,x1=i+.46,y0=i-.46,y1=i+.46,line=dict(color='#D08036',width=3))
    fig.update_layout(title=title,height=380,xaxis_title='Text',yaxis_title='Image',yaxis_autorange='reversed',
        font=dict(size=16),margin=dict(l=70,r=10,t=50,b=50),paper_bgcolor='rgba(0,0,0,0)')
    return fig


def loss_animation(history: list[dict]) -> go.Figure:
    """Each frame couples the current cosine matrix with actual toy loss history."""
    figure = make_subplots(rows=1,cols=2,column_widths=[.47,.53],horizontal_spacing=.15,
                           subplot_titles=['Matching matrix · cosine','Measured toy objective · SGD steps'])
    def traces(iteration):
        snapshot=history[iteration]
        result=[go.Heatmap(z=snapshot['similarity'],x=LABELS,y=LABELS,zmin=-1,zmax=1,
             colorscale='Teal',showscale=False,text=snapshot['similarity'],texttemplate='%{text:.2f}')]
        for key,label,color in [('loss_i2t','Image → text','#137C75'),('loss_t2i','Text → image','#5279C1'),('loss','Symmetric loss','#D08036')]:
            result.append(go.Scatter(x=list(range(iteration+1)),y=[s[key] for s in history[:iteration+1]],
                                    mode='lines+markers',marker=dict(size=4),line=dict(color=color,width=3),name=label))
        return result
    initial=traces(0)
    figure.add_trace(initial[0],row=1,col=1)
    for trace in initial[1:]:figure.add_trace(trace,row=1,col=2)
    figure.update_yaxes(autorange='reversed',row=1,col=1)
    figure.update_xaxes(range=[0,max(1,len(history)-1)],title='Optimizer step',row=1,col=2)
    maximum=max(s[k] for s in history for k in ['loss_i2t','loss_t2i','loss'])
    figure.update_yaxes(range=[0,maximum*1.08],title='Cross-entropy',row=1,col=2)
    for i in range(4):
        figure.add_shape(type='rect',x0=i-.46,x1=i+.46,y0=i-.46,y1=i+.46,xref='x',yref='y',line=dict(color='#D08036',width=3))
    figure.frames=[go.Frame(name=str(i),data=traces(i),traces=[0,1,2,3]) for i in range(len(history))]
    figure.update_layout(height=490,title='Actual toy gradient updates · never pretrained CLIP training',
        legend=dict(orientation='h',y=-.18),margin=dict(t=85,b=90,l=50,r=10),font=dict(size=15),
        paper_bgcolor='rgba(0,0,0,0)',
        updatemenus=[dict(type='buttons',x=0,y=-.12,buttons=[
            dict(label='▶ Play learning cycle',method='animate',args=[None,dict(frame=dict(duration=180,redraw=True),transition=dict(duration=0),fromcurrent=True)]),
            dict(label='Pause',method='animate',args=[[None],dict(frame=dict(duration=0,redraw=False),mode='immediate')])])],
        sliders=[dict(active=0,x=.37,len=.63,y=-.06,currentvalue=dict(prefix='Step '),
            steps=[dict(label=str(i),method='animate',args=[[str(i)],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i in range(len(history))])])
    return figure


def preprocess_svg(image: Image.Image) -> str:
    """A grid on the actual preprocessed 224px image: ViT-B/32's 7×7 patches."""
    lines=[]
    for i in range(1,7):
        position=32*i
        lines.extend([f'<line x1="{position}" y1="0" x2="{position}" y2="224"/>',
                      f'<line x1="0" y1="{position}" x2="224" y2="{position}"/>'])
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="280" height="280" viewBox="0 0 224 224"><image href="{image_uri(image)}" width="224" height="224"/><g stroke="#f7fbff" stroke-width="1" opacity=".8">{"".join(lines)}</g></svg>'
