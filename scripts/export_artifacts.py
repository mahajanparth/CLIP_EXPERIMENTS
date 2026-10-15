"""Export self-contained figures, an asset contact sheet, and measured report."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from clip_explorer.config import ROOT, SEED


def main() -> None:
    artifacts = ROOT/'artifacts'
    results = pd.read_csv(artifacts/'failure_results.csv')
    summary = pd.read_csv(artifacts/'failure_summary.csv')
    metadata = json.loads((artifacts/'embedding_metadata.json').read_text())
    figure = go.Figure()
    for experiment,rows in results.groupby('experiment',sort=False):
        figure.add_trace(go.Bar(name=experiment,x=rows.id,y=rows.margin))
    figure.add_hline(y=0)
    figure.update_layout(title='CLIP ViT-B/32 · measured controlled-probe margins',yaxis_title='Cosine(correct) − cosine(distractor)',height=600,template='plotly_white')
    figure.write_html(artifacts/'failure_margins.html',include_plotlyjs=True,full_html=True)
    prompts = pd.read_csv(artifacts/'prompt_results.csv')
    figure = go.Figure()
    for name,rows in prompts.groupby('strategy',sort=False):
        figure.add_trace(go.Bar(name=name,x=rows.id,y=[getattr(row,f'cosine_{row.label}') for row in rows.itertuples()]))
    figure.update_layout(title='Ground-truth class cosine · same real labeled photographs',yaxis_title='Cosine similarity',barmode='group',height=500,template='plotly_white')
    figure.write_html(artifacts/'prompt_comparison.html',include_plotlyjs=True,full_html=True)
    gallery = [r for r in json.loads((ROOT/'assets/manifest.json').read_text())['gallery'] if r['medium']=='photograph']
    from clip_explorer.embedding_space import default_records, project_joint_embeddings, point_table, embedding_figure, embedding_document
    from clip_explorer.thumbnail_space import thumbnail_document
    from clip_explorer.data import asset_embeddings, balanced_photos
    from clip_explorer.model import encode_text
    import torch
    from clip_explorer.collections import collection_names,collection_photos
    gallery=collection_photos(collection_names()[-1])
    records = balanced_photos(gallery,list(dict.fromkeys(r['category'] for r in gallery)),50)
    captions = [r['caption'] for r in records]
    vectors = torch.cat([asset_embeddings(records),encode_text(captions)])
    coordinates, variance = project_joint_embeddings(vectors.numpy())
    points = point_table(records,captions,coordinates)
    figure3=embedding_figure(points,'PCA',len(records))
    if len(records)>60:figure3.data[0].visible='legendonly'
    (artifacts/'embedding_space_3d.html').write_text(embedding_document(figure3,records,points,'PCA',len(records)<=60))
    (artifacts/'embedding_thumbnails_3d.html').write_text(thumbnail_document(points,records,'PCA',len(records)<=60))
    points.to_csv(artifacts/'embedding_coordinates_3d.csv',index=False)
    projection_metadata={}
    for method,suffix in [('PCA','pca'),('t-SNE','tsne')]:
        coords2,variance2=project_joint_embeddings(vectors.numpy(),method,8.0,dimensions=2)
        points2=point_table(records,captions,coords2)
        plot2=embedding_figure(points2,method,len(records))
        if len(records)>60:plot2.data[0].visible='legendonly'
        stem=f'embedding_space_2d_{suffix}'
        (artifacts/f'{stem}.html').write_text(embedding_document(plot2,records,points2,method,len(records)<=60))
        (artifacts/f'embedding_thumbnails_2d_{suffix}.html').write_text(thumbnail_document(points2,records,method,len(records)<=60))
        points2.to_csv(artifacts/f'embedding_coordinates_2d_{suffix}.csv',index=False)
        projection_metadata[method]={'dimensions':2,'input_shape':list(vectors.shape),'output_shape':list(coords2.shape),
            'seed':SEED,'asset_ids':[r['id'] for r in records],
            'explained_variance_ratio':variance2.tolist() if variance2 is not None else None,
            'perplexity':8.0 if method=='t-SNE' else None,
            'algorithm':('barnes_hut' if len(vectors)>500 else 'exact') if method=='t-SNE' else 'full SVD',
            'fit':'Joint image and text embeddings; independently fitted in 2D'}
        # Shareable static plots, using the same measured coordinates as the HTML.
        from matplotlib.figure import Figure
        from clip_explorer.embedding_space import category_color
        large=points2.Category.nunique()>10
        static=Figure(figsize=(14,11) if large else (10,7),layout=None if large else 'constrained')
        if large:static.subplots_adjust(bottom=.32)
        axis=static.subplots()
        for i in range(len(records) if not large else 0):
            axis.plot([coords2[i,0],coords2[len(records)+i,0]],[coords2[i,1],coords2[len(records)+i,1]],color='#c5ced6',linewidth=.8,zorder=1)
        for (category,modality),group in points2.groupby(['Category','Modality'],sort=False):
            axis.scatter(group.X,group.Y,c=category_color(category),marker='o' if modality=='Image' else 'D',
                         s=65,edgecolors='white',linewidths=.7,label=(category if modality=='Image' else '_nolegend_') if large else f'{category} · {modality}',zorder=2)
        axis.set(xlabel='PC 1' if method=='PCA' else 't-SNE 1',ylabel='PC 2' if method=='PCA' else 't-SNE 2',
                 title=f'CLIP ViT-B/32 · joint image/text embeddings · 2D {method}')
        axis.set_aspect('equal',adjustable='datalim')
        if large:axis.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=5,fontsize=8,title='Category colors · circles: images · diamonds: captions')
        else:axis.legend(loc='best',fontsize=8)
        axis.grid(alpha=.15)
        static.savefig(artifacts/f'{stem}.png',dpi=180)
        static.savefig(artifacts/f'{stem}.svg')
    (artifacts/'embedding_projection_metadata.json').write_text(json.dumps(projection_metadata,indent=2)+'\n')
    gallery=collection_photos('Curated gallery')
    width,cell_height = 160,190
    sheet = Image.new('RGB',(width*8,cell_height*((len(gallery)+7)//8)), '#f5f7fa')
    draw = ImageDraw.Draw(sheet)
    for idx,r in enumerate(gallery):
        with Image.open(ROOT/'assets'/r['file']) as image:
            thumbnail = image.convert('RGB'); thumbnail.thumbnail((145,150))
        x,y = (idx%8)*width,(idx//8)*cell_height
        sheet.paste(thumbnail,(x+(width-thumbnail.width)//2,y+5))
        draw.text((x+8,y+160),r['id'],fill='#172b42')
    sheet.save(artifacts/'gallery_contact_sheet.jpg',quality=90)
    rows = ['| Probe | N | Pairwise accuracy | Mean margin |','|---|---:|---:|---:|']
    rows.extend(f'| {r.experiment} | {r.examples} | {r.accuracy:.1%} | {r.mean_margin:+.4f} |' for r in summary.itertuples())
    strategy_rows = ['| Strategy | N | Accuracy |','|---|---:|---:|']
    for name,subset in prompts.groupby('strategy',sort=False):
        strategy_rows.append(f'| {name} | {len(subset)} | {subset.correct.mean():.1%} |')
    report = '\n'.join([
        '# Measured demonstration artifacts','',
        f'Original OpenAI CLIP **{metadata["model"]}**, upstream revision `{metadata["model_revision"]}`.',
        f'Prepared at {metadata["prepared_at_utc"]}; device `{metadata["device"]}`; seed {metadata["seed"]}.',
        f'Checkpoint SHA-256: `{metadata["checkpoint_sha256"]}`.',
        f'Learned logit_scale = {metadata["learned_logit_scale"]:.6f}; multiplier = exp(logit_scale).','',
        '## Controlled probes','',*rows,'',
        'Accuracy means correct cosine strictly exceeds the paired distractor cosine; ties count as unsuccessful. All 90 examples are controlled drawings. Both alternatives are counterbalanced in spatial and binding probes. Counting uses known counts of 2, 3, or 4. These correlated examples and repeated caption templates are demonstrations, not a representative benchmark. No outcomes are hardcoded.','',
        '## Prompt strategies','',*strategy_rows,'',
        'All strategies use the same real photographs labeled dog, cat, car, and bicycle and the same four candidate labels. Template ensembling normalizes each prompt, averages prompt vectors per class, then renormalizes. Probabilities in prompt_results.csv use the actual checkpoint scale and depend on this candidate set.','',
        '## Files','',
        '- `image_embeddings.npz`: normalized gallery and probe embeddings, without pickle.',
        '- `embedding_metadata.json`: model configuration, exact checkpoint and asset hashes, package versions.',
        '- `failure_results.csv`, `failure_summary.csv`: per-example scores and aggregates.',
        '- `prompt_results.csv`: per-image predictions, similarities, and candidate probabilities.',
        '- `failure_margins.html`, `prompt_comparison.html`: interactive standalone Plotly figures; no CDN needed.',
        '- `gallery_contact_sheet.jpg`: real photographic gallery preview; credits are in assets/PHOTO_CREDITS.md.',
        '- `embedding_space_2d_pca.html`, `embedding_space_2d_tsne.html`: interactive 2D projections of the same measured image/text vectors, with real photo thumbnails. Matching PNG/SVG plots, coordinate CSVs, and embedding_projection_metadata.json are included. t-SNE is fitted independently in 2D, not sliced from the 3D layout.',
        f'- `embedding_space_3d.html`, `embedding_coordinates_3d.csv`: shared PCA projection of {len(records)} measured image vectors and their {len(records)} caption vectors. The three axes retain ' + f'{variance.sum():.1%}' + ' of joint variance. Projection distances are lossy; cosine rankings use the original 512D space.','',
        'Regenerate with `python scripts/prepare.py` then `python scripts/export_artifacts.py`. No network is used when the checkpoint is already present.','',
    ])
    from scripts.export_process_artifacts import export_process_artifacts
    export_process_artifacts()
    report += '\n## End-to-end visual process guide\n\n`clip_process.html` embeds the training/testing architectures, an animated actual toy SGD trajectory, and measured frozen CLIP testing scores. `clip_training_architecture.svg` and `clip_testing_architecture.svg` are reusable vector diagrams. `toy_training_losses.csv` is explicitly synthetic toy-encoder optimization, not CLIP training history. `process_testing_results.csv` contains real local checkpoint scores for the default image and candidates.\n'
    (artifacts/'REPORT.md').write_text(report)
    print('Exported standalone figures, contact sheet, and measured report.')

if __name__=='__main__': main()
