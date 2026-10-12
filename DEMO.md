# Twelve-minute classroom demo

Prepare and launch before class:

```bash
source .venv/bin/activate
python scripts/prepare.py
python scripts/export_artifacts.py
streamlit run app.py
```

Open each lab once to warm the text/model caches. Turn on Presenter Mode. Keep `artifacts/failure_margins.html` and `artifacts/prompt_comparison.html` open as offline visual backups. The checkpoint-free synthetic lab is another fallback if inference is unavailable. The default photo is Chelsea the cat; the dog photograph is available from each image selector.

| Time | Lab and interaction | Narration / question |
|---|---|---|
| 0:00–0:40 | Introduce dual encoders | “What if the class weights came from words?” |
| 0:40–2:40 | Contrastive Learning → synthetic mode; alignment 0 then 0.65; change temperature; switch row/column probabilities; rotate captions | “Which cells are positives? Where does each probability distribution sum to one? Why did the loss stay the same after rotating captions and targets?” |
| 2:40–3:40 | Switch to Pretrained CLIP | “These cosines are now measured. A small demonstration batch is not the original training batch.” |
| 3:40–5:40 | Zero-Shot Classification → cat photo, broad classes; change chart to probabilities; remove cat; add coffee; restore candidates | “The image did not change. Why did the probabilities change? Text vectors act as classifier weights.” |
| 5:40–7:50 | Prompt Engineering → select dog photograph; inspect five phrasings; evaluate Prepared labeled photographs | “Compare the same images and labels. Normalize, average prompt embeddings, renormalize. An ensemble need not improve every example.” |
| 7:50–9:00 | Image Retrieval → “a bicycle”, then “a cup of coffee”; top K 6 | “The same alignment supports natural-language search with a matrix multiplication.” |
| 9:00–11:20 | Failure Laboratory → Counting, Spatial relationships, Attribute binding; inspect one positive and one negative margin if present | “A probe is a question, not a scripted failure. Correct minus distractor is the measured margin. These correlated drawings are not a general benchmark.” |
| 11:20–12:00 | Return to the shared-space idea | “A useful transferable representation does not guarantee calibrated probabilities or reliable compositional reasoning.” |

Every interaction uses prepared local assets or real local CLIP inference; there are no code edits, remote APIs, or asset downloads during the demonstration. If a model outcome differs on another device/version, discuss the measured result instead of promising an expected failure. If there is no negative-margin example in a probe, report that honestly.

## Optional 3D embedding interlude (2 minutes)

After the dual-encoder explanation, open **3D Embedding Space**. Keep the default 12 image–caption pairs, rotate the plot, and hover a circle and diamond. Toggle the pair connections, inspect one image, and compare its paired-caption cosine with its highest-ranked caption in the full 512D space. Add “a bicycle” as a query and inspect its closest image. Switch PCA to t-SNE to show that layouts differ even though the measured underlying vectors do not change. Emphasize that one shared projection is fitted to both modalities and that 3D distance can distort 512D similarity. Use `artifacts/embedding_space_3d.html` as an offline backup. For a strict 12-minute run, use this interlude in place of the retrieval segment and shorten the prompt comparison by one minute.

## Visual training/testing walkthrough (5–7 minutes)

Use **CLIP Process Walkthrough** to introduce the entire algorithm before the individual labs. Show both diagrams under Architecture overview and follow the two colored input streams. Training returns through the loss and update arrows; testing ends at the prediction/evaluation route.

Select Training cycle. The visible images identify four pairs; the toy actually consumes one-hot feature vectors. Use Next to walk through both linear maps, unit normalization, the similarity matrix, and the two probability directions. At Symmetric loss, point at the negative-log probability bars and the average of the two cross-entropies. At Backpropagation + update, inspect the actual gradient norms and next-step loss, then press Play learning cycle. The matrix and loss traces change together. Emphasize that the pretrained model is not being trained and the toy repeatedly sees the same batch.

Select Testing cycle. With the prepared cat photo and broad candidates, step through the actual crop/patch grid and BPE tokens, measured embeddings, similarities, learned-scale logits/softmax, and highest-ranked candidate. End with the labeled photograph confusion matrix. Discuss why evaluation uses labels but zero-shot classification does not need them, and why probabilities depend on the supplied vocabulary. The offline backup is `artifacts/clip_process.html`.

The embedding lab now includes a separate **Photographs at their embedding coordinates** plot below the scatter plot. Drag to rotate in 3D (Shift + drag pans); drag to pan in 2D. Scroll to zoom, reduce thumbnail size when photographs overlap, and click a photo or text diamond for its caption and coordinates. Download the thumbnail HTML or open `artifacts/embedding_thumbnails_3d.html`, `embedding_thumbnails_2d_pca.html`, or `embedding_thumbnails_2d_tsne.html` offline.
