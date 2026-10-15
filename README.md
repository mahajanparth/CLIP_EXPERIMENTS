# CLIP Explorer

An interactive companion to **Learning Transferable Visual Models From Natural Language Supervision** (Radford et al., ICML 2021). Eight local Streamlit labs connect the dual-encoder architecture to contrastive learning, zero-shot classification, prompt ensembling, retrieval, and controlled limitations probes.

## Start the prepared demo in this workspace

```bash
source .venv/bin/activate
streamlit run app.py
```

Open http://localhost:8501. The original ViT-B/32 checkpoint is stored locally in `.cache/clip/ViT-B-32.pt`; assets, normalized embeddings, and measured CSV reports have been prepared. Inference needs no internet. Enable **Presenter Mode** in the sidebar for larger text.

## Install on another machine

Python 3.11+ is required. A GPU is optional.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/prepare.py --download-model
python scripts/export_artifacts.py
streamlit run app.py
```

The first preparation downloads the official pretrained checkpoint (~338 MiB) through the upstream CLIP loader, which verifies its checksum. It then creates gallery/probe artwork, encodes every asset, and writes measured reports. Subsequent `python scripts/prepare.py` runs reuse local weights and perform no network access. The original OpenAI implementation is pinned to revision `a1d071733d7111c9c014f024669f959182114e33`; no training is performed.

All sample assets are included in this project. To regenerate only the deterministic original artwork, without model weights:

```bash
python scripts/prepare.py --assets-only
```

The main demos use 38 real photographs, including 34 curated Wikimedia images and four scikit-image photographs. All are included locally. To redownload the locked Wikimedia source list while online:

```bash
python scripts/download_real_gallery.py
python scripts/prepare.py
python scripts/export_artifacts.py
```

Photographs are preserved when regenerating the artwork. The four original scikit-image photographs can separately be redownloaded with `python scripts/download_photos.py`. Source selection is locked in `assets/photo_sources.json`; attribution and license links are in [assets/PHOTO_CREDITS.md](assets/PHOTO_CREDITS.md). Provenance, authors, licenses, and hashes are in [assets/PROVENANCE.md](assets/PROVENANCE.md) and [assets/manifest.json](assets/manifest.json). No assets are fetched during an app session.

## Labs

| Lab | Experiment | Prepared fallback |
|---|---|---|
| CLIP Process Walkthrough | Visual architecture, seven-stage toy training cycle with actual gradients/loss animation, seven-stage real testing cycle | Real paired photographs and deterministic toy features; local pretrained testing |
| Contrastive Learning | Raw cosines, row/column softmax, adjustable temperature, symmetric loss, rotated captions | Four/six matched photographs; clearly labeled synthetic unit-vector example |
| Zero-Shot Classification | Uploaded/local image, editable candidates/template, cosines vs learned-scale softmax | Broad classes, dog-breed preset, local gallery |
| Prompt Engineering | Phrase sensitivity; raw/contextual/normalized ensemble strategies | Same 17 labeled photographs and four labels |
| Image Activation Inspector | Actual model crop, patch features, 12 vision blocks, spatial maps, class readout, and image–prompt cosine | Frozen image forward pass; curated/50 × 50 collections and uploads |
| Text Activation Inspector | BPE tokens, lookup/position features, 12 contextual layers, EOS readout, feature trajectories and prompt comparison | Real frozen text forward pass; no image input required |
| Image Retrieval | Text-to-image and image-to-image ranking, adjustable K | 38 real photographs with recorded sources and licenses |
| Embedding Space · 2D / 3D | Joint 2D/3D PCA/t-SNE views of image/text vectors; matched-pair links, editable prompts, extra query point | All available categories, up to 50 real photographs/captions per category |
| Failure Laboratory | Counting, spatial relationships, attribute binding; paired margins and accuracy | 30 controlled labeled examples per experiment |

Every page includes a research question, an explanation, an interactive experiment, visualized results, mathematical notes, and a takeaway. Each page offers measured CSV downloads where applicable. The synthetic contrastive lab works without weights; inference pages give setup instructions if weights are missing and never substitute fake model predictions.

## Artifacts

- [Measured report](artifacts/REPORT.md): model identity, probe accuracy/margins, prompt strategy comparison, interpretation limits.
- `artifacts/failure_results.csv`, `failure_summary.csv`, `prompt_results.csv`: real pretrained predictions.
- `artifacts/failure_margins.html`, `prompt_comparison.html`: self-contained interactive figures, including Plotly locally for offline viewing.
- `artifacts/gallery_contact_sheet.jpg`: all 38 real photographic gallery images.
- `artifacts/embedding_space_2d_pca.html`, `embedding_space_2d_tsne.html`: offline 2D plots with real photographs, plus matching PNG/SVG plots and coordinate CSVs. t-SNE is independently fitted in 2D.
- `artifacts/embedding_thumbnails_3d.html`, `embedding_thumbnails_2d_pca.html`, `embedding_thumbnails_2d_tsne.html`: separate offline plots with real photographs centered at their projected coordinates. Rotate 3D, pan 2D, zoom, resize thumbnails, toggle text diamonds and pair links, and inspect captions. The same thumbnail views are included in the main embedding HTML files and the live lab.
- `artifacts/embedding_space_3d.html`, `embedding_coordinates_3d.csv`: standalone 3D image/text PCA view and coordinates. The lab also exports its current view, including custom prompts and queries.
- `artifacts/image_embeddings.npz`: normalized image vectors, with no pickle.
- `artifacts/embedding_metadata.json`: checkpoint/asset SHA-256 hashes, model revision, seed, device, preparation timestamp, package versions.
- [12-minute walkthrough](DEMO.md) and [45-minute presentation outline](PRESENTATION.md).

## Mathematical and experimental guarantees

For normalized image and text vectors, `S = V @ T.T` is cosine similarity. The zero-shot and prompt-report probabilities use `softmax(exp(model.logit_scale) * S)`, with the actual parameter from the checkpoint. These probabilities depend on the candidate descriptions; they are **not calibrated confidence**. The temperature slider deliberately overrides the scale in the contrastive teaching lab only.

The contrastive objective is `0.5 * (CE(logits, targets) + CE(logits.T, inverse_targets))`. Rotating captions also rotates the positives; the diagonal is positive only with aligned ordering. The column-normalized text-to-image display stays in image-row layout.

Prompt ensembling normalizes every prompt vector, averages vectors per class, and renormalizes each class mean. It does not average probabilities. Prompt accuracy is shown only when known labels belong to the selected candidate set. For uploaded images, the app reports scores/rankings without asserting ground truth.

Failure probes use measured cosine scores, `margin = correct − distractor`, and strictly positive-margin accuracy. Exact ties count as unsuccessful. The near-zero slider only controls interpretation; it does not redefine accuracy. These correlated drawings and repeated captions are small experimental probes, not representative benchmarks or guaranteed failures. Rendered geometry and domain shift can affect results.

## Visual end-to-end walkthrough

Open **CLIP Process Walkthrough**. The **Architecture overview** presents two dual-stream vector diagrams: training closes a loss/gradient/update loop, while testing uses frozen encoders and class prompts to produce a candidate ranking. Each diagram is downloadable as SVG. Dimension labels distinguish aligned training batches (N × N scores), inference batches (B × C scores), and the 4-pair toy (4 × 8 vectors). ViT-B/32 details show 3 × 224 × 224 image inputs, 50 visual tokens of width 768, 77 text tokens of width 512, and 512D joint embeddings. Equations are typeset into vector paths, including transposes, indices, fractions, both probability directions, losses, and updates. Live testing dimensions track the candidate count and evaluation batch. Matplotlib renders the SVG math locally; browser LaTeX services are not needed.

**Training cycle** offers seven visual stages: paired inputs, two encoders, unit normalization, a full matching matrix, row/column softmax, per-positive cross-entropy and symmetric loss, and backpropagation/updates. The inspection slider selects an optimizer step, and the animated matrix/loss plot has separate Play/Pause and step controls. Learning rate and initial temperature are adjustable. These are actual, deterministic SGD updates on two tiny 4→8 linear maps and a learned scale using synthetic one-hot input features. The real photographs illustrate pair identities; the toy does not consume pixels or BPE tokens. It never trains the original CLIP model and its loss curve is not CLIP training history. The toy reuses one batch and does not demonstrate generalization.

**Testing cycle** traces real original ViT-B/32 inference. Visual stages show the actual preprocessed 224px image overlaid with a 7×7 patch grid, original BPE token IDs, slices of measured 512D embeddings, cosine scores, the actual learned-scale logits and candidate softmax, prediction, and a confusion matrix on available labeled photographs. Encoders stay frozen throughout. Uploaded images and custom candidates are supported.

The standalone `artifacts/clip_process.html` embeds diagrams, paired photographs, the animated toy learning cycle, and measured testing charts for offline use. Additional artifacts include `clip_training_architecture.svg`, `clip_testing_architecture.svg`, `toy_training_losses.csv`, and `process_testing_results.csv`. Regenerate them with `python scripts/export_artifacts.py` or `python scripts/export_process_artifacts.py`.

## 2D and 3D embedding demonstrations

Open **Embedding Space · 2D / 3D** in the sidebar. The actual photographs are displayed above the plot. Image vectors appear as circles, text vectors as diamonds, with category colors and optional links between paired assets/captions. Choose **2D** or **3D** with Plot dimensions and **PCA** or **t-SNE** with Projection. Pan/rotate, zoom, hover, toggle legend groups, and inspect an image to see its original-space caption ranking. Select gallery categories, adjust the sample size, change the caption template, or add a custom text query.

PCA is the default: one centered, unwhitened projection is fitted jointly to image and text embeddings, and the retained variance is reported. Seeded t-SNE is an optional view with adjustable perplexity. Both project measured normalized 512D CLIP vectors into the selected two or three dimensions; neither preserves the full geometry. Rankings and cosine scores are computed in 512D, never from plotted distances. Adding points or changing captions refits the view. The t-SNE axes and distant-cluster separations have no direct semantic meaning.

Download the current coordinates as CSV or the current interactive plot as self-contained HTML, including the photographs, their credits, and a hover-linked image preview. `python scripts/export_artifacts.py` also regenerates the default offline PCA artifact using real local inference.

## Inspect text input features and contextual activations

Open **Text Activation Inspector** in the sidebar and enter a prompt. The inspector shows original BPE token IDs, learned token lookup and positional vectors, the token-plus-position input, outputs of each of the 12 residual Transformer blocks, final layer normalization, projected EOS, and the normalized CLIP vector. Choose a stage and feature window, follow a token/coordinate across depth, or compare each prompt's EOS trajectory. Full tensor shapes and readout equations are shown.

These are actual frozen model values captured under `torch.no_grad()`, with temporary observation hooks protected by the shared model lock. Hooks are always removed, including after errors, and never modify the forward outputs or weights. The final normalized vector is checked against the regular text-encoding interface. Activations are cached by prompt for classroom latency.

Feature magnitude is not attention, attribution, causal importance, or proof of a named semantic neuron. Input lookup vectors depend on token identity; contextual states depend on earlier tokens under CLIP's causal mask. EOS is the readout position. Post-EOS context positions are zero-filled with ID 0, not a dedicated PAD token; hide or show them explicitly.

Download the complete arrays as NPZ (no pickle) and token/prompt metadata as JSON/CSV. An offline stage viewer and default tensors are in `artifacts/text_activations.html`, `text_activations.npz`, and `text_activations_metadata.json`. Regenerate with:

```bash
python scripts/export_text_activations.py --prompt "a photo of a golden retriever"
```

## Tests

The normal suite runs on synthetic tensors and included assets and requires neither a GPU nor a checkpoint download:

```bash
python -m pytest -q
```

Tests cover normalization, cosine similarity, learned-scale logits, symmetric/permuted contrastive loss, temperature scaling, both probability axes, candidate dependence, normalized prompt ensembling, retrieval order/ties, margins, template validation, asset integrity, shared PCA geometry, seeded t-SNE, point metadata alignment, actual toy gradients and loss traces, frozen testing-cycle parameter checks, and the checkpoint-free UI.

After preparing weights, enable real inference and lab-rendering smoke tests:

```bash
RUN_CLIP_SMOKE=1 python -m pytest -q
# CPU-only real smoke test
CUDA_VISIBLE_DEVICES="" RUN_CLIP_SMOKE=1 python -m pytest -q -m model
```

These smoke tests never download weights themselves.

## Implementation

```text
app.py                         Streamlit entrypoint and five-page navigation
clip_explorer/config.py        Model/checkpoint/revision/seed/templates
clip_explorer/math_ops.py      Pure, testable tensor mathematics
clip_explorer/model.py         Frozen cached CLIP inference, original preprocessing/tokenizer
clip_explorer/data.py          Manifest assets and validated persisted embeddings
clip_explorer/ui.py             Shared presentation components
clip_explorer/labs.py           Five original experiment pages
clip_explorer/embedding_lab.py  Interactive 2D/3D image/text lab
clip_explorer/embedding_space.py Joint projections and 2D/3D figure construction
clip_explorer/process_lab.py    End-to-end visual training/testing walkthrough
clip_explorer/process_visuals.py SVG pipelines and animated matrix/loss plots
clip_explorer/training_demo.py  Isolated synthetic linear-encoder SGD demo
clip_explorer/text_activations.py Safe read-only capture of text encoder states
clip_explorer/activation_lab.py Token/feature activation visualization
scripts/generate_assets.py     Reproducible CC0 Pillow artwork
scripts/download_photos.py     Four original photographic supplements
scripts/download_real_gallery.py Curated, locked Wikimedia photo gallery
scripts/prepare.py             Checkpoint setup, embedding cache, measured reports
scripts/export_artifacts.py    Offline figures, contact sheet, report
scripts/export_process_artifacts.py Standalone visual process guide
```

Model loading uses `st.cache_resource` once per process. Encoding runs under `torch.no_grad()` with `model.eval()` and frozen parameters; embeddings are normalized float32 CPU tensors. Image/text inference is batched, protected by a shared lock, and content-cached with `st.cache_data`. Persisted image embeddings are checked against exact checkpoint, upstream revision, and current image hashes before reuse. Candidate text embeddings are cached after first use. Changing templates or uploading a new image requires genuine inference.

The seed is 2021. Artwork and tie-breaking are deterministic. Small numeric differences between CPU/GPU or software versions can occur; preparation records the environment. The tested dependency versions are recorded in `artifacts/embedding_metadata.json`.

## Troubleshooting

- **Missing checkpoint:** run `python scripts/prepare.py --download-model` while online. The app itself never downloads.
- **Wrong Python environment:** activate `.venv` and run `python -m streamlit run app.py`. `import clip` must resolve to the official OpenAI package, not another package with the same name.
- **CPU latency:** run preparation first; gallery image vectors then load from disk. The first text/model call is slower than later cached calls. Startup checks the checkpoint hash once per process.
- **CUDA out of memory / incompatible GPU:** launch with `CUDA_VISIBLE_DEVICES="" streamlit run app.py`. CPU execution is supported.
- **Torch/torchvision import errors:** install matching Torch and torchvision builds using the [official PyTorch installer](https://pytorch.org/get-started/locally/).
- **Long descriptions:** original CLIP has a 77-token context; shorten the text. The app rejects overlong input rather than silently truncating.
- **Invalid template:** use exactly one literal `{}` and no other braces.
- **Poor predictions:** inspect preprocessing (center crop), candidate set, domain shift, and prompt wording. No result is forced to match a pedagogical expectation.
- **Changed weights/assets:** rerun preparation and export. Hash validation prevents stale persisted image vectors from being used.
- **Port already in use:** add `--server.port 8502` to the launch command.

## Research sources

[Paper](https://arxiv.org/abs/2103.00020), especially Figure 3 and §3.1.4; [official OpenAI CLIP implementation](https://github.com/openai/CLIP). The pretrained ViT-B/32 demonstration should not be conflated with every reported paper result or its larger best-performing model. No original CLIP training data is redistributed here. Original drawing assets are dedicated under CC0-1.0; photographic permissions are recorded separately.

The embedding lab defaults to the **Caltech · 50 × 50** collection: exactly 2,500 distinct real images across 50 categories, with 50 in every category. The separate curated gallery retains 38 photographs across 7 broad categories. Select a collection, filter categories, and choose 1–50 photographs each. Large t-SNE views use Barnes–Hut approximation; smaller views use exact t-SNE. Offline exports use the larger collection when prepared. The thumbnail category filter hides groups without refitting or changing coordinates.


## Inspect image features and activations

Open **Image Activation Inspector**. Choose a collection, category, and photograph, or upload an image. Inspect the actual CLIP center crop with its 7 × 7 grid, patch projection (49 × 768), class-plus-patch residual states (50 × 768), all 12 vision blocks, final class layer normalization, and projected/normalized image vector (1 × 512). Heatmaps show selected features, spatial maps show patch RMS or one feature, and trajectories follow a class or patch token across depth.

Enter a text prompt to see the image–text cosine and each normalized coordinate's contribution to that sum. The prompt does not alter the image activations: the two encoders run independently. These maps display real feature values, not attention or saliency. Download full activation tensors, metadata, or a self-contained HTML snapshot. Default artifacts: `artifacts/image_activations.html`, `image_activations.npz`, and `image_activations_metadata.json`.

```bash
python scripts/export_image_activations.py
```

The larger dataset is a reproducible subset of [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02), credited to its dataset creators under the repository's CC BY 4.0 record. Original JPEG bytes, archive-member paths, and SHA-256 hashes are retained. Selection excludes categories with fewer than 50 files and listed nonphotographic/redundant groups, then takes the first 50 eligible categories and first 50 filenames each. See `assets/CALTECH50_CREDITS.md` and `assets/caltech50_manifest.json`; this is an educational subset, not a benchmark split.

```bash
python scripts/prepare_caltech50.py  # Official archive download, cached locally
python scripts/cache_collection.py  # Extend measured embeddings using local weights
python scripts/export_artifacts.py
```
