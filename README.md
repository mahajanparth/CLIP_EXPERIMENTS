# CLIP Explorer

[![Spinning 3D PCA of CLIP image and text embeddings](artifacts/demo_pca_spin.gif)](artifacts/demo_pca_spin.mp4)

*Real CLIP embeddings for Caltech-101 photos and their class captions, projected to 3D with PCA. Recorded from the Embedding Explorer in the web version. Click for the MP4.*

A set of interactive Streamlit labs for exploring the CLIP paper, [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021). It runs the original OpenAI ViT-B/32 model locally and lets you poke at contrastive learning, zero-shot classification, prompt engineering, retrieval, embeddings, activations, and some of CLIP's known weak spots.

![CLIP process walkthrough](artifacts/demo_process_overview.png)

## Quick start

```bash
source .venv/bin/activate
streamlit run app.py
```

Then open http://localhost:8501. If the model weights and embeddings are already prepared, nothing needs the internet. Turn on **Presenter Mode** in the sidebar for bigger text.

## Setup on a new machine

Needs Python 3.11+. A GPU helps but isn't required.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/prepare.py --download-model
python scripts/export_artifacts.py
streamlit run app.py
```

`prepare.py --download-model` grabs the official checkpoint (about 338 MiB) through the OpenAI CLIP loader, then encodes all the images and writes the reports. After that, runs are fully offline. The CLIP repo is pinned to commit `a1d071733d7111c9c014f024669f959182114e33`. Nothing here trains CLIP.

## Web version

There's also a browser-only version of the project, built as a Claude artifact: [How CLIP Works](https://claude.ai/artifact/7JbXUPpy1CNJeg2tU9RNUP). It has an animated nine-scene walkthrough, the embedding explorer shown above, and pages for blind spots, prompt wording, and zero-shot classification. CLIP runs in the browser, so you can try your own images and captions.

![Animated walkthrough](artifacts/web_walkthrough.png)
![CLIP's blind spots](artifacts/web_failures.png)
![Prompt wording](artifacts/web_prompts.png)
![Zero-shot classifier](artifacts/web_zeroshot.png)

## Screenshots of the Streamlit app

### Zero-shot classification
Type in candidate classes and a prompt template, and CLIP ranks them against an image.

![Zero-shot classification](artifacts/demo_zero_shot.png)

### Contrastive learning
The similarity matrix, row and column softmax, temperature, and the symmetric loss.

![Contrastive learning](artifacts/demo_contrastive.png)

### Training and testing walkthrough

![Toy training cycle](artifacts/demo_process_training.png)
![Toy training losses](artifacts/demo_process_losses.png)
![Testing cycle](artifacts/demo_process_testing.png)

### Prompt engineering

![Prompt engineering](artifacts/demo_prompts.png)

### Image retrieval

![Image retrieval](artifacts/demo_retrieval.png)

### Embedding space (2D and 3D)

![3D embeddings](artifacts/demo_embeddings_3d.png)
![2D PCA](artifacts/demo_embedding_2d_pca.png)
![2D t-SNE](artifacts/demo_embedding_2d_tsne.png)
![Offline embedding view with photos](artifacts/demo_embedding_photos_offline.png)

### Activation inspectors

![Image activations](artifacts/demo_image_activations.png)
![Text activations](artifacts/demo_text_activations.png)

### Failure lab
Counting, spatial relations, and attribute binding probes.

![Failure lab](artifacts/demo_failures.png)

## Labs

| Lab | What you can do |
|---|---|
| CLIP Process Walkthrough | Architecture diagrams, a small toy training loop with real gradients, and a real inference trace |
| Contrastive Learning | Play with cosine scores, softmax direction, temperature, and caption order |
| Zero-Shot Classification | Upload or pick an image, edit candidate classes and the prompt template |
| Prompt Engineering | Compare prompt wordings and a few ensembling strategies |
| Image Activation Inspector | Look at patch features, all 12 vision blocks, and image/prompt cosine |
| Text Activation Inspector | Look at BPE tokens, 12 text layers, and the EOS readout |
| Image Retrieval | Text-to-image and image-to-image search |
| Embedding Space 2D / 3D | PCA or t-SNE views of image and text vectors together |
| Failure Laboratory | 30 controlled examples each for counting, spatial relations, and attribute binding |

Most pages let you download the measured results as CSV.

## Data

- **Curated gallery:** 38 real photos (34 from Wikimedia, 4 from scikit-image). Credits and licenses are in [assets/PHOTO_CREDITS.md](assets/PHOTO_CREDITS.md) and [assets/PROVENANCE.md](assets/PROVENANCE.md).
- **Caltech 50 x 50:** 2,500 images from [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02), 50 categories with 50 images each. This is the default in the embedding lab. See `assets/CALTECH50_CREDITS.md`.
- **Probe drawings:** generated with Pillow and released under CC0.

To redownload or rebuild:

```bash
python scripts/download_real_gallery.py   # Wikimedia gallery
python scripts/download_photos.py         # scikit-image photos
python scripts/prepare_caltech50.py       # Caltech subset
python scripts/cache_collection.py        # embed the Caltech subset
python scripts/prepare.py --assets-only   # just the probe drawings
python scripts/export_artifacts.py
```

## Artifacts

Everything in `artifacts/` works offline:

- [REPORT.md](artifacts/REPORT.md): model info, probe accuracy, prompt comparisons
- CSVs of real model predictions (`failure_results.csv`, `prompt_results.csv`, etc.)
- Standalone HTML plots for embeddings, prompts, failure margins, and the process walkthrough
- `text_activations.*` and `image_activations.*` for the activation inspectors
- `image_embeddings.npz` and `embedding_metadata.json` with hashes and versions

There's also a [12-minute demo script](DEMO.md) and a [45-minute presentation outline](PRESENTATION.md).

## A few notes on the math

- Image and text vectors are normalized, so `V @ T.T` is cosine similarity.
- Zero-shot probabilities use `softmax(exp(logit_scale) * S)` with the checkpoint's real scale. They depend on which candidates you give it, so they aren't calibrated confidence.
- The contrastive loss is `0.5 * (CE(logits, targets) + CE(logits.T, targets))`.
- Prompt ensembling averages normalized text vectors per class and renormalizes. It doesn't average probabilities.
- Failure probes report `margin = correct - distractor`. Ties count as wrong. These are small probes, not benchmarks.
- PCA and t-SNE are just for looking. All rankings and scores are computed in the full 512D space.
- The toy training demo is two tiny linear layers on synthetic features. It is not CLIP's real training curve.
- Activation views show raw feature values, not attention or saliency.

## Tests

```bash
python -m pytest -q
```

This runs without a GPU or the checkpoint. With weights prepared, you can also run the real-model smoke tests:

```bash
RUN_CLIP_SMOKE=1 python -m pytest -q
CUDA_VISIBLE_DEVICES="" RUN_CLIP_SMOKE=1 python -m pytest -q -m model   # CPU only
```

## Project layout

```text
app.py                          Streamlit entry point
clip_explorer/config.py         Model, paths, seed, templates
clip_explorer/math_ops.py       Core tensor math
clip_explorer/model.py          Cached CLIP loading and inference
clip_explorer/data.py           Assets and saved embeddings
clip_explorer/labs.py           Main lab pages
clip_explorer/process_lab.py    Training/testing walkthrough
clip_explorer/embedding_lab.py  2D/3D embedding lab
clip_explorer/activation_lab.py Activation views
scripts/                        Download, prepare, and export scripts
tests/                          Test suite
```

## Troubleshooting

- **Missing checkpoint:** run `python scripts/prepare.py --download-model` while online.
- **Wrong environment:** activate `.venv` and use `python -m streamlit run app.py`. Make sure `import clip` is the OpenAI package.
- **Out of GPU memory:** `CUDA_VISIBLE_DEVICES="" streamlit run app.py` runs on CPU.
- **Torch import errors:** reinstall matching torch and torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).
- **Text too long:** CLIP has a 77-token limit. The app rejects longer input instead of cutting it off.
- **Bad template:** it needs exactly one `{}`.
- **Port in use:** add `--server.port 8502`.

## References

- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP code](https://github.com/openai/CLIP)

This uses the ViT-B/32 model, which is smaller than the best model in the paper, so results won't match every number reported there.
