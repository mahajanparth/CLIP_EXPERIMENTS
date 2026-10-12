# Measured demonstration artifacts

Original OpenAI CLIP **ViT-B/32**, upstream revision `a1d071733d7111c9c014f024669f959182114e33`.
Prepared at 2026-10-12T21:42:28.950022+00:00; device `cuda`; seed 2021.
Checkpoint SHA-256: `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
Learned logit_scale = 4.605170; multiplier = exp(logit_scale).

## Controlled probes

| Probe | N | Pairwise accuracy | Mean margin |
|---|---:|---:|---:|
| Attribute binding | 30 | 50.0% | -0.0005 |
| Counting | 30 | 73.3% | +0.0089 |
| Spatial relationships | 30 | 50.0% | +0.0002 |

Accuracy means correct cosine strictly exceeds the paired distractor cosine; ties count as unsuccessful. All 90 examples are controlled drawings. Both alternatives are counterbalanced in spatial and binding probes. Counting uses known counts of 2, 3, or 4. These correlated examples and repeated caption templates are demonstrations, not a representative benchmark. No outcomes are hardcoded.

## Prompt strategies

| Strategy | N | Accuracy |
|---|---:|---:|
| Raw class names | 17 | 100.0% |
| Single contextual prompt | 17 | 100.0% |
| Template ensemble | 17 | 100.0% |

All strategies use the same real photographs labeled dog, cat, car, and bicycle and the same four candidate labels. Template ensembling normalizes each prompt, averages prompt vectors per class, then renormalizes. Probabilities in prompt_results.csv use the actual checkpoint scale and depend on this candidate set.

## Files

- `image_embeddings.npz`: normalized gallery and probe embeddings, without pickle.
- `embedding_metadata.json`: model configuration, exact checkpoint and asset hashes, package versions.
- `failure_results.csv`, `failure_summary.csv`: per-example scores and aggregates.
- `prompt_results.csv`: per-image predictions, similarities, and candidate probabilities.
- `failure_margins.html`, `prompt_comparison.html`: interactive standalone Plotly figures; no CDN needed.
- `gallery_contact_sheet.jpg`: real photographic gallery preview; credits are in assets/PHOTO_CREDITS.md.
- `embedding_space_2d_pca.html`, `embedding_space_2d_tsne.html`: interactive 2D projections of the same measured image/text vectors, with real photo thumbnails. Matching PNG/SVG plots, coordinate CSVs, and embedding_projection_metadata.json are included. t-SNE is fitted independently in 2D, not sliced from the 3D layout.
- `embedding_space_3d.html`, `embedding_coordinates_3d.csv`: shared PCA projection of 12 measured image vectors and their 12 caption vectors. The three axes retain 65.8% of joint variance. Projection distances are lossy; cosine rankings use the original 512D space.

Regenerate with `python scripts/prepare.py` then `python scripts/export_artifacts.py`. No network is used when the checkpoint is already present.

## End-to-end visual process guide

`clip_process.html` embeds the training/testing architectures, an animated actual toy SGD trajectory, and measured frozen CLIP testing scores. `clip_training_architecture.svg` and `clip_testing_architecture.svg` are reusable vector diagrams. `toy_training_losses.csv` is explicitly synthetic toy-encoder optimization, not CLIP training history. `process_testing_results.csv` contains real local checkpoint scores for the default image and candidates.
