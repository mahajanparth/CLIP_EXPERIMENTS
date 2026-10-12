# Asset provenance and usage

The main demonstration gallery contains 38 real photographs: 34 curated Wikimedia photographs and four scikit-image photographs. The original 60 illustrations remain archived as generator assets and are excluded from the main demo selectors, retrieval, projections, and labeled comparisons. The controlled-probe set contains 90 original drawings. `manifest.json` records each file's SHA-256, label or caption alternatives, origin, medium, and license.

## Original drawings

Generated with Pillow by `scripts/generate_assets.py`, seed 2021. No third-party artwork, model-generated image service, or external dataset is used for these drawings. The original artwork is dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The images contain no written labels that could leak caption answers through OCR.

The gallery has six categories with ten drawings each: animals, vehicles, landscapes, food, robotics, and household objects. Labels describe the intended icon; these are stylized illustrations, not photographs. Some objects share geometry. Labels and correlations make this a teaching set, not a realistic recognition benchmark.

Counting has ten drawings for each count in {2,3,4}; distractors alternate between the two other counts. Apples are separated and all appear within the central image region. Spatial relations have 15 matched left/right pairs. Attribute binding has 15 matched color-swap pairs and counterbalanced left/right positions. Those two sets vary five vertical offsets and three background colors. The two members of each matched pair keep nuisance variables fixed. These repeated templates/geometry create correlated examples; no statistical population claim is supported by the sample size.

## Curated Wikimedia photographs

The main gallery includes dogs, cats, cars, bicycles, coffee, pizza, apples, mountains, forests, beaches, chairs, clocks, and industrial robots. Source images were visually reviewed to remove irrelevant search matches, artwork, and mislabeled subjects. Thumbnails preserve the photographic content; some contain people or contextual objects in addition to the named subject. Labels are descriptive teaching annotations, not independently audited benchmark labels.

The exact selected sources, original file pages, photographer credits, and permissions are locked in `photo_sources.json`. See [PHOTO_CREDITS.md](PHOTO_CREDITS.md) for the complete attribution list. Commons CC BY, CC BY-SA, CC0, and public-domain permissions are recorded per file; follow the respective source license when redistributing individual photographs. `python scripts/download_real_gallery.py` reuses the locked source URLs. Assets are prepared before class and no live internet is needed by the app.

## Original scikit-image photographs

Upstream files are taken without modification from scikit-image v0.19.3. Permission evidence is in its [data fetcher documentation](https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/_fetchers.py).

| File | Subject | Credit | Permission |
|---|---|---|---|
| gallery/chelsea.png | Chelsea the cat | Stefan van der Walt | CC0-1.0 |
| gallery/coffee.png | Coffee cup | Rachel Michetti; courtesy of Pikolo Espresso Bar | CC0-1.0 |
| gallery/astronaut.png | Astronaut Eileen Collins | NASA | Public domain |
| gallery/rocket.jpg | Falcon 9 / DSCOVR launch | SpaceX | Public domain |

Source URLs and file hashes are recorded per image. `python scripts/download_photos.py` optionally fetches these four files while online. No live downloads occur during a presentation. User uploads remain in the local app session/cache and are not sent to an external inference service.
