# Exploratory scripts

These are **research / analysis scripts**, not part of the reproducible
pipeline. They run top-level code with hard-coded paths and assume intermediate
artifacts already exist. Read them for method detail, but run them by hand and
expect to edit paths first.

They import the shared `src/` modules (`acoustics`, `activations`, `decoding`,
`spec_idx`) plus the sibling helpers here (`viz_helpers`, `gammatone_helpers`),
so run them with both directories importable, e.g. from `src/`:

```bash
cd src
PYTHONPATH=. python exploratory/cosine_phone_talker_decoders.py
```

| Script | What it explores |
| --- | --- |
| `gammatone_kld.py` | Gammatone-baseline version of the speaker-specificity KLD metric (saved from the former `get_gammatone_and_decode.py`). |
| `distribution_visualization.py` | Cross-layer KLD "drift" distance per vowel, from per-layer LDA Gaussians. |
| `cosine_phone_talker_decoders.py` | Cosine geometry between talker vs. phone logistic-regression decoders. |
| `rotate_hidden_representation.py` | Representational-rotation analysis. |
| `FFN_and_forward_hook.py` | Decode talker from a phone-FFN's hidden activations via forward hooks. |
| `extract_vowel_formants.py` | Clip individual vowel tokens and extract their f1/f2/f3 formant tracks. |
| `select_words.py` | Context-size experiment: find monosyllabic words, cut single-word clips, and decode across frame-window sizes. |
| `viz_helpers.py` | Distribution/ellipsoid plotting helpers (moved out of the old helpers.py). |
| `gammatone_helpers.py` | Gammatone waveform/LDA helpers (needs `eelbrain`, conda-only; moved out of the old helpers.py). |

The speaker-specificity **index** itself (the KL ratio that is the project's
headline talker/phone measure) lives in `helpers.get_spec_idx_for_every_layer`
and is shipped as the tracked `results/results_spec_idx/` tables, plotted by
`docs/make_figure.py`.
