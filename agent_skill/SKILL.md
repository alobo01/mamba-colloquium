---
name: three-brothers-manim-talk
description: Build, fix or extend the "Three Brothers" Manim talk on Transformers, Mamba-1, Mamba-2, MoE and interpretability. Use when editing slides/*.py, visual/common.py, or rendering the deck/video.
---

# Three Brothers — Manim production skill

## Read first

1. `PRESENTATION_VISUAL_SCRIPT.md`: the slide-by-slide visual script (signature frame, beats, math, Manim notes, "Don't say" box). This is the spec. Code must match it.
2. `reference/original_plan/TRANSFORMERS_MAMBA2_MOE_MANIM_STORYBOARD.md` §4 (math contract) and §9 (source audit).
3. The third-party Manim agent skill [`edoardob90/manim-skills`](https://github.com/edoardob90/manim-skills) (`npx skills add edoardob90/manim-skills`). Start with `manim/SKILL.md`, then `animations.md`, `mobjects.md`, `rendering.md` and `gotchas.md`. It targets ManimCE 0.20.1; this repo is pinned to 0.19.0, so check APIs against <https://docs.manim.community/en/stable/>.

## Environment

- Python venv with `manim==0.19.0`, `manim-slides`, `numpy` and `pytest`. System packages: FFmpeg, `libpango1.0-dev`, a LaTeX distribution with `dvisvgm` (`texlive-latex-extra`, `texlive-science`).
- `python -m pytest math_contracts` must pass before rendering any SSD slide. Slides import the tested reference functions, so the numbers on screen are the tested numbers.

## Layout of the code

- `visual/common.py`: palette (Q cyan, K purple, V yellow, residual white, attention blue, Mamba coral, Mamba-2 mint, MoE gold), `Deck` / `Deck3D` base classes, and shared mobjects (`token_card`, `brother`, `heatmap`, `evidence_badge`, …).
- `slides/act1_attention.py` … `slides/act5_theory.py`: one class per slide, `S01ColdOpen` … `S26Closing`. Every scene builds its own objects; there is no state shared across scenes.
- `Deck.pause()` is one presenter click. In video mode it is a short hold; with `MANIM_DECK=slides` it becomes `next_slide()`.

## Commands

```bash
scripts/render.sh l slides/act3_ssd.py S14MatrixForm       # one slide, 480p15, logs in media/logs/
scripts/contact_sheet.sh media/videos/act3_ssd/480p15/S14MatrixForm.mp4 /tmp/s14.png 12   # visual QA grid
scripts/build_video.sh m                                    # all slides at 720p30 + concat → media/final/
MANIM_DECK=slides manim-slides render slides/act1_attention.py S01ColdOpen S02QKVLibrary  # live deck
manim-slides present S01ColdOpen S02QKVLibrary
```

## Production loop (mandatory)

1. Edit one slide, then render it at `-ql`.
2. Build a contact sheet and **look at it**: check for clipped text, overlaps with the header or badge, and labels off the frame edge.
3. Fix it and render again. Only then do the full `build_video.sh` pass.
4. Parallel renders can race on Manim's caches. `render.sh` gives each job a private TeX dir and retries failures serially.

## Guardrails (never violate)

- The KV cache is not prefill, and cached decode is O(T) per token, never O(1).
- SSD: `S[t] = a[t]·S[t-1] + outer(k[t], v[t])`, `y[t] = S[t]ᵀ q[t]`, which equals `(L ⊙ QKᵀ)V`. **No softmax** anywhere in SSD.
- Mamba-1 training is a *parallel* scan. A fixed state is not "forced to forget".
- FlashAttention is exact and IO-aware, **not** subquadratic. GQA still caches O(T) positions.
- Heads, experts and features never get guaranteed semantic labels. Toy numbers carry a `SCHEMATIC` badge, and interpretability results stay `HYPOTHESIS` unless real logs are imported.
- The Mamba state moves across *time* within a layer, and the residual moves across *depth*. Draw them as two different arrows.
- Model-card numbers (MoE expert counts, hybrid layer schedules) are marked "verify" on screen.
