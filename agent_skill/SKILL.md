---
name: three-brothers-manim-talk
description: Build, fix or extend the "Three Brothers" Manim talk on Transformers, Mamba-1, Mamba-2, MoE and interpretability. Use when editing slides/sNN_*.py, visual/*.py, TALK_SCRIPT.md, or rendering the deck/video.
---

# Three Brothers: Manim production skill

## Read first

1. `TALK_SCRIPT.md`: what each slide shows and what the presenter says, click by click. Slides and script must stay in step.
2. `reference/original_plan/TRANSFORMERS_MAMBA2_MOE_MANIM_STORYBOARD.md` §4 (math contract) and §9 (source audit).
3. The third-party Manim agent skill [`edoardob90/manim-skills`](https://github.com/edoardob90/manim-skills) (`npx skills add edoardob90/manim-skills`): `manim/SKILL.md`, then `animations.md`, `mobjects.md`, `rendering.md`, `gotchas.md`. It targets ManimCE 0.20.1; this repo is pinned to 0.19.0, so check APIs against <https://docs.manim.community/en/stable/>.

## Environment

- Python venv with `manim==0.19.0`, `manim-slides`, `numpy`, `pytest`. System packages: FFmpeg, `libpango1.0-dev`, a LaTeX distribution with `dvisvgm`, the Inter font.
- `python -m pytest math_contracts` must pass before rendering slides 07, 17, 19 and 22. Those slides import the tested reference functions, so the numbers on screen are the tested numbers.

## Layout of the code

- **One slide = one file** `slides/sNN_name.py` with one scene class `SNNName(Deck)`. The module docstring names the slide, its sources and its click map (▶1, ▶2, …), which must match `TALK_SCRIPT.md`.
- Inside a slide file:
  - `build_*` functions return mobjects and never animate;
  - `animate_*(scene: Deck, ...)` functions play one beat and return whatever later beats need;
  - `construct()` is only a sequence of those calls separated by `self.pause()`.
- `visual/`:
  - `theme.py`: the palette;
  - `equations.py`: `ceq`, `morph`, `collapse`, `residual_eq`;
  - `shapes.py`;
  - `characters.py`;
  - `deck.py`: `Deck`, with `title()`, `pause()`, `clear_body()`, `clear_all()` and `check_layout()`.
- `Deck.pause()` is one presenter click. In video mode it is a short hold; with `MANIM_DECK=slides` it becomes `next_slide()`.
- Progress dots count the slide files automatically. Adding `s25_*.py` adds a dot.

## Equations

- Build every equation with `ceq(*parts)`, one symbol per part. The colour comes from the leading symbol:
  - q/Q/c cyan, k/K/b purple, v/V/x yellow;
  - S/h/s coral (state), a/L/γ/Δ mint (decay);
  - softmax and W blue.
  - Override a part with `(tex, colour)`.
- Derivations use two moves:
  - `morph(scene, src, dst, bring={j: mobject})` rewrites a line in place, and the substituted piece flies into slot `j`;
  - `collapse(scene, lines, result)` replaces all the lines with the boxed result.
  - Slides 07 and 19 are the reference examples.

## Commands

```bash
scripts/render.sh l slides/s07_ssd_unroll.py                                   # one slide, 480p15, logs in media/logs/
DECK_CHECK=1 scripts/render.sh l                                               # every slide + overlap/off-frame report in the logs
scripts/contact_sheet.sh media/videos/s07_ssd_unroll/480p15/S07SSDUnroll.mp4 /tmp/s07.png 12   # visual QA grid
scripts/build_video.sh m                                                       # all slides at 720p30 + concat → media/final/
MANIM_DECK=slides manim-slides render slides/s07_ssd_unroll.py S07SSDUnroll
```

## Production loop (mandatory)

1. Edit one slide, render it at `-ql`.
2. Build a contact sheet and **look at it**: check for clipped text, overlaps and labels off the frame edge.
3. Fix and render again. Only then run the full `build_video.sh`.
4. Parallel renders can race on Manim's caches; `render.sh` gives each job a private TeX dir and retries failures serially.

## Content rules (what must stay true)

- Cached decode is O(T) per token with O(T) cache memory, never O(1); the cache is not prefill.
- SSD: `S[t] = a[t]·S[t-1] + outer(k[t], v[t])`, `y[t] = S[t]ᵀ q[t]`, equal to `(L ⊙ QKᵀ)V`. No softmax in SSD.
- Mamba-1 training is a parallel scan; its fixed state limits recall but nothing is "forced to forget".
- FlashAttention is exact and IO-aware, not subquadratic. GQA still caches O(T) positions.
- Heads, experts and features have no guaranteed semantic labels.
- The Mamba state moves along time within a layer; the residual stream moves along depth.
- Lower bounds (slides 15–17) are about exact recall, selective copying and state tracking. They are not claims that Mamba is worse on average.
- Universality (slide 18) means approximation in the limit; it says nothing about size.
- The representation curves on slide 22 are schematic. The numbers on slides 14, 16 and 20 come from the cited papers.
