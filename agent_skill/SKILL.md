---
name: three-brothers-manim-talk
description: Build, fix or extend the "Three Brothers" Manim talk on Transformers, Mamba-1, Mamba-2, MoE and interpretability. Use when editing slides/*.py, visual/common.py, TALK_SCRIPT.md, or rendering the deck/video.
---

# Three Brothers: Manim production skill

## Read first

1. `TALK_SCRIPT.md`: what each slide shows and what the presenter says, click by click. Slides and script must stay in step.
2. `reference/original_plan/TRANSFORMERS_MAMBA2_MOE_MANIM_STORYBOARD.md` §4 (math contract) and §9 (source audit).
3. The third-party Manim agent skill [`edoardob90/manim-skills`](https://github.com/edoardob90/manim-skills) (`npx skills add edoardob90/manim-skills`): `manim/SKILL.md`, then `animations.md`, `mobjects.md`, `rendering.md`, `gotchas.md`. It targets ManimCE 0.20.1; this repo is pinned to 0.19.0, so check APIs against <https://docs.manim.community/en/stable/>.

## Environment

- Python venv with `manim==0.19.0`, `manim-slides`, `numpy`, `pytest`. System packages: FFmpeg, `libpango1.0-dev`, a LaTeX distribution with `dvisvgm`, the Inter font.
- `python -m pytest math_contracts` must pass before rendering S06–S08 or S15. Those slides import the tested reference functions, so the numbers on screen are the tested numbers.

## Layout of the code

- `visual/common.py`: palette (Q cyan, K purple, V yellow, residual white, attention blue, Mamba coral, Mamba-2 mint, MoE gold), the `Deck` base class (backdrop, progress dots, `title()`, `pause()`, `clear_all()`), and shared mobjects (`brother`, `card`, `pill`, `chip`, `token_card`, `heat`, `glow`, `arrow`).
- `slides/part1_attention.py` … `part4_theory.py`: one class per slide, `S01Intro` … `S16Interp`. Each scene builds its own objects; nothing is shared between scenes.
- `Deck.pause()` is one presenter click. In video mode it is a short hold; with `MANIM_DECK=slides` it becomes `next_slide()`.

## Style rules

- No captions, subtitles, badges or warning boxes on screen. Slides carry labels and equations only; explanations live in `TALK_SCRIPT.md`.
- One idea per slide, at most one primary equation on screen at a time.
- Keep the slide count and runtime small (16 slides, about 20 s each); cut before adding.

## Commands

```bash
scripts/render.sh l slides/part2_mamba.py S07SSDMatrix                        # one slide, 480p15, logs in media/logs/
scripts/contact_sheet.sh media/videos/part2_mamba/480p15/S07SSDMatrix.mp4 /tmp/s07.png 12   # visual QA grid
scripts/build_video.sh m                                                       # all slides at 720p30 + concat → media/final/
MANIM_DECK=slides manim-slides render slides/part1_attention.py S01Intro S02QKV
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
