# The Three Brothers: a Manim talk on Transformers, Mamba, Mamba-2 and MoE

A short, visual explanation (16 slides, about 5 minutes as a video, about 20 minutes as a live talk) for researchers who already know Transformers.

| File | What it is |
|---|---|
| `TALK_SCRIPT.md` | What to say on every slide, click by click, plus a formula cheat sheet, likely questions and sources |
| `slides/part1_attention.py` … `part4_theory.py` | The Manim slides, one scene class per slide (`S01Intro` … `S16Interp`) |
| `visual/common.py` | Look and feel: backdrop, palette, glow, cards, the three brothers |
| `math_contracts/test_ssd_equivalence.py` | NumPy checks that the Mamba-2 formulas on screen agree (recurrence = matrix form = chunked algorithm) |
| `agent_skill/SKILL.md` | Brief for coding agents (references the `edoardob90/manim-skills` Manim skill) |
| `reference/original_plan/` | The original storyboard and agent contract this talk grew out of |

## Build

```bash
python -m venv .venv && . .venv/bin/activate
pip install "manim==0.19.0" manim-slides numpy pytest   # also needs FFmpeg, pango dev headers, LaTeX + dvisvgm, the Inter font
python -m pytest math_contracts
scripts/build_video.sh m        # 720p30 → media/final/three_brothers_720p30.mp4
```

For a live, click-stepped deck (each click matches a ▶ in `TALK_SCRIPT.md`):

```bash
MANIM_DECK=slides manim-slides render slides/part1_attention.py S01Intro S02QKV S03KVCache S04Bank
manim-slides present S01Intro S02QKV S03KVCache S04Bank
```
