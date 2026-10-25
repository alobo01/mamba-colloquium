# The Three Brothers: a Manim talk on Transformers, Mamba, Mamba-2 and MoE

A 3Blue1Brown-style visual explanation, in 26 slides, for researchers who already know Transformers.

| File | What it is |
|---|---|
| `PRESENTATION_VISUAL_SCRIPT.md` | The visual script: each slide's signature frame, beats, math, Manim notes and the misconception to avoid |
| `slides/act*.py` | Manim implementation of every slide (`S01ColdOpen` … `S26Closing`) |
| `visual/common.py` | Palette, deck base classes, shared mobjects |
| `math_contracts/test_ssd_equivalence.py` | NumPy checks for every identity on screen (recurrent = matrix = chunked SSD, linear-attention case, 4×4 mask, block rank) |
| `agent_skill/SKILL.md` | Brief for coding agents (references the `edoardob90/manim-skills` Manim skill) |
| `reference/original_plan/` | The original research-audited storyboard and agent contract |

## Build

```bash
python -m venv .venv && . .venv/bin/activate
pip install "manim==0.19.0" manim-slides numpy pytest   # also needs FFmpeg, pango dev headers, LaTeX + dvisvgm
python -m pytest math_contracts
scripts/build_video.sh m        # 720p30 → media/final/three_brothers_720p30.mp4
```

For a live, click-stepped deck: `MANIM_DECK=slides manim-slides render slides/act1_attention.py S01ColdOpen …`, then `manim-slides present …`.
