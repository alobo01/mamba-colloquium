"""Slide 14 · The bill, measured: Transformer ÷ Mamba memory and time as the context doubles.

Numbers: Koledoye et al. 2026 (arXiv 2601.01237), Table 1. 50M LLaMA vs 50M Mamba, one A100, FP32, batch 1.

Clicks:  ▶1 bars grow with N   ▶2 the Transformer runs out of memory   ▶3 why: N²/N = N
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

LENGTHS = ["512", "1k", "2k", "4k", "8k"]
MEMORY_RATIO = [1.35, 2.34, 5.21, 12.46]        # Transformer / Mamba, 8k = out of memory
TIME_RATIO = [2.01, 3.02, 5.47, 10.67]
XS = [-5.0, -3.1, -1.2, 0.7, 2.6]
BASE_Y, SCALE, BAR_W = -2.2, 0.33, 0.55


def build_bar(x: float, value: float, color, side: int) -> VGroup:
    """One bar (left or right of the group centre) with its value on top."""
    b = Rectangle(width=BAR_W, height=value * SCALE, stroke_width=0, fill_color=color, fill_opacity=0.9)
    b.move_to([x + side * (BAR_W / 2 + 0.05), BASE_Y + value * SCALE / 2, 0])
    return VGroup(b, T(f"{value:g}×", 15, color, bold=True).next_to(b, UP, buff=0.08))


def build_axis() -> VGroup:
    """Baseline with the context lengths underneath."""
    base = Line([XS[0] - 0.9, BASE_Y, 0], [XS[-1] + 0.9, BASE_Y, 0], color=FADE, stroke_width=2)
    labs = VGroup(*[T(n, 20, TOKEN).move_to([x, BASE_Y - 0.35, 0]) for n, x in zip(LENGTHS, XS)])
    return VGroup(base, labs, T("context length N", 18, FADE).next_to(labs, DOWN, buff=0.15))


def build_legend() -> VGroup:
    return VGroup(T("Transformer ÷ Mamba", 22, RESID, bold=True), pill("memory", ATTN, 18), pill("time", Q_COL, 18)) \
        .arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to(RIGHT * 5.4 + UP * 1.6)


def animate_bars(scene: Deck):
    """Memory and time ratios rise group by group."""
    for x, m, t in zip(XS, MEMORY_RATIO, TIME_RATIO):
        mb, tb = build_bar(x, m, ATTN, -1), build_bar(x, t, Q_COL, +1)
        scene.play(GrowFromEdge(mb[0], DOWN), GrowFromEdge(tb[0], DOWN), FadeIn(mb[1]), FadeIn(tb[1]), run_time=0.6)


def animate_oom(scene: Deck):
    """At 8k the Transformer does not fit: a dashed ghost bar and a red label."""
    x = XS[-1]
    ghost = DashedVMobject(Rectangle(width=2 * BAR_W + 0.1, height=4.3, stroke_color=PROB, stroke_width=3), num_dashes=40)
    ghost.move_to([x, BASE_Y + 2.15, 0])
    lab = T("out of\nmemory", 17, PROB, bold=True).move_to([x, BASE_Y + 2.15, 0])
    scene.play(Create(ghost), FadeIn(lab, scale=0.9))


def animate_why(scene: Deck):
    """Cost per token grows like N²/N = N, so the ratio roughly doubles when N doubles."""
    eq = ceq((r"\frac{N^2}{N}", ATTN), "=", ("N", MAMBA), size=48).move_to(RIGHT * 5.4 + DOWN * 0.5)
    note = T("double N → ratio ≈ doubles", 18, FADE).next_to(eq, DOWN, buff=0.3)
    scene.play(Write(eq), FadeIn(note))


class S14MeasuredCost(Deck):
    def construct(self):
        self.title("The bill, measured", ATTN)
        self.play(FadeIn(build_axis()), FadeIn(build_legend()))
        animate_bars(self)
        self.pause()
        animate_oom(self)
        self.pause()
        animate_why(self)
        self.play(FadeIn(T("Koledoye et al. 2026 · 50M LLaMA vs 50M Mamba · one A100", 15, FADE).move_to(DOWN * 3.35 + LEFT * 1.2)))
        self.pause(1.4)
        self.clear_all()
