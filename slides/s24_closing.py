"""Slide 24 · Closing: three memories merge into one residual vector.

Clicks:  ▶1 the three paths and the final line
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

CAST = [("Theo", ATTN, "remembers everything, pays per token"), ("Manu", MAMBA, "fixed notebook, chooses what to keep"),
        ("Max", MAMBA2, "same notebook, matmul-fast")]


def animate_merge(scene: Deck):
    """Three coloured paths flow into one thick white residual arrow."""
    starts = [LEFT * 5.6 + UP * 1.5, LEFT * 5.6, LEFT * 5.6 + DOWN * 1.5]
    end = RIGHT * 1.6
    paths = VGroup(*[CubicBezier(s + RIGHT * 0.2, s + RIGHT * 3, end + LEFT * 2.2, end, stroke_color=c, stroke_width=7)
                     for s, (_, c, _) in zip(starts, CAST)])
    names = VGroup(*[VGroup(T(n, 24, c, bold=True), T(d, 15, FADE)).arrange(RIGHT, buff=0.25).next_to(s, UP, buff=0.12, aligned_edge=LEFT)
                     for s, (n, c, d) in zip(starts, CAST)])
    scene.play(FadeIn(names), LaggedStart(*[Create(p) for p in paths], lag_ratio=0.2), run_time=1.8)
    scene.play(GrowArrow(arrow(end, end + RIGHT * 3.6, RESID, 12, 0.35)), FadeIn(glow(Line(end, end + RIGHT * 3.4), RESID, 3)))


class S24Closing(Deck):
    def construct(self):
        animate_merge(self)
        self.play(Write(T("Different memories.  Often similar computations.", 38, RESID, bold=True).to_edge(DOWN, buff=0.9)), run_time=1.6)
        self.pause(2.0)
        self.clear_all()
