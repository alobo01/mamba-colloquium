"""Slide 11 · How modern LLMs run MoE: shared + fine-grained experts, load balancing, expert parallelism.

Clicks:  ▶1 many small + 1 shared   ▶2 bias nudges the load even   ▶3 experts across GPUs, all-to-all
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

XS = [-4.75, 0, 4.75]
HEADINGS = ["many small + 1 shared", "keep the load even", "experts across GPUs"]


def caption(text: str, x: float) -> Text:
    """Small grey note at the foot of a panel."""
    return T(text, 17, FADE).move_to([x, -2.55, 0])


def animate_fine_grained(scene: Deck, x: float):
    """One always-on shared expert above a 4×4 grid of small experts; 8 light up."""
    shared = card(1.2, 1.0, V_COL, 0.5).move_to([x, 0.6, 0])
    small = VGroup(*[RoundedRectangle(width=0.5, height=0.5, corner_radius=0.08, stroke_color=c, stroke_width=2, fill_color=c,
                                      fill_opacity=0.12) for c in ROOM * 2]).arrange_in_grid(4, 4, buff=0.12).move_to([x, -1.0, 0])
    scene.play(FadeIn(shared), FadeIn(T("shared", 16, BG1, bold=True).move_to(shared)), FadeIn(small, lag_ratio=0.03))
    scene.play(LaggedStart(*[small[i].animate.set_fill(ROOM[i % 8], 0.8) for i in (1, 5, 6, 10, 12, 14, 3, 9)], lag_ratio=0.08), run_time=1.3)
    scene.play(FadeIn(caption("DeepSeek-V3: 256 + 1, top-8", x)))


def animate_balancing(scene: Deck, x: float):
    """Skewed expert loads (overloaded in red) relax to an even load."""
    skew = np.array([0.95, 0.3, 0.8, 0.15, 0.7, 0.2, 0.55, 0.25])
    hist = VGroup(*[Rectangle(width=0.32, height=2.3 * v, stroke_width=0, fill_color=PROB if v > 0.65 else ROOM[i], fill_opacity=0.9)
                    for i, v in enumerate(skew)]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN).move_to([x, -0.2, 0])
    base = Line(hist.get_corner(DL) + LEFT * 0.1, hist.get_corner(DR) + RIGHT * 0.1, color=FADE, stroke_width=2)
    scene.play(FadeIn(base), LaggedStart(*[GrowFromEdge(b, DOWN) for b in hist], lag_ratio=0.08), run_time=1.2)
    even = VGroup(*[Rectangle(width=0.32, height=1.15, stroke_width=0, fill_color=ROOM[i], fill_opacity=0.9)
                    for i in range(8)]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN).move_to(hist, aligned_edge=DOWN)
    scene.play(Transform(hist, even), FadeIn(caption("nudge the router's bias", x)), run_time=1.6)


def animate_expert_parallel(scene: Deck, x: float):
    """Four GPUs with two experts each; tokens travel along an all-to-all mesh."""
    gpus = VGroup(*[card(1.55, 1.05, MOE, 0.1, radius=0.12, sw=2) for _ in range(4)]).arrange_in_grid(2, 2, buff=0.9).move_to([x, 0.0, 0])
    for k, g in enumerate(gpus):
        g.add(VGroup(*[Square(0.3, stroke_color=ROOM[2 * k + j], stroke_width=2, fill_color=ROOM[2 * k + j], fill_opacity=0.5) for j in (0, 1)])
              .arrange(RIGHT, buff=0.12).move_to(g))
    mesh = VGroup(*[Line(gpus[i].get_center(), gpus[j].get_center(), color=MOE, stroke_width=1.5, stroke_opacity=0.55)
                    for i in range(4) for j in range(i + 1, 4)])
    scene.play(FadeIn(gpus, lag_ratio=0.15), Create(mesh))
    dots = VGroup(*[Dot(radius=0.07, color=RESID).move_to(l.get_start()) for l in mesh])
    scene.add(dots)
    scene.play(*[d.animate.move_to(l.get_end()) for d, l in zip(dots, mesh)], run_time=1.2, rate_func=there_and_back_with_pause)
    scene.play(FadeIn(caption("tokens travel: all-to-all", x)))


class S11MoEProduction(Deck):
    def construct(self):
        self.title("How modern LLMs run MoE", MOE)
        panels = VGroup(*[card(4.4, 5.3, MOE, 0.06).move_to([x, -0.35, 0]) for x in XS])
        heads = VGroup(*[T(t, 24, MOE, bold=True).move_to([x, 1.85, 0]) for x, t in zip(XS, HEADINGS)])
        self.play(FadeIn(panels, lag_ratio=0.2), FadeIn(heads, lag_ratio=0.2))
        animate_fine_grained(self, XS[0])
        self.pause()
        animate_balancing(self, XS[1])
        self.pause()
        animate_expert_parallel(self, XS[2])
        self.pause(1.4)
        self.clear_all()
