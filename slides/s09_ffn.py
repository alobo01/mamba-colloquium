"""Slide 09 · Why a feed-forward network, and why it is the expensive part.

Clicks:  ▶1 attention mixes tokens   ▶2 FFN works per token   ▶3 parameter split + two memories
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

NAMES = ["Leo", "hid", "the", "key", "under"]
LINKS = [(0, 3), (1, 3), (2, 3), (3, 4), (0, 4)]


def build_tokens() -> VGroup:
    return VGroup(*[chip(n, TOKEN, 24) for n in NAMES]).arrange(RIGHT, buff=0.7).move_to(UP * 1.5)


def animate_mixing(scene: Deck, toks: VGroup) -> VGroup:
    """Arcs between tokens: attention moves information sideways. Returns arcs + label."""
    arcs = VGroup(*[ArcBetweenPoints(toks[i].get_top(), toks[j].get_top(), angle=-0.9, stroke_color=ATTN, stroke_width=4)
                    for i, j in LINKS])
    lab = pill("attention: tokens talk to each other", ATTN, 22).move_to(UP * 0.25)
    scene.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), FadeIn(lab), run_time=1.6)
    return VGroup(arcs, lab)


def animate_per_token(scene: Deck, toks: VGroup) -> VGroup:
    """An hourglass under every token (expand, squeeze); hidden units flash."""
    hours = VGroup(*[hourglass(t.get_bottom() + DOWN * 0.15, MOE) for t in toks])
    lab = pill("FFN: each token thinks on its own", MOE, 22).move_to(DOWN * 2.3)
    scene.play(LaggedStart(*[FadeIn(h, shift=DOWN * 0.2) for h in hours], lag_ratio=0.12), FadeIn(lab))
    scene.play(LaggedStart(*[Flash(h[1][k], color=MOE, flash_radius=0.18, line_length=0.08) for h in hours for k in (1, 2)],
                           lag_ratio=0.03), run_time=1.2)
    return VGroup(hours, lab)


def animate_cost_split(scene: Deck):
    """Attention 4d² vs FFN 8d²: about two thirds of a layer; two kinds of memory."""
    total = 10.8
    bar_a = Rectangle(width=total / 3, height=1.0, stroke_width=0, fill_color=ATTN, fill_opacity=0.9)
    bar_f = Rectangle(width=total * 2 / 3, height=1.0, stroke_width=0, fill_color=MOE, fill_opacity=0.9)
    VGroup(bar_a, bar_f).arrange(RIGHT, buff=0.04).move_to(UP * 0.2)
    scene.play(GrowFromEdge(bar_a, LEFT), GrowFromEdge(bar_f, LEFT), run_time=1.1)
    scene.play(FadeIn(M(r"\text{attention}\ 4d^2", size=34, color=BG1).move_to(bar_a)),
               FadeIn(M(r"\text{FFN}\ 8d^2", size=40, color=BG1).move_to(bar_f)))
    br = Brace(bar_f, DOWN, color=MOE)
    scene.play(GrowFromCenter(br), Write(M(r"\approx \tfrac{2}{3}\ \text{of each layer}", size=40, color=MOE).next_to(br, DOWN, buff=0.15)))
    scene.play(FadeIn(pill("memory of this context", ATTN, 20).move_to(LEFT * 3.6 + DOWN * 2.6), shift=UP * 0.2),
               FadeIn(pill("memory of the training data", MOE, 20).move_to(RIGHT * 3.0 + DOWN * 2.6), shift=UP * 0.2))


class S09FFN(Deck):
    def construct(self):
        self.title("Why a feed-forward network?", MOE)
        toks = build_tokens()
        self.play(LaggedStart(*[FadeIn(t, shift=DOWN * 0.2) for t in toks], lag_ratio=0.1))
        mix = animate_mixing(self, toks)
        self.pause()
        self.play(FadeOut(mix))
        per = animate_per_token(self, toks)
        self.pause()
        self.play(FadeOut(VGroup(toks, per)))
        animate_cost_split(self)
        self.pause(1.4)
        self.clear_all()
