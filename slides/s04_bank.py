"""Slide 04 · "bank": the same embedding drifts to two meanings in the residual stream.

Clicks:  ▶1 two sentences, same starting vector   ▶2 each layer adds a delta → river vs money, update rule
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

LEFT_X, RIGHT_X = -3.55, 3.55
DELTAS_RIVER = [(UP * 0.5 + RIGHT * 0.4, ATTN), (UP * 0.15 + RIGHT * 0.55, MOE), (UP * 0.55 + RIGHT * 0.35, ATTN), (UP * 0.2 + RIGHT * 0.5, MOE)]
DELTAS_MONEY = [(DOWN * 0.3 + RIGHT * 0.5, ATTN), (DOWN * 0.05 + RIGHT * 0.6, MOE), (DOWN * 0.35 + RIGHT * 0.45, ATTN), (DOWN * 0.1 + RIGHT * 0.5, MOE)]


def build_lane(x: float, sentence: str, word: str, accent) -> tuple:
    """A framed plane with the sentence on top, the key word highlighted. Returns (plane, sentence)."""
    plane = card(6.3, 4.0, accent, 0.07).move_to([x, -0.35, 0])
    a, b = sentence.split(word)
    s = VGroup(T(a, 22, TOKEN), T(word, 26, accent, bold=True), T(b, 22, TOKEN)).arrange(RIGHT, buff=0.14).move_to([x, 1.35, 0])
    return plane, s


def animate_start(scene: Deck, planes) -> list:
    """The same embedding arrow appears in both lanes. Returns the two arrows."""
    origins = [p.get_center() + DOWN * 0.45 + LEFT * 2.2 for p in planes]
    e0 = [arrow(o, o + RIGHT * 0.8 + UP * 0.08, TOKEN, 6) for o in origins]
    labs = VGroup(*[T("starts as the same embedding", 16, TOKEN).move_to(p.get_bottom() + UP * 0.3, aligned_edge=DOWN)
                    .align_to(p, LEFT).shift(RIGHT * 0.3) for p in planes])
    scene.play(*[GrowArrow(a) for a in e0], FadeIn(labs))
    return e0


def animate_layers(scene: Deck, e0: list):
    """Each layer writes a coloured delta tip-to-tail; the final vectors point to river and money."""
    o_l, o_r = e0[0].get_start(), e0[1].get_start()
    tip_l, tip_r = e0[0].get_end(), e0[1].get_end()
    for (dl, cl), (dr, cr) in zip(DELTAS_RIVER, DELTAS_MONEY):
        al, ar = arrow(tip_l, tip_l + dl * 0.95, cl, 5), arrow(tip_r, tip_r + dr * 0.85, cr, 5)
        scene.play(GrowArrow(al), GrowArrow(ar), run_time=0.6)
        tip_l, tip_r = al.get_end(), ar.get_end()
    scene.play(Create(arrow(o_l, tip_l, RESID, 7)), Create(arrow(o_r, tip_r, RESID, 7)),
               FadeIn(glow(Dot(tip_l), RIVER, 3)), FadeIn(glow(Dot(tip_r), FIN, 3)))
    scene.play(FadeIn(T("river", 26, RIVER, bold=True).next_to(tip_l, RIGHT, buff=0.3)),
               FadeIn(T("money", 26, FIN, bold=True).next_to(tip_r, RIGHT, buff=0.3)))
    scene.play(Write(residual_eq().to_edge(DOWN, buff=0.45)))


class S04Bank(Deck):
    def construct(self):
        self.title("Tokens evolve in the residual stream", RESID, accent=ATTN)
        pl, sl = build_lane(LEFT_X, "She sat on the river bank", "bank", RIVER)
        pr, sr = build_lane(RIGHT_X, "He paid cash at the bank", "bank", FIN)
        self.play(FadeIn(pl), FadeIn(pr), FadeIn(sl), FadeIn(sr))
        e0 = animate_start(self, (pl, pr))
        self.pause()
        animate_layers(self, e0)
        self.pause(1.6)
        self.clear_all()
