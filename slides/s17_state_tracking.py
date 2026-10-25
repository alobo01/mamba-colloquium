"""Slide 17 · Tracking state: a blind spot for both, and the one-line fix for parity.

Merrill, Petty & Sabharwal 2024 "The Illusion of State in State-Space Models" (both families in TC⁰);
Sarrof, Veitsman & Hahn 2024 and Grazzi et al. ICLR 2025 (decays in [0,1] cannot do parity; allow negative ones).

Clicks:  ▶1 the shell game   ▶2 parity, two lanes   ▶3 the product formula
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

SWAPS = [(0, 3), (1, 4), (2, 3), (0, 4), (1, 2)]      # five cups: the hard case is S5 (non-solvable)
BITS = [1, 0, 1, 1, 0]                   # three ones -> odd parity
CUP_X = [-6.1, -5.0, -3.9, -2.8, -1.7]
LANE_Y = [0.35, -1.45]


def build_cup(x: float) -> VGroup:
    body = Polygon([-0.42, -0.4, 0], [0.42, -0.4, 0], [0.3, 0.4, 0], [-0.3, 0.4, 0], stroke_color=MOE, stroke_width=3,
                   fill_color=MOE, fill_opacity=0.35)
    return VGroup(body).move_to([x, 0.6, 0])


def animate_shell_game(scene: Deck) -> VGroup:
    """A ball hides under the middle cup; five swaps; the question."""
    cups = VGroup(*[build_cup(x) for x in CUP_X])
    ball = Dot(radius=0.16, color=Q_COL).move_to([CUP_X[2], 0.0, 0])
    scene.play(FadeIn(ball), FadeIn(cups, shift=DOWN * 0.3))
    scene.play(cups.animate.shift(DOWN * 0.45), run_time=0.5)
    scene.remove(ball)
    order = list(range(len(CUP_X)))      # order[slot] = cup index standing at that slot
    for a, b in SWAPS:
        ca, cb = cups[order[a]], cups[order[b]]
        scene.play(Swap(ca, cb, path_arc=PI / 2), run_time=0.6)
        order[a], order[b] = order[b], order[a]
    q = chip("where is the ball?", Q_COL, 22).move_to([CUP_X[2], 1.85, 0])
    scene.play(FadeIn(q, scale=0.85))
    labs = VGroup(pill("composing swaps of 5 cups", MOE, 18), T("fixed-depth Transformers and Mamba", 16, FADE), T("both live in TC⁰: not for long sequences", 16, FADE))
    labs.arrange(DOWN, buff=0.15).move_to([CUP_X[2], -1.35, 0])
    scene.play(FadeIn(labs, shift=UP * 0.2))
    return VGroup(cups, q, labs)


def build_lane(y: float, *tex: str) -> MathTex:
    """Label for one parity lane, e.g. a_t ∈ [0, 1]."""
    return ceq(*tex, size=32).move_to([1.6, y, 0])


def animate_parity(scene: Deck):
    """Bits arrive; with a ∈ [0,1] the state arrow can only shrink, with a ∈ [−1,1] it flips on every 1."""
    bits = VGroup(*[chip(str(b), RESID if b else FADE, 24) for b in BITS]).arrange(RIGHT, buff=0.3).move_to([4.2, 1.85, 0])
    scene.play(FadeIn(T("parity", 24, RESID, bold=True).next_to(bits, LEFT, buff=0.4)), FadeIn(bits))
    lanes = VGroup(build_lane(LANE_Y[0], "a_t", r"\in[0,1]"), build_lane(LANE_Y[1], "a_t", r"\in[-1,1]"))
    origins = [np.array([3.6, y, 0]) for y in LANE_Y]
    shrink = arrow(origins[0], origins[0] + UP * 0.75, STATE, 6)
    flip = arrow(origins[1], origins[1] + UP * 0.75, STATE, 6)
    axes = VGroup(*[DashedLine(o + DOWN * 0.85, o + UP * 0.85, color=FADE, stroke_width=1.5) for o in origins])
    scene.play(FadeIn(lanes), FadeIn(axes), GrowArrow(shrink), GrowArrow(flip))
    length = 0.75
    for b, c in zip(BITS, bits):
        if b:
            length *= 0.6
            scene.play(Indicate(c, color=Q_COL, scale_factor=1.2),
                       Transform(shrink, arrow(origins[0], origins[0] + UP * length, STATE, 6)),
                       Rotate(flip, PI, about_point=origins[1]), run_time=0.7)
        else:
            scene.play(Indicate(c, color=FADE, scale_factor=1.1), run_time=0.4)
    scene.play(FadeIn(pill("fades, can't flip", PROB, 18).move_to([5.6, LANE_Y[0], 0])),
               FadeIn(pill("odd: points down", MAMBA2, 18).move_to([5.6, LANE_Y[1], 0])))


def animate_formula(scene: Deck):
    """s_T = (∏ a_t) s_0 with a_t = −1 on every 1-bit gives (−1)^{#ones} s_0."""
    eq = ceq("s_T", "=", r"\textstyle\prod_t", "a_t", r"\,", "s_0", "=", r"(-1)^{\#1}", "s_0", size=38).move_to(RIGHT * 3.4 + DOWN * 3.0)
    scene.play(Write(eq))


class S17StateTracking(Deck):
    def construct(self):
        self.title("Tracking state: a blind spot for both", PROB)
        animate_shell_game(self)
        self.pause()
        animate_parity(self)
        self.pause()
        animate_formula(self)
        self.pause(1.4)
        self.clear_all()
