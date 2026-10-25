"""Slide 05 · Long context, then the ladder RNN → Mamba → Mamba-2 (each fixes the last one's problem).

Clicks:  ▶1 attention cost grows   ▶2 a fixed-size state stays flat
         ▶3 RNN and its two problems   ▶4 Mamba's fix and its problem   ▶5 Mamba-2's fix
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

COLS = [(-4.75, "RNN", FADE), (0, "Mamba", MAMBA), (4.75, "Mamba-2", MAMBA2)]


def animate_growing_wall(scene: Deck):
    """Theo's wall of cards grows while the attention cost curve climbs; the flat state line follows."""
    theo = brother("Theo", ATTN, "cards", 0.9).move_to(LEFT * 5.2 + DOWN * 1.8)
    scene.play(FadeIn(theo))
    ax = Axes(x_range=[0, 10, 1], y_range=[0, 10, 1], x_length=5.2, y_length=3.6, tips=False,
              axis_config={"color": FADE, "include_ticks": False, "stroke_width": 2}).move_to(RIGHT * 2.8 + DOWN * 0.1)
    quad = ax.plot(lambda x: min(10, 0.1 * x * x), x_range=[0, 10], color=ATTN, stroke_width=6)
    flat = ax.plot(lambda x: 1.2, x_range=[0, 10], color=MAMBA, stroke_width=6)
    scene.play(Create(ax), FadeIn(T("context length", 20, FADE).next_to(ax, DOWN, buff=0.15)))
    wall = VGroup(*[token_card("", w=0.5, h=0.65).move_to(theo.get_top() + UP * 0.95 + RIGHT * (0.45 * (k % 5))
                                                          + UP * (0.7 * (k // 5)) + LEFT * 0.6) for k in range(10)])
    scene.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in wall], lag_ratio=0.1), Create(quad), run_time=2.2)
    scene.play(FadeIn(T("attention", 22, ATTN, bold=True).next_to(quad.point_from_proportion(0.8), UL, buff=0.12)))
    scene.pause()
    scene.play(Create(flat), FadeIn(T("fixed-size state", 22, MAMBA, bold=True).next_to(flat.point_from_proportion(0.8), UP, buff=0.2)))


def build_columns() -> VGroup:
    """Three tall frames with headings and arrows between them."""
    frames = VGroup(*[card(4.3, 6.2, c, 0.07).move_to([x, -0.1, 0]) for x, _, c in COLS])
    heads = VGroup(*[T(n, 34, c, bold=True).move_to([x, 2.55, 0]) for x, n, c in COLS])
    arrows = VGroup(*[arrow([x - 0.3, 2.55, 0], [x + 0.3, 2.55, 0], RESID, 5, 0.16) for x in (-2.375, 2.375)])
    return VGroup(frames, heads, arrows)


def animate_rnn(scene: Deck, x: float):
    """Dominoes fall one after another (sequential), a backward pulse fades (vanishing gradients)."""
    dom = VGroup(*[RoundedRectangle(width=0.24, height=0.8, corner_radius=0.05, stroke_width=0, fill_color=FADE, fill_opacity=0.9)
                   for _ in range(5)]).arrange(RIGHT, buff=0.3).move_to([x, 1.2, 0])
    scene.play(FadeIn(dom))
    scene.play(LaggedStart(*[Rotate(d, -0.9, about_point=d.get_bottom() + RIGHT * 0.14) for d in dom], lag_ratio=0.25), run_time=1.5)
    pulse = VGroup(*[Arrow(RIGHT * 0.28, LEFT * 0.28, buff=0, stroke_width=6, tip_length=0.14, color=PROB).set_opacity(1 - 0.2 * i)
                     for i in range(5)]).arrange(LEFT, buff=0.12).move_to([x, 0.15, 0])
    scene.play(LaggedStart(*[GrowArrow(a) for a in pulse], lag_ratio=0.2), run_time=1.2)
    scene.play(FadeIn(pill("one token at a time", PROB, 19).move_to([x, -1.0, 0]), shift=UP * 0.15),
               FadeIn(pill("gradients fade away", PROB, 19).move_to([x, -1.8, 0]), shift=UP * 0.15))


def animate_mamba(scene: Deck, x: float):
    """The Δ dial turns with the input; a scan tree combines in parallel; but no matmul."""
    d = dial(MAMBA).move_to([x - 1.0, 1.2, 0])
    delta = ceq(r"\Delta_t", size=34).next_to(d, DOWN, buff=0.1)
    leaves = VGroup(*[Dot(radius=0.07, color=MAMBA) for _ in range(4)]).arrange(RIGHT, buff=0.35).move_to([x + 0.9, 0.8, 0])
    mid = VGroup(*[Dot(radius=0.07, color=MAMBA) for _ in range(2)]).arrange(RIGHT, buff=0.7).move_to([x + 0.9, 1.3, 0])
    top = Dot(radius=0.07, color=MAMBA).move_to([x + 0.9, 1.8, 0])
    tree = VGroup(*[Line(l.get_center(), m.get_center(), color=MAMBA, stroke_width=2.5) for l, m in zip(leaves, [mid[0], mid[0], mid[1], mid[1]])],
                  *[Line(m.get_center(), top.get_center(), color=MAMBA, stroke_width=2.5) for m in mid])
    scene.play(FadeIn(pill("input decides what to keep", MAMBA2, 19).move_to([x, -0.4, 0]), shift=UP * 0.15), FadeIn(d), FadeIn(delta))
    scene.play(Rotate(d[1], 1.2, about_point=d[0].get_center()), FadeIn(pill("parallel scan", MAMBA2, 19).move_to([x, -1.1, 0]), shift=UP * 0.15),
               FadeIn(VGroup(tree, leaves, mid, top)), run_time=1.3)
    scene.play(FadeIn(pill("scan can't use matmul chips", PROB, 19).move_to([x, -1.9, 0]), shift=UP * 0.15))


def animate_mamba2(scene: Deck, x: float):
    """One decay per head turns the work into tiles of matmul; bigger state, faster."""
    tiles = VGroup(*[Square(0.42, stroke_color=MAMBA2, stroke_width=2, fill_color=MAMBA2, fill_opacity=0.15) for _ in range(12)])
    tiles.arrange_in_grid(3, 4, buff=0.08).move_to([x, 1.3, 0])
    scene.play(FadeIn(pill("one decay per head", MAMBA2, 19).move_to([x, -0.4, 0]), shift=UP * 0.15), FadeIn(tiles))
    scene.play(LaggedStart(*[t.animate.set_fill(opacity=0.85) for t in tiles], lag_ratio=0.06),
               FadeIn(pill("everything becomes matmul", MAMBA2, 19).move_to([x, -1.1, 0]), shift=UP * 0.15), run_time=1.4)
    res = pill("bigger state · 2–8× faster", MAMBA2, 19, filled=True).move_to([x, -1.9, 0])
    scene.play(FadeIn(res, scale=0.9), FadeIn(glow(res[0], MAMBA2, 3)))


class S05Ladder(Deck):
    def construct(self):
        animate_growing_wall(self)
        self.pause()
        self.clear_all()
        cols = build_columns()
        self.play(FadeIn(cols[0], lag_ratio=0.2), FadeIn(cols[1], lag_ratio=0.2), FadeIn(cols[2]), run_time=1.2)
        animate_rnn(self, COLS[0][0])
        self.pause()
        animate_mamba(self, COLS[1][0])
        self.pause()
        animate_mamba2(self, COLS[2][0])
        self.pause(1.4)
        self.clear_all()
