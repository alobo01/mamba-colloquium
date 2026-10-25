"""Slide 21 · Same algorithm, different wiring: the induction circuit in a Transformer and in Mamba.

Wang et al., ICLR 2025: in Mamba-130M the key SSM states sit in layer 17; the short convolution merges A and B
into position B+1 and writes that into the state ("off-by-one"); the second A reads it back.
Transformers: previous-token head + induction head (Olsson et al. 2022).

Clicks:  ▶1 the pattern A B … A → B   ▶2 Transformer wiring   ▶3 Mamba wiring   ▶4 off-by-one
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

TOKENS = ["A", "B", "x", "…", "A"]
XS = [-4.6, -2.9, -1.2, 0.6, 2.4]
ROW_Y = 0.2
A1, B1, NEXT, A2 = 0, 1, 2, 4


def animate_pattern(scene: Deck) -> tuple:
    """The prompt row and the prediction slot. Returns (row, answer chip)."""
    row = VGroup(*[chip(t, Q_COL if i in (A1, A2) else (K_COL if i == B1 else TOKEN), 28).move_to([x, ROW_Y, 0])
                   for i, (t, x) in enumerate(zip(TOKENS, XS))])
    ans = chip("B", K_COL, 28).move_to([4.6, ROW_Y, 0])
    scene.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in row], lag_ratio=0.15), run_time=1.2)
    scene.play(GrowArrow(arrow(row[A2].get_right() + RIGHT * 0.1, ans.get_left() + LEFT * 0.1, RESID, 4)), FadeIn(ans, scale=0.8))
    return row, ans


def animate_transformer(scene: Deck, row: VGroup):
    """Previous-token head (B looks at A), then induction head (second A finds 'the token after A')."""
    lane = pill("Transformer", ATTN, 18).move_to([-5.6, 1.6, 0])
    prev = ArcBetweenPoints(row[B1].get_top(), row[A1].get_top(), angle=1.2, stroke_color=ATTN, stroke_width=4)
    ind = ArcBetweenPoints(row[A2].get_top(), row[B1].get_top(), angle=1.0, stroke_color=ATTN, stroke_width=5)
    scene.play(FadeIn(lane))
    scene.play(Create(prev), FadeIn(T("previous-token head", 15, ATTN).next_to(prev, UP, buff=0.08)))
    scene.play(Create(ind), FadeIn(T("induction head", 15, ATTN, bold=True).next_to(ind, UP, buff=0.08)))


def animate_mamba(scene: Deck, row: VGroup):
    """Short conv at B+1 merges A and B; it is written into the layer-17 state and read at the second A."""
    lane = pill("Mamba", STATE, 18).move_to([-5.6, -1.4, 0])
    br = BraceBetweenPoints(row[A1].get_corner(DL) + DOWN * 0.08, row[NEXT].get_corner(DR) + DOWN * 0.08, color=STATE)
    blab = T("short conv at B+1", 15, STATE).next_to(br, DOWN, buff=0.08).align_to(br, LEFT)
    belt = Line([XS[NEXT], -1.9, 0], [XS[A2] + 0.6, -1.9, 0], color=STATE, stroke_width=9)
    write = arrow(row[NEXT].get_bottom() + DOWN * 0.45, [XS[NEXT], -1.8, 0], STATE, 5, 0.16)
    read = arrow([XS[A2], -1.8, 0], row[A2].get_bottom() + DOWN * 0.08, Q_COL, 5, 0.16)
    slab = T("SSM state · layer 17 of 24", 15, STATE, bold=True).next_to(belt, DOWN, buff=0.15)
    scene.play(FadeIn(lane), GrowFromCenter(br), FadeIn(blab))
    scene.play(GrowArrow(write), Create(belt), FadeIn(glow(belt, STATE, 3)), FadeIn(slab), run_time=1.2)
    scene.play(GrowArrow(read), Flash(row[A2], color=Q_COL))


def animate_off_by_one(scene: Deck, row: VGroup):
    """B's information enters the state one position late."""
    scene.play(Indicate(row[NEXT], color=STATE, scale_factor=1.3))
    scene.play(FadeIn(pill("off-by-one: B is written at B+1", STATE, 18).move_to([0.6, -3.0, 0]), shift=UP * 0.2),
               FadeIn(pill("same algorithm", MAMBA2, 18, filled=True).move_to([4.6, 1.6, 0]), scale=0.9))


class S21Induction(Deck):
    def construct(self):
        self.title("Same algorithm, different wiring", Q_COL)
        row, ans = animate_pattern(self)
        self.pause()
        animate_transformer(self, row)
        self.pause()
        animate_mamba(self, row)
        self.pause()
        animate_off_by_one(self, row)
        self.pause(1.4)
        self.clear_all()
