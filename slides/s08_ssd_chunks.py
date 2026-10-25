"""Slide 08 · Fast to train: chunks, low-rank far blocks, the four SSD steps and the cost.

Clicks:  ▶1 chunked matrix   ▶2 far block = (C×N)(N×C)   ▶3 four steps + cost
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

N_TOK, CELL, CHUNK = 8, 0.4, 2
STEPS = [("1", "inside each chunk", DECAY), ("2", "summarise chunk → state", STATE),
         ("3", "pass states along", STATE), ("4", "add the past back in", Q_COL)]


def animate_chunked_matrix(scene: Deck) -> VGroup:
    """The T×T mask with diagonal chunks bright (dense) and far blocks dim."""
    cells = VGroup()
    for i in range(N_TOK):
        for j in range(i + 1):
            same = i // CHUNK == j // CHUNK
            cells.add(Square(CELL, stroke_width=1, stroke_color=BG1, fill_color=DECAY if same else ATTN,
                             fill_opacity=0.85 if same else 0.35).move_to([j * CELL, -i * CELL, 0]))
    cells.move_to(LEFT * 4.3 + DOWN * 0.3)
    scene.play(FadeIn(cells, lag_ratio=0.01), run_time=1.2)
    scene.play(FadeIn(pill("inside a chunk: dense matmul", DECAY, 18).next_to(cells, UP, buff=0.35).shift(RIGHT * 0.2)))
    return cells


def animate_low_rank(scene: Deck):
    """A far-away C×C block equals a tall (C×N) times a wide (N×C) factor."""
    big = Square(1.2, stroke_color=ATTN, stroke_width=3, fill_color=ATTN, fill_opacity=0.35)
    tall = Rectangle(width=0.35, height=1.2, stroke_color=Q_COL, stroke_width=3, fill_color=Q_COL, fill_opacity=0.5)
    wide = Rectangle(width=1.2, height=0.35, stroke_color=K_COL, stroke_width=3, fill_color=K_COL, fill_opacity=0.5)
    row = VGroup(big, M("=", size=44), VGroup(tall, M(r"\times", size=36), wide).arrange(RIGHT, buff=0.25)).arrange(RIGHT, buff=0.4)
    row.move_to(RIGHT * 2.8 + UP * 2.0)
    scene.play(FadeIn(row, shift=LEFT * 0.3), FadeIn(T("far-away block: rank ≤ N", 20, ATTN, bold=True).next_to(row, DOWN, buff=0.2)))


def animate_steps_and_cost(scene: Deck):
    """The four SSD steps as numbered rows, then the training and decoding cost."""
    rows = VGroup()
    for k, t, c in STEPS:
        dot = VGroup(Circle(radius=0.22, stroke_width=0, fill_color=c, fill_opacity=1), T(k, 22, BG1, bold=True))
        dot[1].move_to(dot[0])
        rows.add(VGroup(dot, T(t, 22, RESID)).arrange(RIGHT, buff=0.25))
    rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to(RIGHT * 3.2 + DOWN * 0.95)
    scene.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.3), run_time=1.8)
    cost = ceq(r"\text{train: }", "O", r"\big(T\,(", ("NP", STATE), "+", ("C", DECAY), r"(N{+}P))\big)",
               r"\qquad\text{decode: }", "O", r"(", ("NP", STATE), r")\ \text{per token}", size=34)
    scene.play(Write(cost.to_edge(DOWN, buff=0.7)))


class S08SSDChunks(Deck):
    def construct(self):
        self.title("Fast to train: chunks and low rank", MAMBA2)
        animate_chunked_matrix(self)
        self.pause(0.4)
        animate_low_rank(self)
        self.pause()
        animate_steps_and_cost(self)
        self.pause(1.4)
        self.clear_all()
