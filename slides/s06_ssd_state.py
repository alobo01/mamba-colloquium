"""Slide 06 · Mamba-2 in one picture: an N×P sheet that fades, gets a rank-1 write, and is read.

Clicks:  ▶1 the sheet + update rule   ▶2 fade   ▶3 write k vᵀ   ▶4 read with q
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

N, P, CELL = 4, 6, 0.6
K_OPAC = [0.9, 0.3, 0.7, 0.5]
V_OPAC = [0.4, 0.9, 0.3, 0.8, 0.6, 0.5]
Q_OPAC = [0.8, 0.4, 0.9, 0.3]


def state_eq(size: float = 44) -> MathTex:
    """S_t = a_t S_{t-1} + k_t v_tᵀ,  y_t = S_tᵀ q_t  (parts 0..11)."""
    return ceq("S_t", "=", "a_t", "S_{t-1}", "+", "k_t", r"v_t^\top", r",\quad", "y_t", "=", r"S_t^\top", "q_t", size=size)


def build_sheet() -> tuple:
    """The state sheet with braces N and P. Returns (sheet, decorations)."""
    rng = np.random.default_rng(3)
    g = sheet(N, P, CELL, STATE)
    for sq in g:
        sq.set_fill(opacity=float(rng.uniform(0.15, 0.7)))
    g.move_to(LEFT * 1.2 + UP * 0.2)
    bN, bP = Brace(g, LEFT, color=FADE), Brace(g, UP, color=FADE)
    deco = VGroup(glow(SurroundingRectangle(g, buff=0.05, corner_radius=0.1), STATE, 3), bN, bP,
                  M("N", size=30, color=FADE).next_to(bN, LEFT, buff=0.1), M("P", size=30, color=FADE).next_to(bP, UP, buff=0.1),
                  T("the notebook: a fixed N × P sheet", 22, STATE, bold=True).next_to(g, DOWN, buff=0.35))
    return g, deco


def build_steps() -> VGroup:
    """Three pills naming the three moves."""
    s = VGroup(pill("1  fade", DECAY, 22), pill("2  write", V_COL, 22), pill("3  read", Q_COL, 22)).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
    return s.move_to(RIGHT * 5.0 + UP * 0.3)


def animate_fade(scene: Deck, g: VGroup, eq: MathTex, step):
    """Every cell is multiplied by a_t < 1."""
    scene.play(Indicate(eq[2:4], color=DECAY), step.animate.scale(1.12),
               *[sq.animate.set_fill(opacity=sq.get_fill_opacity() * 0.55) for sq in g], run_time=1.3)
    scene.play(step.animate.scale(1 / 1.12))


def animate_write(scene: Deck, g: VGroup, eq: MathTex, step, deco: VGroup) -> VGroup:
    """A key column times a value row lays a rank-1 layer onto the sheet. Returns the key column."""
    kcol = VGroup(*[Square(CELL, stroke_color=K_COL, stroke_width=2, fill_color=K_COL, fill_opacity=o) for o in K_OPAC]).arrange(DOWN, buff=0)
    kcol.next_to(g, LEFT, buff=0.35)
    vrow = VGroup(*[Square(CELL, stroke_color=V_COL, stroke_width=2, fill_color=V_COL, fill_opacity=o) for o in V_OPAC]).arrange(RIGHT, buff=0)
    vrow.next_to(g, UP, buff=0.62)
    scene.play(FadeOut(deco[1:5]), run_time=0.4)
    scene.play(FadeIn(kcol, shift=RIGHT * 0.3), FadeIn(vrow, shift=DOWN * 0.3), Indicate(eq[5:7], color=V_COL), step.animate.scale(1.12))
    layer = VGroup(*[Square(CELL, stroke_width=0, fill_color=V_COL, fill_opacity=0.75 * K_OPAC[n] * V_OPAC[p]).move_to(g[n * P + p])
                     for n in range(N) for p in range(P)])
    scene.play(FadeIn(layer), run_time=1.2)
    scene.play(*[sq.animate.set_fill(opacity=min(0.95, sq.get_fill_opacity() + l.get_fill_opacity())) for sq, l in zip(g, layer)],
               FadeOut(layer), step.animate.scale(1 / 1.12), run_time=0.9)
    return kcol


def animate_read(scene: Deck, g: VGroup, eq: MathTex, step, kcol: VGroup, deco: VGroup):
    """The key column turns into a query column; the sheet answers with an output row y."""
    qcol = VGroup(*[Square(CELL, stroke_color=Q_COL, stroke_width=2, fill_color=Q_COL, fill_opacity=o) for o in Q_OPAC]).arrange(DOWN, buff=0).move_to(kcol)
    scene.play(Transform(kcol, qcol), step.animate.scale(1.12), Indicate(eq[10:12], color=Q_COL))
    yrow = VGroup(*[Square(CELL, stroke_color=RESID, stroke_width=2, fill_color=RESID, fill_opacity=o)
                    for o in (0.5, 0.8, 0.3, 0.6, 0.9, 0.4)]).arrange(RIGHT, buff=0).next_to(g, DOWN, buff=0.35).align_to(g, LEFT)
    scene.play(FadeOut(deco[5]), Flash(g, color=Q_COL, flash_radius=2.6, line_length=0.3), FadeIn(yrow, shift=UP * 0.3),
               FadeIn(T("y", 26, RESID, bold=True).next_to(yrow, LEFT, buff=0.2)), run_time=1.4)


class S06SSDState(Deck):
    def construct(self):
        self.title("Mamba-2 in one picture", MAMBA2)
        g, deco = build_sheet()
        eq = state_eq().to_edge(DOWN, buff=0.75)
        steps = build_steps()
        self.play(FadeIn(g), FadeIn(deco))
        self.play(Write(eq))
        self.play(FadeIn(steps, lag_ratio=0.2))
        self.pause()
        animate_fade(self, g, eq, steps[0])
        self.pause(0.4)
        kcol = animate_write(self, g, eq, steps[1], deco)
        self.pause(0.4)
        animate_read(self, g, eq, steps[2], kcol, deco)
        self.pause(1.4)
        self.clear_all()
