"""Slide 12 · The residual stream is one shared bus; the Mamba state runs along time instead.

Clicks:  ▶1 blocks read and write the belt, the stack is a sum   ▶2 depth (residual) vs time (state)
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

BELT_Y = -0.5
BLOCKS = [("attention", ATTN, -5.3), ("MoE", MOE, -3.6), ("Mamba-2", MAMBA2, -1.9), ("attention", ATTN, -0.2)]


def animate_belt(scene: Deck) -> VGroup:
    """Each block reads the belt (grey arrow) and writes back (coloured arrow); a stack grows. Returns the stack."""
    belt = Line(LEFT * 6.4 + UP * BELT_Y, RIGHT * 0.6 + UP * BELT_Y, color=RESID, stroke_width=8)
    scene.play(Create(belt), FadeIn(glow(belt, RESID, 3)), FadeIn(T("r", 28, RESID, bold=True).next_to(belt, LEFT, buff=0.1)))
    stack = VGroup()
    for k, (name, c, x) in enumerate(BLOCKS):
        m = card(1.5, 0.85, c, 0.2, radius=0.14).move_to([x, 1.2, 0])
        rd = arrow([x - 0.35, BELT_Y + 0.1, 0], [x - 0.35, 0.72, 0], FADE, 3, 0.12)
        wr = arrow([x + 0.35, 0.72, 0], [x + 0.35, BELT_Y + 0.1, 0], c, 6, 0.16)
        seg = Rectangle(width=1.1, height=0.55, stroke_width=0, fill_color=c, fill_opacity=0.9).move_to([3.0, -2.4 + 0.275 + 0.58 * k, 0])
        scene.play(FadeIn(m), FadeIn(T(name, 17, c, bold=True).move_to(m)), GrowArrow(rd), run_time=0.4)
        scene.play(GrowArrow(wr), FadeIn(seg, shift=UP * 0.2), run_time=0.6)
        stack.add(seg)
    lab = T("sum of every write", 18, RESID).next_to(stack, UP, buff=0.2)
    scene.play(FadeIn(lab))
    scene.play(Write(residual_eq().to_edge(DOWN, buff=0.7).shift(LEFT * 3.2)))
    return VGroup(stack, lab)


def animate_two_axes(scene: Deck):
    """A grid of token×layer dots: mint arrows along time (state), white arrows up depth (residual)."""
    pan = card(4.6, 4.4, FADE, 0.05).move_to(RIGHT * 4.6 + UP * 0.1)
    pts = [[RIGHT * (3.1 + 1.0 * c) + UP * (-1.0 + 1.1 * r) for c in range(4)] for r in range(3)]
    scene.play(FadeIn(pan), FadeIn(VGroup(*[Dot(p, radius=0.08, color=TOKEN) for row in pts for p in row])))
    tarr = VGroup(*[arrow(pts[1][c] + RIGHT * 0.12, pts[1][c + 1] - RIGHT * 0.12, STATE, 5, 0.14) for c in range(3)])
    darr = VGroup(*[arrow(pts[r][c] + UP * 0.12, pts[r + 1][c] - UP * 0.12, RESID, 4, 0.12) for c in (0, 3) for r in range(2)])
    scene.play(LaggedStart(*[GrowArrow(a) for a in tarr], lag_ratio=0.2),
               FadeIn(T("time: Mamba state", 18, STATE, bold=True).next_to(pan, DOWN, buff=-0.45)))
    scene.play(LaggedStart(*[GrowArrow(a) for a in darr], lag_ratio=0.1),
               FadeIn(T("depth: residual", 18, RESID, bold=True).next_to(pan.get_top(), DOWN, buff=0.15)))


class S12Bus(Deck):
    def construct(self):
        self.title("The residual stream: one shared bus", RESID, accent=MOE)
        stack = animate_belt(self)
        self.pause()
        self.play(FadeOut(stack))
        animate_two_axes(self)
        self.pause(1.4)
        self.clear_all()
