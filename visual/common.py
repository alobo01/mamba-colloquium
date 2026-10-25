"""Shared palette and mobject helpers for every slide (Appendix A of the script)."""

from manim import *
import numpy as np

import os

# Video export uses plain Scenes. Set MANIM_DECK=slides when running
# `manim-slides render` to get click-stepped presentations instead.
_Slide, _ThreeDSlide = Scene, ThreeDScene
if os.environ.get("MANIM_DECK") == "slides":
    from manim_slides import Slide as _Slide, ThreeDSlide as _ThreeDSlide

BG = "#11141C"
RESID = "#F2F2F2"
TOKEN = "#D8D5CD"
Q_COL = "#3AD5D0"
K_COL = "#B78CFF"
V_COL = "#FFD166"
ATTN = "#4F8EF7"
MAMBA = "#F28C6B"
MAMBA2 = "#69D6B4"
MOE = "#F2C14E"
FADE = "#747D8C"

config.background_color = BG
FONT = "DejaVu Sans"

BADGE_COLORS = {"IDENTITY": RESID, "PROPERTY": ATTN, "SCHEMATIC": FADE, "MEASURED": MOE}


def T(text, size=28, color=RESID, **kw):
    return Text(text, font=FONT, font_size=size, color=color, **kw)


def M(*tex, size=40, color=RESID, **kw):
    return MathTex(*tex, font_size=size, color=color, **kw)


class _PauseMixin:
    """`pause()` = one presenter click in the deck, a short hold in the video."""

    def pause(self, t=0.8):
        self.wait(t)
        if hasattr(self, "next_slide"):
            self.next_slide()

    def header(self, number, title, act=None, badge=None):
        t = T(f"{number}  {title}", 34)
        if t.width > 10.8:
            t.scale_to_fit_width(10.8)
        t.to_edge(UP, buff=0.35).to_edge(LEFT, buff=0.5)
        grp = VGroup(t)
        if act:
            grp.add(T(act, 18, FADE).next_to(t, DOWN, aligned_edge=LEFT, buff=0.1))
        if badge:
            grp.add(evidence_badge(badge).to_corner(UR, buff=0.35))
        self.play(FadeIn(grp, shift=DOWN * 0.2), run_time=0.8)
        return grp

    def caption(self, text, color=RESID, size=26):
        c = T(text, size, color).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(c, shift=UP * 0.2), run_time=0.6)
        return c

    def dont_say(self, text):
        box = VGroup(T("⚠ Don't say: " + text, 20, MAMBA))
        rect = SurroundingRectangle(box, color=MAMBA, buff=0.15, corner_radius=0.1)
        g = VGroup(rect, box).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(g), run_time=0.6)
        return g

    def clear_all(self, run_time=0.6):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=run_time)


class Deck(_PauseMixin, _Slide):
    pass


class Deck3D(_PauseMixin, _ThreeDSlide):
    pass


def evidence_badge(kind):
    col = BADGE_COLORS[kind]
    label = T(kind, 16, col)
    rect = RoundedRectangle(width=label.width + 0.35, height=0.4, corner_radius=0.18,
                            stroke_color=col, stroke_width=2)
    if kind == "SCHEMATIC":
        rect = DashedVMobject(rect, num_dashes=24)
    return VGroup(rect, label.move_to(rect))


def token_card(text, w=1.15, h=1.3, key=K_COL, val=V_COL, size=20):
    body = RoundedRectangle(width=w, height=h, corner_radius=0.08, stroke_color=TOKEN,
                            stroke_width=2, fill_color=val, fill_opacity=0.18)
    strip = Rectangle(width=w, height=0.32, stroke_width=0, fill_color=key, fill_opacity=0.85)
    strip.align_to(body, UP)
    label = T(text, size, RESID).move_to(body).shift(DOWN * 0.1)
    return VGroup(body, strip, label)


def token_square(text, size=0.7, color=TOKEN, fs=18):
    sq = RoundedRectangle(width=max(size, 0.22 * len(text) + 0.3), height=size, corner_radius=0.08,
                          stroke_color=color, stroke_width=2)
    return VGroup(sq, T(text, fs, color).move_to(sq))


def brother(name, color, prop):
    head = Circle(radius=0.35, stroke_color=color, stroke_width=4)
    body = Arc(radius=0.6, start_angle=0, angle=PI, stroke_color=color, stroke_width=4)
    body.next_to(head, DOWN, buff=0.05)
    if prop == "cards":
        p = VGroup(*[Rectangle(width=0.28, height=0.38, stroke_color=color, stroke_width=2,
                               fill_color=K_COL, fill_opacity=0.3) for _ in range(4)]).arrange(RIGHT, buff=0.05)
    elif prop == "notebook":
        p = VGroup(Rectangle(width=0.9, height=0.6, stroke_color=color, fill_color=color, fill_opacity=0.25),
                   *[Line(LEFT * 0.35, RIGHT * 0.35, stroke_width=1.5, color=color).shift(UP * (0.15 - 0.15 * i))
                     for i in range(3)])
    else:  # ledger of tiles
        p = VGroup(*[Square(0.2, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.35)
                     for _ in range(6)]).arrange_in_grid(2, 3, buff=0.05)
    p.next_to(body, DOWN, buff=0.12)
    label = T(name, 24, color).next_to(p, DOWN, buff=0.15)
    return VGroup(head, body, p, label)


def heatmap(data, cell=0.3, pos_color=MAMBA2, neg_color=MAMBA, zero_color="#1B2030", stroke=FADE):
    data = np.asarray(data, dtype=float)
    n, m = data.shape
    vmax = np.max(np.abs(data)) or 1.0
    cells = VGroup()
    for i in range(n):
        for j in range(m):
            v = data[i, j] / vmax
            col = interpolate_color(ManimColor(zero_color), ManimColor(pos_color if v >= 0 else neg_color), min(1, abs(v)))
            cells.add(Square(cell, stroke_width=0.6, stroke_color=stroke, fill_color=col, fill_opacity=1)
                      .move_to([j * cell, -i * cell, 0]))
    cells.move_to(ORIGIN)
    return cells


def causal_softmax(n, seed=0, temp=1.5):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, n)) * temp
    s[np.triu_indices(n, 1)] = -np.inf
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def vec_arrow(start, end, color, sw=5):
    return Arrow(start, end, buff=0, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.15)


def bar_row(values, width=0.35, max_h=1.4, color=K_COL, buff=0.15):
    bars = VGroup(*[Rectangle(width=width, height=max(0.02, max_h * v), stroke_width=0,
                              fill_color=color, fill_opacity=0.9) for v in values])
    bars.arrange(RIGHT, buff=buff, aligned_edge=DOWN)
    return bars
