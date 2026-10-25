"""Reusable shapes: cards, pills, token cards, heatmaps, state sheets and glows."""

import numpy as np
from manim import (DOWN, RIGHT, Arrow, Circle, ManimColor, Polygon, Rectangle, RoundedRectangle, Square, VGroup,
                   interpolate_color)

from .theme import ATTN, BG1, K_COL, MAMBA2, RESID, SOFT, TOKEN, V_COL, T


def glow(mob, color=None, layers: int = 4, base: float = 2.0, spread: float = 3.0, opacity: float = 0.13) -> VGroup:
    """Soft halo behind a shape (stroke-only copies, widest first)."""
    g = VGroup()
    for i in range(layers, 0, -1):
        g.add(mob.copy().set_fill(opacity=0).set_stroke(color or mob.get_color(), width=base + spread * i, opacity=opacity))
    return g


def card(w: float, h: float, color=ATTN, fill: float = 0.16, radius: float = 0.2, sw: float = 2.5) -> RoundedRectangle:
    """Rounded panel used for every frame on every slide."""
    return RoundedRectangle(width=w, height=h, corner_radius=radius, stroke_color=color, stroke_width=sw,
                            fill_color=color, fill_opacity=fill)


def pill(text: str, color=ATTN, size: float = 20, filled: bool = False, pad: float = 0.28) -> VGroup:
    """A short labelled capsule: VGroup(frame, text)."""
    t = T(text, size, BG1 if filled else color, bold=True)
    r = RoundedRectangle(width=t.width + pad * 2, height=t.height + 0.3, corner_radius=0.25,
                         stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=1 if filled else 0.14)
    return VGroup(r, t.move_to(r))


def chip(text: str, color=TOKEN, size: float = 20) -> VGroup:
    """A token or short phrase: VGroup(frame, text)."""
    t = T(text, size, color)
    r = RoundedRectangle(width=t.width + 0.4, height=0.55, corner_radius=0.16, stroke_color=color,
                         stroke_width=2, fill_color=color, fill_opacity=0.1)
    return VGroup(r, t.move_to(r))


def token_card(text: str, w: float = 1.2, h: float = 1.35, key=K_COL, val=V_COL, size: float = 20) -> VGroup:
    """An index card: purple key strip on top, yellow value body. VGroup(body, strip, label)."""
    body = RoundedRectangle(width=w, height=h, corner_radius=0.12, stroke_color=TOKEN, stroke_width=2,
                            fill_color=val, fill_opacity=0.16)
    strip = Rectangle(width=w - 0.04, height=0.3, stroke_width=0, fill_color=key, fill_opacity=0.9)
    strip.move_to(body.get_top() + DOWN * 0.17)
    label = T(text, size, RESID).move_to(body).shift(DOWN * 0.12)
    return VGroup(body, strip, label)


def vec(n: int = 4, color=ATTN, cell: float = 0.17, vals=None, horizontal: bool = False) -> VGroup:
    """A vector drawn as a strip of cells whose opacity is the entry."""
    vals = np.linspace(0.3, 1, n) if vals is None else vals
    g = VGroup(*[Square(cell, stroke_color=color, stroke_width=1.2, fill_color=color, fill_opacity=float(v)) for v in vals])
    return g.arrange(RIGHT if horizontal else DOWN, buff=0.02)


def arrow(start, end, color=RESID, sw: float = 5, tip: float = 0.18) -> Arrow:
    return Arrow(start, end, buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=0.35)


def heat(data, cell: float = 0.34, pos=MAMBA2, zero="#1E2740", stroke="#0A0D15", numbers: bool = False,
         nsize: float = 14) -> VGroup:
    """Heatmap of |data|, row-major, centred at the origin. Optional printed values."""
    from .theme import fmt_weight
    data = np.asarray(data, dtype=float)
    n, m = data.shape
    vmax = np.abs(data).max() or 1.0
    g = VGroup()
    for i in range(n):
        for j in range(m):
            v = abs(data[i, j]) / vmax
            col = interpolate_color(ManimColor(zero), ManimColor(pos), min(1, v))
            sq = Square(cell, stroke_width=1, stroke_color=stroke, fill_color=col, fill_opacity=1)
            sq.move_to([j * cell, -i * cell, 0])
            g.add(sq)
            if numbers and data[i, j] != 0:
                g.add(T(fmt_weight(data[i, j]), nsize, BG1 if v > 0.55 else RESID, bold=True).move_to(sq))
    return g.move_to([0, 0, 0])


def causal_softmax(n: int, seed: int = 0, temp: float = 1.5) -> np.ndarray:
    """A plausible causal softmax attention map for illustrations."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, n)) * temp
    s[np.triu_indices(n, 1)] = -np.inf
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def triangle(n: int, cell: float, color, pattern=lambda i, j: True, opacity: float = 0.85) -> VGroup:
    """Lower-triangular grid of cells; `pattern(i, j)` decides which cells are lit."""
    g = VGroup()
    for i in range(n):
        for j in range(i + 1):
            on = pattern(i, j)
            g.add(Square(cell, stroke_width=0.8, stroke_color=BG1, fill_color=color if on else SOFT,
                         fill_opacity=opacity if on else 0.6).move_to([j * cell, -i * cell, 0]))
    return g.move_to([0, 0, 0])


def sheet(N: int, P: int, cell: float = 0.55, color=MAMBA2, opacity: float = 0.35) -> VGroup:
    """The N x P state matrix as a grid of cells, row-major."""
    g = VGroup(*[Square(cell, stroke_color=color, stroke_width=1.5, fill_color=color, fill_opacity=opacity)
                 .move_to([p * cell, -n * cell, 0]) for n in range(N) for p in range(P)])
    return g.move_to([0, 0, 0])


def bars(heights, color=ATTN, width: float = 0.22, buff: float = 0.08) -> VGroup:
    """Bottom-aligned bar chart (no axes)."""
    return VGroup(*[Rectangle(width=width, height=max(0.02, h), stroke_width=0, fill_color=color, fill_opacity=0.9)
                    for h in heights]).arrange(RIGHT, buff=buff, aligned_edge=DOWN)


def hourglass(top, color, w: float = 0.5, h: float = 1.4) -> VGroup:
    """The FFN glyph: expand then squeeze, with four hidden units in the middle."""
    from manim import LEFT, Dot
    poly = Polygon(top + LEFT * 0.2, top + RIGHT * 0.2, top + DOWN * h / 2 + RIGHT * w, top + DOWN * h + RIGHT * 0.2,
                   top + DOWN * h + LEFT * 0.2, top + DOWN * h / 2 + LEFT * w,
                   stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.18)
    dots = VGroup(*[Dot(radius=0.05, color=color).move_to(top + DOWN * h / 2 + RIGHT * dx) for dx in (-0.3, -0.1, 0.1, 0.3)])
    return VGroup(poly, dots)


def dial(color, radius: float = 0.32) -> VGroup:
    """A knob: VGroup(ring, needle)."""
    from manim import ORIGIN, Line
    return VGroup(Circle(radius=radius, stroke_color=color, stroke_width=4),
                  Line(ORIGIN, [0, radius * 0.8, 0], color=color, stroke_width=4))
