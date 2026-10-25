"""Slide 13 · Real Transformers are cleverer: ten tricks, who introduced them, who uses them.

Clicks:  ▶1 the ten tiles
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

TRICKS = [  # (name, introduced by, flagship use, icon, colour)
    ("MQA", "Shazeer 2019", "PaLM", "mqa", MAMBA2),
    ("GQA", "Ainslie et al. 2023", "Llama 2 70B, Llama 3", "gqa", MAMBA2),
    ("Sliding window", "Longformer 2020", "Mistral 7B", "swa", V_COL),
    ("Local + global", "Google 2024", "Gemma 2 / 3", "localglobal", V_COL),
    ("Attention sinks", "Xiao et al. 2023", "StreamingLLM", "sink", V_COL),
    ("MLA", "DeepSeek 2024", "DeepSeek-V2 / V3", "mla", MAMBA2),
    ("FlashAttention", "Dao et al. 2022", "nearly every LLM", "flash", Q_COL),
    ("PagedAttention", "vLLM 2023", "LLM serving", "paged", Q_COL),
    ("Sparse attention", "BigBird 2020 · NSA 2025", "DeepSeek", "sparse", K_COL),
    ("RoPE + YaRN", "Su 2021 · Peng 2023", "Llama, Qwen", "rope", FIN),
]

MASKS = {"swa": lambda i, j: i - j <= 1, "localglobal": lambda i, j: i - j <= 1 or j % 3 == 0,
         "sink": lambda i, j: j == 0 or i - j <= 1, "sparse": lambda i, j: (i * 3 + j * 5) % 4 == 0 or i == j}


def sharing(nq: int, nkv: int, color) -> VGroup:
    """Query heads (top) wired to fewer shared key/value heads (bottom)."""
    qs = VGroup(*[Dot(radius=0.065, color=Q_COL) for _ in range(nq)]).arrange(RIGHT, buff=0.2)
    ks = VGroup(*[Dot(radius=0.075, color=color) for _ in range(nkv)]).arrange(RIGHT, buff=0.2 * nq / nkv).move_to(qs).shift(DOWN * 0.8)
    lines = VGroup(*[Line(q.get_center(), ks[min(nkv - 1, i * nkv // nq)].get_center(), color=color, stroke_width=1.8, stroke_opacity=0.8)
                     for i, q in enumerate(qs)])
    return VGroup(lines, qs, ks)


def icon(kind: str, color) -> VGroup:
    """Tiny diagram for one trick."""
    if kind == "mqa":
        return sharing(6, 1, color)
    if kind == "gqa":
        return sharing(6, 2, color)
    if kind in MASKS:
        return triangle(6, 0.17, color, MASKS[kind], opacity=0.95)
    if kind == "mla":
        wide = Rectangle(width=1.0, height=0.28, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.35)
        lat = Rectangle(width=0.35, height=0.28, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.9)
        g = VGroup(wide, lat, wide.copy()).arrange(DOWN, buff=0.22)
        return VGroup(g, *[Arrow(a.get_bottom(), b.get_top(), buff=0.03, stroke_width=2, tip_length=0.08, color=FADE)
                           for a, b in [(g[0], g[1]), (g[1], g[2])]])
    if kind == "flash":
        grid = VGroup(*[Square(0.26, stroke_color=color, stroke_width=1.5, fill_color=color, fill_opacity=0.12) for _ in range(9)])
        grid.arrange_in_grid(3, 3, buff=0.04)[4].set_fill(color, 0.9)
        return grid
    if kind == "paged":
        return VGroup(*[Rectangle(width=0.3, height=0.5, stroke_color=c, stroke_width=2, fill_color=c, fill_opacity=0.5)
                        for c in (color, V_COL, MAMBA2, K_COL)]).arrange(RIGHT, buff=0.14)
    if kind == "rope":
        return VGroup(Circle(radius=0.45, stroke_color=color, stroke_width=3), arrow(ORIGIN, UP * 0.4 + RIGHT * 0.2, color, 4, 0.12),
                      arrow(ORIGIN, RIGHT * 0.42 + UP * 0.02, Q_COL, 3, 0.1).rotate(-0.5, about_point=ORIGIN))
    return VGroup()


def build_tile(name: str, who: str, model: str, kind: str, color) -> VGroup:
    """Card with icon, name, origin and flagship model."""
    fr = card(2.55, 2.8, color, 0.09, radius=0.18)
    texts = [T(name, 20, color, bold=True).move_to(fr.get_center() + DOWN * 0.4),
             T(who, 14, TOKEN).move_to(fr.get_center() + DOWN * 0.82), T(model, 14, FADE).move_to(fr.get_center() + DOWN * 1.1)]
    for t in texts:
        if t.width > 2.3:
            t.scale_to_fit_width(2.3)
    return VGroup(fr, icon(kind, color).scale(1.1).move_to(fr.get_top() + DOWN * 0.9), *texts)


class S13Toolbox(Deck):
    def construct(self):
        self.title("Real Transformers are cleverer than that", ATTN)
        tiles = VGroup(*[build_tile(*t) for t in TRICKS]).arrange_in_grid(2, 5, buff=0.17).move_to(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.3, scale=0.94) for t in tiles], lag_ratio=0.12), run_time=3.2)
        self.pause(1.6)
        self.clear_all()
