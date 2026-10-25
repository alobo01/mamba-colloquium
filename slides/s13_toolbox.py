"""Slide 13 · Real Transformers are cleverer: ten tricks grouped by how they work.

Four columns, one per mechanism: share the cache, see fewer tokens, same maths with less memory traffic,
reach further. Each tile names the trick, who introduced it and where it ships.

Clicks:  ▶1 share the cache   ▶2 see fewer tokens   ▶3 same maths, less traffic   ▶4 reach further
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

GROUPS = [  # (heading, what it shrinks, colour, [(name, introduced by, flagship use, icon)])
    ("Share the cache", "fewer keys & values stored", MAMBA2, [
        ("MQA", "Shazeer 2019", "PaLM", "mqa"),
        ("GQA", "Ainslie et al. 2023", "Llama 2 70B, Llama 3", "gqa"),
        ("MLA", "DeepSeek 2024", "DeepSeek-V2 / V3", "mla")]),
    ("See fewer tokens", "a sparser mask", V_COL, [
        ("Sliding window", "Longformer 2020", "Mistral 7B", "swa"),
        ("Local + global", "Google 2024", "Gemma 2 / 3", "localglobal"),
        ("Attention sinks", "Xiao et al. 2023", "StreamingLLM", "sink"),
        ("Sparse attention", "BigBird 2020 · NSA 2025", "DeepSeek", "sparse")]),
    ("Same maths, less traffic", "exact attention, better memory use", Q_COL, [
        ("FlashAttention", "Dao et al. 2022", "nearly every LLM", "flash"),
        ("PagedAttention", "vLLM 2023", "LLM serving", "paged")]),
    ("Reach further", "positions beyond training length", FIN, [
        ("RoPE + YaRN", "Su 2021 · Peng 2023", "Llama, Qwen", "rope")]),
]
COL_W, TILE_H, TILE_GAP = 3.15, 1.1, 0.12
COL_X = [-4.95, -1.65, 1.65, 4.95]
TILES_TOP = 1.5

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
    """Compact horizontal card: icon on the left; name, origin and flagship model on the right."""
    fr = card(COL_W, TILE_H, color, 0.09, radius=0.16)
    ic = icon(kind, color)
    ic.scale(min(0.8 / ic.width, 0.8 / ic.height)).move_to(fr.get_left() + RIGHT * 0.62)
    texts = VGroup(T(name, 18, color, bold=True), T(who, 13, TOKEN), T(model, 13, FADE))
    for t in texts:
        if t.width > COL_W - 1.45:
            t.scale_to_fit_width(COL_W - 1.45)
    texts.arrange(DOWN, aligned_edge=LEFT, buff=0.07)
    texts.move_to(fr.get_left() + RIGHT * 1.27, aligned_edge=LEFT)
    return VGroup(fr, ic, texts)


def build_column(x: float, heading: str, sub: str, color, tricks: list) -> tuple:
    """Group heading (pill + what it shrinks) above a stack of tiles. Returns (header, tiles)."""
    header = VGroup(pill(heading, color, 18), T(sub, 14, FADE)).arrange(DOWN, buff=0.12).move_to([x, 2.1, 0])
    tiles = VGroup(*[build_tile(*t, color) for t in tricks]).arrange(DOWN, buff=TILE_GAP)
    tiles.move_to([x, TILES_TOP, 0], aligned_edge=UP)
    return header, tiles


def animate_column(scene: Deck, header: VGroup, tiles: VGroup):
    """Heading drops in, then its tiles cascade down."""
    scene.play(FadeIn(header, shift=DOWN * 0.2), run_time=0.5)
    scene.play(LaggedStart(*[FadeIn(t, shift=UP * 0.25, scale=0.95) for t in tiles], lag_ratio=0.2), run_time=0.5 + 0.35 * len(tiles))


class S13Toolbox(Deck):
    def construct(self):
        self.title("Real Transformers are cleverer than that", ATTN)
        for x, (heading, sub, color, tricks) in zip(COL_X, GROUPS):
            animate_column(self, *build_column(x, heading, sub, color, tricks))
            self.pause()
        self.wait(0.7)
        self.clear_all()
