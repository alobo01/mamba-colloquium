"""Slide 03 · Prefill → KV cache → decode.

Clicks:  ▶1 prefill paints the T×T triangle   ▶2 cache + decode, memory grows   ▶3 with vs without a cache
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

N, CELL = 8, 0.34


def animate_prefill(scene: Deck) -> VGroup:
    """The whole prompt at once: the causal triangle fills in parallel, cost T²."""
    pre = triangle(N, CELL, ATTN).move_to(LEFT * 4.2 + DOWN * 0.3).set_fill(opacity=0.1)
    pre_t = T("prefill", 30, ATTN, bold=True).next_to(pre, UP, buff=0.35)
    cost = ceq((r"T^2", ATTN), size=44).next_to(pre, DOWN, buff=0.4)
    scene.play(FadeIn(pre), FadeIn(pre_t))
    scene.play(LaggedStart(*[sq.animate.set_fill(opacity=0.85) for sq in pre], lag_ratio=0.012), run_time=1.6)
    scene.play(Write(cost))
    return pre


def animate_decode(scene: Deck, pre: VGroup, steps: int = 3):
    """K and V go into the cache; each new query scans it and appends one card. Memory bar grows."""
    cache = VGroup(*[token_card("", w=0.36, h=0.7) for _ in range(N)]).arrange(RIGHT, buff=0.05)
    cache.move_to(RIGHT * 3.0 + UP * 0.9, aligned_edge=LEFT).align_to(RIGHT * 1.0, LEFT)
    cache_t = T("KV cache", 30, V_COL, bold=True).move_to(RIGHT * 3.3 + UP * 2.0)
    scene.play(TransformFromCopy(pre, cache), FadeIn(cache_t), run_time=1.3)
    meter_bg = RoundedRectangle(width=6, height=0.3, corner_radius=0.15, stroke_color=FADE, stroke_width=2).move_to(RIGHT * 3.3 + DOWN * 1.9)
    meter = RoundedRectangle(width=6 * N / 16, height=0.3, corner_radius=0.15, stroke_width=0, fill_color=V_COL, fill_opacity=0.85)
    meter.align_to(meter_bg, LEFT).move_to(meter_bg, coor_mask=[0, 1, 0])
    mt = T("memory", 20, V_COL).next_to(meter_bg, DOWN, buff=0.12)
    scene.play(FadeIn(meter_bg), FadeIn(meter), FadeIn(mt))
    for k in range(steps):
        q = Rectangle(width=cache.width + 0.5, height=0.14, stroke_width=0, fill_color=Q_COL, fill_opacity=0.9)
        q.next_to(cache, DOWN, buff=0.15).align_to(cache, LEFT)
        newc = token_card("", w=0.36, h=0.7).next_to(cache, RIGHT, buff=0.05)
        scene.play(FadeIn(newc, shift=LEFT * 0.3), GrowFromEdge(q, LEFT), run_time=0.55)
        cache.add(newc)
        scene.play(FadeOut(q), meter.animate.stretch_to_fit_width(6 * (N + k + 1) / 16).align_to(meter_bg, LEFT), run_time=0.3)
    dec_t = T("decode", 30, Q_COL, bold=True).move_to(RIGHT * 3.3 + DOWN * 0.45)
    cost = ceq(("T", Q_COL), (r"\ \text{per new token}", Q_COL), size=38).next_to(dec_t, DOWN, buff=0.25)
    scene.play(FadeIn(dec_t), Write(cost))


def animate_comparison(scene: Deck):
    """Two pills: recompute everything vs read the cache."""
    redo = pill("no cache  →  redo  T²  every token", PROB, 22).to_edge(DOWN, buff=0.7).shift(LEFT * 3.4)
    okay = pill("with cache  →  T  per token", MAMBA2, 22).to_edge(DOWN, buff=0.7).shift(RIGHT * 3.4)
    scene.play(FadeIn(redo, shift=UP * 0.2))
    scene.play(FadeIn(okay, shift=UP * 0.2))


class S03KVCache(Deck):
    def construct(self):
        self.title("Prefill  →  KV cache  →  decode", ATTN)
        pre = animate_prefill(self)
        self.pause()
        animate_decode(self, pre)
        self.pause()
        animate_comparison(self)
        self.pause(1.4)
        self.clear_all()
