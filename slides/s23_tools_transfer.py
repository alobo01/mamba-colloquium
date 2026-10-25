"""Slide 23 · The interpretability toolbox transfers from Transformers to Mamba.

Paulo, Marshall & Belrose 2024 (tuned lens, steering, "state steering" on Mamba/RWKV);
Sen Sharma, Atkinson & Bau 2024 (causal tracing + ROME in Mamba: middle layers at the last subject token);
Endy et al. 2025 (attention knockout adapted to Mamba-1/2); Wang et al. 2025 (SAEs, circuits).

Clicks:  ▶1 five tools   ▶2 they work, and facts sit where they sit in GPT
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

TOOLS = [("tuned lens", "Paulo et al. 2024", ATTN), ("state steering", "Paulo et al. 2024", STATE),
         ("causal tracing + ROME", "Sen Sharma et al. 2024", MOE), ("knockout", "Endy et al. 2025", Q_COL),
         ("SAEs + circuits", "Wang et al. 2025", K_COL)]


def build_tile(name: str, who: str, color) -> VGroup:
    fr = card(2.5, 3.9, color, 0.09, radius=0.18)
    nm = T(name, 18, color, bold=True).move_to(fr.get_center() + DOWN * 1.05)
    wh = T(who, 14, FADE).move_to(fr.get_center() + DOWN * 1.45)
    for t in (nm, wh):
        if t.width > 2.3:
            t.scale_to_fit_width(2.3)
    return VGroup(fr, nm, wh)


def tile_anchor(tile: VGroup):
    return tile[0].get_center() + UP * 0.6


def lens_visual(at) -> VGroup:
    """Next-token guesses per layer: flat early, peaked late."""
    return bars([0.45, 0.4, 0.5, 0.42, 0.45], ATTN, 0.24).move_to(at)


def steer_visual(at) -> VGroup:
    """A small state grid with a steering vector added."""
    g = sheet(3, 3, 0.3, STATE, 0.35).move_to(at + LEFT * 0.3)
    v = arrow(at + RIGHT * 0.95 + DOWN * 0.45, at + RIGHT * 0.95 + UP * 0.45, MOE, 5, 0.15)
    return VGroup(g, M("+", size=30).move_to(at + RIGHT * 0.45), v)


def trace_visual(at) -> VGroup:
    """Layers × tokens grid: the hot spot is mid-depth at the last subject token."""
    data = np.full((5, 5), 0.08)
    data[1:4, 2] = [0.6, 1.0, 0.7]
    data[3:, 4] = [0.5, 0.8]
    return heat(data, 0.26, pos=MOE).move_to(at)


def knockout_visual(at) -> VGroup:
    """Three tokens with information arcs; one arc is cut."""
    dots = VGroup(*[Dot(radius=0.09, color=TOKEN) for _ in range(3)]).arrange(RIGHT, buff=0.55).move_to(at + DOWN * 0.35)
    a1 = ArcBetweenPoints(dots[2].get_top(), dots[0].get_top(), angle=1.1, stroke_color=Q_COL, stroke_width=4)
    a2 = ArcBetweenPoints(dots[2].get_top(), dots[1].get_top(), angle=1.1, stroke_color=Q_COL, stroke_width=4)
    cut = Cross(scale_factor=0.12, stroke_color=PROB, stroke_width=5).move_to(a1.point_from_proportion(0.5))
    return VGroup(dots, a1, a2, cut)


def sae_visual(at) -> VGroup:
    """Dense activation → sparse features."""
    dense = vec(5, K_COL, 0.2, vals=[0.8, 0.6, 0.9, 0.7, 0.5]).move_to(at + LEFT * 0.55)
    sparse = VGroup(*[Square(0.2, stroke_color=K_COL, stroke_width=1.2, fill_color=K_COL, fill_opacity=0.9 if i in (1, 5) else 0.05)
                      for i in range(7)]).arrange(DOWN, buff=0.02).move_to(at + RIGHT * 0.55)
    return VGroup(dense, Arrow(LEFT * 0.2, RIGHT * 0.2, buff=0, stroke_width=3, color=FADE).move_to(at), sparse)


def animate_tools(scene: Deck) -> VGroup:
    tiles = VGroup(*[build_tile(*t) for t in TOOLS]).arrange(RIGHT, buff=0.2).move_to(DOWN * 0.15)
    scene.play(LaggedStart(*[FadeIn(t, shift=UP * 0.3) for t in tiles], lag_ratio=0.12), run_time=1.6)
    vis = [lens_visual(tile_anchor(tiles[0])), steer_visual(tile_anchor(tiles[1])), trace_visual(tile_anchor(tiles[2])),
           knockout_visual(tile_anchor(tiles[3])), sae_visual(tile_anchor(tiles[4]))]
    scene.play(*[FadeIn(v) for v in vis])
    lens = vis[0]
    scene.play(Transform(lens, bars([0.2, 0.25, 1.25, 0.3, 0.2], ATTN, 0.24).move_to(tile_anchor(tiles[0])).align_to(lens, DOWN)),
               vis[1][2].animate.set_color(STATE), Circumscribe(vis[2], color=MOE, fade_out=True),
               vis[4][2][1].animate.scale(1.25), vis[4][2][5].animate.scale(1.25), run_time=1.6)
    return tiles


class S23ToolsTransfer(Deck):
    def construct(self):
        self.title("The interpretability toolbox transfers", K_COL)
        animate_tools(self)
        self.pause()
        self.play(FadeIn(pill("facts live mid-depth at the last subject token, in both", MOE, 18).move_to(DOWN * 2.85), shift=UP * 0.2))
        self.pause(1.4)
        self.clear_all()
