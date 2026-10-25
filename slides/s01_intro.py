"""Slide 01 · Three brothers, three memories.

Clicks:  ▶1 cast + story   ▶2 how each brother stores it   ▶3 the question
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

STORY = ["Leo", "hid", "the green key", "under the bridge"]


def build_cast() -> VGroup:
    """Theo, Manu and Max standing in a row: VGroup(theo, manu, max)."""
    theo = brother("Theo", ATTN, "cards").move_to(LEFT * 4.6 + DOWN * 1.5)
    manu = brother("Manu", MAMBA, "notebook").move_to(DOWN * 1.5)
    maxx = brother("Max", MAMBA2, "tiles").move_to(RIGHT * 4.6 + DOWN * 1.5)
    return VGroup(theo, manu, maxx)


def animate_title(scene: Deck) -> Text:
    """Big title that shrinks into the corner."""
    ttl = T("Three brothers.  Three memories.", 56, RESID, bold=True)
    scene.play(Write(ttl), run_time=1.4)
    scene.wait(0.6)
    scene.play(ttl.animate.scale(0.55).to_corner(UL, buff=0.5))
    return ttl


def animate_story(scene: Deck, cast: VGroup) -> VGroup:
    """Brothers walk in, then the story ribbon appears above them."""
    scene.play(LaggedStart(*[FadeIn(b, shift=UP * 0.4) for b in cast], lag_ratio=0.25), run_time=1.4)
    ribbon = VGroup(*[chip(w, TOKEN, 24) for w in STORY]).arrange(RIGHT, buff=0.25).to_edge(UP, buff=1.2)
    scene.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.3) for c in ribbon], lag_ratio=0.2), run_time=1.4)
    return ribbon


def animate_memories(scene: Deck, cast: VGroup, ribbon: VGroup) -> tuple:
    """Theo pins one card per token, Manu writes one page, Max fills tiles. Returns (wall, page, tiles)."""
    theo, manu, maxx = cast
    wall = VGroup(*[token_card(w.split()[-1][:6], w=0.95, h=1.1, size=15) for w in STORY]).arrange(RIGHT, buff=0.08)
    wall.next_to(theo, UP, buff=0.3).align_to(theo, LEFT).shift(LEFT * 0.2)
    page = card(2.2, 1.2, MAMBA, 0.18).next_to(manu, UP, buff=0.3)
    lines = VGroup(*[Line(LEFT * 0.8, RIGHT * 0.8, color=MAMBA, stroke_width=3).move_to(page).shift(UP * (0.36 - 0.24 * i))
                     for i in range(4)])
    tiles = VGroup(*[Square(0.5, stroke_color=MAMBA2, stroke_width=2, fill_color=MAMBA2, fill_opacity=0.2) for _ in range(6)])
    tiles.arrange_in_grid(2, 3, buff=0.1).next_to(maxx, UP, buff=0.4)
    scene.play(*[TransformFromCopy(ribbon[i], wall[i]) for i in range(4)], FadeIn(page), FadeIn(tiles), run_time=1.4)
    scene.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.25),
               LaggedStart(*[t.animate.set_fill(opacity=0.7) for t in tiles], lag_ratio=0.15), run_time=1.6)
    return wall, page, tiles


def animate_question(scene: Deck, wall: VGroup, page, tiles: VGroup):
    """The question lights up the right card on Theo's wall and the whole page/tiles of the others."""
    q = chip("Where is Leo's key?", Q_COL, 26).move_to(UP * 1.7)
    scene.play(FadeIn(q, scale=0.8))
    scene.play(FadeIn(glow(wall[3][0], Q_COL)), Indicate(wall[3], color=Q_COL, scale_factor=1.25),
               page.animate.set_fill(MAMBA, 0.4), tiles.animate.set_fill(opacity=0.9), run_time=1.4)


class S01Intro(Deck):
    def construct(self):
        animate_title(self)
        cast = build_cast()
        ribbon = animate_story(self, cast)
        self.pause()
        wall, page, tiles = animate_memories(self, cast, ribbon)
        self.pause()
        animate_question(self, wall, page, tiles)
        self.pause(1.4)
        self.clear_all()
