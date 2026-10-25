"""Slide 15 · Recall needs memory that grows: a phone book into a wall vs into a fixed notebook.

Evidence: Jelassi et al. 2024 "Repeat After Me" (copying needs state ∝ length; 2-layer Transformers copy),
Arora et al. 2023 "Zoology" (multi-query associative recall explains most of the quality gap).

Clicks:  ▶1 the pairs stream in   ▶2 the wall keeps all, the notebook overwrites   ▶3 the query   ▶4 the bound
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

PAIRS = [("A", "3"), ("B", "7"), ("C", "1"), ("D", "9"), ("E", "4"), ("F", "2")]
SLOTS = 4                       # the notebook holds 4 slots; pair i lands in slot i % 4
QUERY = 4                       # ask for "E": its slot also holds "A"
PANEL_Y = -0.55


def build_pair(key: str, val: str, size: float = 22) -> VGroup:
    """A phone-book entry: purple key chip glued to a yellow value chip."""
    return VGroup(chip(key, K_COL, size), chip(val, V_COL, size)).arrange(RIGHT, buff=0.04)


def build_panels() -> VGroup:
    left = card(6.4, 3.6, ATTN, 0.06).move_to([-3.6, PANEL_Y, 0])
    right = card(6.4, 3.6, STATE, 0.06).move_to([3.6, PANEL_Y, 0])
    heads = VGroup(T("Theo: keep every card", 22, ATTN, bold=True).move_to(left.get_top() + DOWN * 0.35),
                   T("Manu & Max: one fixed notebook", 22, STATE, bold=True).move_to(right.get_top() + DOWN * 0.35))
    return VGroup(left, right, heads)


def build_notebook(panel) -> VGroup:
    """2×2 grid of slots inside the right panel."""
    slots = VGroup(*[card(1.6, 0.95, STATE, 0.1, radius=0.12) for _ in range(SLOTS)]).arrange_in_grid(2, 2, buff=0.2)
    return slots.move_to(panel.get_center() + UP * 0.05)


def animate_stream(scene: Deck) -> VGroup:
    """The six pairs arrive left to right along the top."""
    row = VGroup(*[build_pair(k, v) for k, v in PAIRS]).arrange(RIGHT, buff=0.35).move_to(UP * 1.85)
    scene.play(LaggedStart(*[FadeIn(p, shift=DOWN * 0.2) for p in row], lag_ratio=0.15), run_time=1.4)
    return row


def animate_store(scene: Deck, row: VGroup, panels: VGroup, slots: VGroup) -> tuple:
    """Copies go to the wall (all kept) and to the notebook (later pairs pile onto earlier slots).

    Returns (wall, writes) where writes[i] is the copy of pair i inside its slot.
    """
    wall = VGroup(*[build_pair(k, v, 20) for k, v in PAIRS]).arrange_in_grid(2, 3, buff=(0.35, 0.3))
    wall.move_to(panels[0].get_center() + DOWN * 0.2)
    writes = VGroup()
    for i, (k, v) in enumerate(PAIRS):
        w = build_pair(k, v, 20).scale(0.8).move_to(slots[i % SLOTS])
        if i >= SLOTS:                          # a second pair lands on an occupied slot: both smear
            w.shift(RIGHT * 0.12 + DOWN * 0.08)
            writes[i - SLOTS].shift(LEFT * 0.12 + UP * 0.08)
        writes.add(w)
    for i in range(len(PAIRS)):
        anims = [TransformFromCopy(row[i], wall[i]), TransformFromCopy(row[i], writes[i])]
        if i >= SLOTS:
            anims += [writes[i - SLOTS].animate.set_opacity(0.45), slots[i % SLOTS].animate.set_stroke(PROB)]
        scene.play(*anims, run_time=0.55)
    scene.play(writes[SLOTS:].animate.set_opacity(0.45))
    return wall, writes


def animate_query(scene: Deck, row: VGroup, wall: VGroup, slots: VGroup, panels: VGroup):
    """“E?” — the wall answers 4 exactly; the notebook slot holds 3 and 4 on top of each other."""
    q = chip("E ?", Q_COL, 26).move_to(UP * 1.85)
    scene.play(FadeOut(row), FadeIn(q, scale=0.8))
    hit = wall[QUERY]
    scene.play(Indicate(hit, color=Q_COL, scale_factor=1.2), FadeIn(glow(hit[1][0], Q_COL, 3)),
               Indicate(slots[QUERY % SLOTS], color=PROB, scale_factor=1.08))
    good = pill("4", MAMBA2, 24, filled=True).move_to(panels[0].get_bottom() + UP * 0.38)
    bad = pill("3 or 4 ?", PROB, 22).move_to(panels[1].get_bottom() + UP * 0.38)
    scene.play(FadeIn(good, shift=UP * 0.2), FadeIn(bad, shift=UP * 0.2))


def animate_bound(scene: Deck):
    """Exact recall of n pairs needs at least n log|V| bits of state."""
    eq = ceq((r"\text{bits of state}", STATE), r"\ \ge\ ", "n", r"\,\log|\mathcal V|", size=42).move_to(DOWN * 3.0)
    scene.play(Write(eq))


class S15Recall(Deck):
    def construct(self):
        self.title("Recall needs memory that grows", ATTN)
        row = animate_stream(self)
        self.pause()
        panels = build_panels()
        slots = build_notebook(panels[1])
        self.play(FadeIn(panels), FadeIn(slots))
        wall, writes = animate_store(self, row, panels, slots)
        self.pause()
        animate_query(self, row, wall, slots, panels)
        self.pause()
        animate_bound(self)
        self.pause(1.4)
        self.clear_all()
