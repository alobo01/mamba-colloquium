"""Slide 10 · MoE as a hospital: a triage nurse (router) opens 2 of 8 specialist rooms per patient (token).

Clicks:  ▶1 building + router   ▶2 three patients routed, gating formula   ▶3 built vs open
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

N_EXPERTS = 8
PICKS = [([1, 4], "Leo"), ([2, 6], "key"), ([0, 4], "bridge")]


def build_hospital() -> tuple:
    """Eight rooms, each with its own symbol, inside one building. Returns (building, rooms)."""
    building = card(12.9, 2.9, MOE, 0.05, radius=0.3, sw=2).move_to(UP * 0.95)
    rooms = VGroup(*[card(1.25, 1.7, c, 0.14, radius=0.16, sw=2.5) for c in ROOM]).arrange(RIGHT, buff=0.24).move_to(building)
    symbols = [Circle(radius=0.2), Triangle().scale(0.22), Square(0.36), Star(5, outer_radius=0.24, inner_radius=0.11),
               RegularPolygon(6).scale(0.22), Circle(radius=0.2), Triangle().scale(0.22).rotate(PI), Square(0.36).rotate(PI / 4)]
    for r, c, sym in zip(rooms, ROOM, symbols):
        r.add(sym.set_stroke(c, 3).set_fill(c, 0.25).move_to(r))
    return building, rooms


def build_router() -> VGroup:
    """The triage nurse: a ring with a medical plus. VGroup(badge, label)."""
    plus = VGroup(RoundedRectangle(width=0.16, height=0.52, corner_radius=0.04, stroke_width=0, fill_color=MOE, fill_opacity=1),
                  RoundedRectangle(width=0.52, height=0.16, corner_radius=0.04, stroke_width=0, fill_color=MOE, fill_opacity=1))
    badge = VGroup(Circle(radius=0.5, stroke_color=MOE, stroke_width=4, fill_color=MOE, fill_opacity=0.2), plus).move_to(LEFT * 3.2 + DOWN * 1.35)
    return VGroup(badge, T("router", 22, MOE, bold=True).next_to(badge, DOWN, buff=0.15))


def gate_eq() -> MathTex:
    """y = Σ_{i∈top-k} g_i E_i(x), router weights and experts in gold."""
    return ceq("y", "=", r"\sum_{i\in\text{top-}k}", ("g_i", MOE), (r"\,E_i", MOE), r"(", "x", ")", size=38)


def animate_routing(scene: Deck, rooms: VGroup, nurse: VGroup):
    """Each token is sent to two rooms (line width = gate weight); the others dim."""
    for r in rooms:
        r.save_state()
    for idx, (pk, word) in enumerate(PICKS):
        pt = chip(word, TOKEN, 22).move_to(LEFT * 6.0 + DOWN * 1.35)
        scene.play(pt.animate.next_to(nurse[0], LEFT, buff=0.2), run_time=0.5)
        lines = VGroup(*[Line(nurse[0].get_top(), rooms[j].get_bottom(), color=ROOM[j], stroke_width=2 + 6 * w) for j, w in zip(pk, (0.6, 0.4))])
        scene.play(*[Create(l) for l in lines], *[rooms[j].animate.set_fill(ROOM[j], 0.55) for j in pk],
                   *[rooms[j].animate.fade(0.7) for j in range(N_EXPERTS) if j not in pk], run_time=0.8)
        if idx == 0:
            scene.play(Write(gate_eq().move_to(RIGHT * 3.6 + DOWN * 1.45)))
        scene.play(FadeOut(lines), FadeOut(pt), *[Restore(r) for r in rooms], run_time=0.5)


class S10MoE(Deck):
    def construct(self):
        self.title("MoE: a hospital with specialist rooms", MOE)
        building, rooms = build_hospital()
        self.play(FadeIn(building), LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in rooms], lag_ratio=0.08), run_time=1.5)
        nurse = build_router()
        self.play(FadeIn(nurse))
        self.pause()
        animate_routing(self, rooms, nurse)
        self.pause()
        self.play(FadeIn(pill("8 rooms built", MOE, 24).move_to(LEFT * 3.2 + DOWN * 3.05), shift=UP * 0.2),
                  FadeIn(pill("only 2 open per token", MAMBA2, 24, filled=True).move_to(RIGHT * 3.4 + DOWN * 3.05), shift=UP * 0.2))
        self.pause(1.4)
        self.clear_all()
