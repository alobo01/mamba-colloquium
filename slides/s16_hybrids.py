"""Slide 16 · Neither pure model is cheap: function composition and why hybrids win.

Cooper, Diakonikolas, Ma & Sala 2026, "Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models"
(arXiv 2603.08859): Thm 4.2 (pure-model lower bounds), Thm 4.3 (2-layer hybrid), Fig. 4 (selective copy).

Clicks:  ▶1 the task: selective copy   ▶2 pure SSM bound   ▶3 pure Transformer bound
         ▶4 the hybrid trick on the row   ▶5 measured accuracy at ~2k parameters
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

TOKENS = ["#3", "g", "h", "k", "…", "w", "b", "d", "f", "?"]
CONTROL, ANSWER, ASK = 0, 6, 9
HYB = V_COL
XS = [-4.75, 0, 4.75]
RESULTS = [("pure Transformer", 0.352, ATTN), ("pure SSM", 0.305, STATE), ("Transformer → SSM", 0.433, FADE), ("SSM → Transformer", 0.999, HYB)]


def build_row() -> VGroup:
    """The selective-copy prompt: a number token far back, then filler, then the question mark."""
    row = VGroup(*[chip(t, MOE if i == CONTROL else (Q_COL if i == ASK else TOKEN), 24) for i, t in enumerate(TOKENS)])
    return row.arrange(RIGHT, buff=0.22).move_to(UP * 1.9)


def animate_task(scene: Deck, row: VGroup) -> VGroup:
    """Show the rule: output the token 3 positions back from the end, 3 given by the last number. Returns the labels."""
    scene.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in row], lag_ratio=0.08), run_time=1.2)
    look = CurvedArrow(row[ASK].get_bottom() + DOWN * 0.05, row[ANSWER].get_bottom() + DOWN * 0.05, angle=-1.2, color=Q_COL, stroke_width=4)
    scene.play(Create(look), Indicate(row[CONTROL], color=MOE), row[ANSWER].animate.set_color(Q_COL))
    labels = VGroup(T("control: the last number", 18, MOE).next_to(row[CONTROL], UP, buff=0.2),
                    T("answer: 3 back", 18, Q_COL).next_to(row[ANSWER], UP, buff=0.2))
    scene.play(FadeIn(labels))
    scene.wait(0.5)
    scene.play(FadeOut(look))
    return labels


def build_card(x: float, title: str, color, eq: MathTex, note: str) -> VGroup:
    """Lower card with heading, bound and one-line meaning."""
    fr = card(4.4, 2.9, color, 0.07).move_to([x, -1.3, 0])
    head = T(title, 22, color, bold=True).move_to(fr.get_top() + DOWN * 0.38)
    eq.move_to(fr.get_center() + DOWN * 0.05)
    if eq.width > 4.0:
        eq.scale_to_fit_width(4.0)
    sub = T(note, 16, FADE).move_to(fr.get_bottom() + UP * 0.42)
    return VGroup(fr, head, eq, sub)


def ssm_card() -> VGroup:
    return build_card(XS[0], "pure SSM", STATE,
                      ceq(r"\sum_i", (r"\log|S_i|", STATE), r"\ \ge\ ", "N", r"\log M", size=36),
                      "state must hold the last N tokens")


def tf_card() -> VGroup:
    return build_card(XS[1], "pure Transformer", ATTN,
                      ceq(r"\sum_i", ("W_i", ATTN), r"\ \ge\ ", r"\Omega(L)", size=36),
                      "windows must reach back to #3")


def hybrid_card() -> VGroup:
    return build_card(XS[2], "hybrid: SSM → attention", HYB,
                      ceq((r"|S|", STATE), "=", r"O(|\mathcal V|)", r",\quad", ("W", ATTN), "=", "O(N)", size=36),
                      "polylog parameters, small memory")


def animate_window(scene: Deck, row: VGroup) -> Mobject:
    """A full-length attention window bracket under the whole row."""
    br = BraceBetweenPoints(row[CONTROL].get_corner(DL) + DOWN * 0.1, row[ASK].get_corner(DR) + DOWN * 0.1, color=ATTN)
    scene.play(GrowFromCenter(br))
    return br


def animate_hybrid(scene: Deck, row: VGroup, window: Mobject, labels: VGroup):
    """The SSM layer stamps '3' under every later token; attention then only needs a 3-token window."""
    badges = VGroup(*[pill("3", MOE, 14) .scale(0.8).next_to(row[i], DOWN, buff=0.16) for i in range(CONTROL + 1, len(TOKENS))])
    scene.play(FadeOut(window))
    scene.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in badges], lag_ratio=0.12), run_time=1.5)
    short = BraceBetweenPoints(row[ASK].get_corner(UR) + UP * 0.1, row[ANSWER].get_corner(UL) + UP * 0.1, color=ATTN)
    look = CurvedArrow(row[ASK].get_top() + UP * 0.45, row[ANSWER].get_top() + UP * 0.45, angle=1.0, color=Q_COL, stroke_width=4)
    scene.play(FadeOut(labels), GrowFromCenter(short))
    scene.play(Create(look), Flash(row[ANSWER], color=Q_COL))


def animate_results(scene: Deck):
    """Selective-copy accuracy at about 2,000 parameters (Fig. 4)."""
    base_y, h = -2.6, 3.6
    xs = [-3.9, -1.3, 1.3, 3.9]
    grp = VGroup()
    for (name, acc, c), x in zip(RESULTS, xs):
        b = Rectangle(width=1.2, height=acc * h, stroke_width=0, fill_color=c, fill_opacity=0.9).move_to([x, base_y + acc * h / 2, 0])
        grp.add(VGroup(b, T(f"{acc:.2f}", 20, c, bold=True).next_to(b, UP, buff=0.1),
                       T(name, 17, c, bold=True).move_to([x, base_y - 0.3, 0])))
    head = T("selective copy accuracy · ≈ 2k parameters", 22, FADE).move_to(UP * 2.1)
    scene.play(FadeIn(head), LaggedStart(*[GrowFromEdge(g[0], DOWN) for g in grp], lag_ratio=0.15),
               LaggedStart(*[FadeIn(VGroup(g[1], g[2])) for g in grp], lag_ratio=0.15), run_time=1.6)
    scene.play(FadeIn(glow(grp[3][0], HYB, 3)),
               FadeIn(pill("pure models need ≈ 6× the parameters", HYB, 18).move_to(LEFT * 1.5 + UP * 1.5)))


class S16Hybrids(Deck):
    def construct(self):
        self.title("Neither pure model is cheap", HYB)
        row = build_row()
        labels = animate_task(self, row)
        self.pause()
        cards = [ssm_card(), tf_card(), hybrid_card()]
        self.play(FadeIn(cards[0], shift=UP * 0.2))
        self.pause()
        self.play(FadeIn(cards[1], shift=UP * 0.2))
        window = animate_window(self, row)
        self.pause()
        self.play(FadeIn(cards[2], shift=UP * 0.2))
        animate_hybrid(self, row, window, labels)
        self.pause()
        self.clear_body()
        animate_results(self)
        self.pause(1.4)
        self.clear_all()
